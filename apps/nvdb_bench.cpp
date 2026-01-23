#include "nvdb/vector_dataset.h"
#include "nvdb/flat_index.h"
#include "nvdb/flat_index_omp.h"
#include "nvdb/flat_index_async.h"
#include "nvdb/flat_index_pool.h"
#include "nvdb/score_dispatch.h"
#include "nvdb/topK.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>

#if NVDB_HAS_OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <cstdlib>
#include "nvdb/simd_dot.h"

static double ms_since(const std::chrono::steady_clock::time_point& t0,
                       const std::chrono::steady_clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static inline const void* base_row_ptr_for_prefetch(const nvdb::VectorDataset& base, uint64_t i) {
  // Return address of row i payload (f32 or f16)
  if (base.dtype() == static_cast<uint32_t>(nvdb::DType::Float32)) {
    return (const void*)base.vector_ptr_f32(i);
  } else {
    return (const void*)base.vector_ptr_f16(i);
  }
}

static inline void do_prefetch(const nvdb::VectorDataset& base, uint64_t i, int prefetch_dist) {
  if (prefetch_dist <= 0) return;
  const uint64_t pf = i + (uint64_t)prefetch_dist;
  if (pf >= base.count()) return;
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(base_row_ptr_for_prefetch(base, pf), 0, 1);
#endif
}

static void batched_scan_omp_or_st(
    const nvdb::VectorDataset& base,
    const nvdb::VectorDataset& query,
    uint32_t k,
    const std::string& mode,   // "st" or "omp"
    int threads,               // for omp_set_num_threads
    int batch_q,
    int tile_vecs,
    int prefetch_dist,
    std::vector<double>& lat_ms,
    volatile float& sink
) {
  nvdb::ensure_supported_base_dtype(base);
  const uint32_t dt  = base.dtype();
  const uint32_t dim = base.dim();
  const uint64_t N   = base.count();
  const uint64_t Q   = query.count();

#if NVDB_HAS_OPENMP
  if (mode == "omp" && threads > 0) {
    omp_set_num_threads(threads);
  }
#endif

  // process queries in batches
  for (uint64_t qb = 0; qb < Q; qb += (uint64_t)batch_q) {
    const int b = (int)std::min<uint64_t>((uint64_t)batch_q, Q - qb);

    // per-query topk buffers for this batch
    std::vector<nvdb::TopKBuffer> topks;
    topks.reserve(b);
    for (int i = 0; i < b; ++i) topks.emplace_back(k);

    // pointers to query vectors
    std::vector<const float*> qptrs(b);
    for (int i = 0; i < b; ++i) qptrs[i] = query.vector_ptr_f32(qb + i);

    auto t0 = std::chrono::steady_clock::now();

    // base scan with tiling
    const uint64_t T = (tile_vecs > 0) ? (uint64_t)tile_vecs : 1024ULL;

#if NVDB_HAS_OPENMP
    if (mode == "omp") {
      // Each thread has local topk buffers, then merge
      const int nthreads = omp_get_max_threads();
      std::vector<std::vector<nvdb::TopKBuffer>> locals(nthreads);
      for (int t = 0; t < nthreads; ++t) {
        locals[t].reserve(b);
        for (int i = 0; i < b; ++i) locals[t].emplace_back(k);
      }

      #pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        auto& ltopk = locals[tid];

        #pragma omp for schedule(static)
        for (uint64_t base0 = 0; base0 < N; base0 += T) {
          const uint64_t base1 = std::min<uint64_t>(base0 + T, N);
          for (uint64_t vi = base0; vi < base1; ++vi) {
            // compute score for each query in batch, reuse the base vector load path
            /*for (int qi = 0; qi < b; ++qi) {
              float s = nvdb::score_query_base_at(base, qptrs[qi], vi, dim, dt);
              ltopk[qi].consider(vi, s);
            }*/
            //with prefetch
            do_prefetch(base, vi, prefetch_dist);

            for (int qi = 0; qi < b; ++qi) {
                float s = nvdb::score_query_base_at(base, qptrs[qi], vi, dim, dt);
                ltopk[qi].consider(vi, s);
            }
          }
        }
      } // omp parallel

      // merge thread-local topk -> global topk for each query
      for (int qi = 0; qi < b; ++qi) {
        for (int t = 0; t < (int)locals.size(); ++t) {
          topks[qi].merge_from(locals[t][qi].finalize_sorted_desc());
        }
      }
    } else
#endif
    {
      // single-thread baseline batch scan
      for (uint64_t base0 = 0; base0 < N; base0 += T) {
        const uint64_t base1 = std::min<uint64_t>(base0 + T, N);
        for (uint64_t vi = base0; vi < base1; ++vi) {
          for (int qi = 0; qi < b; ++qi) {
            float s = nvdb::score_query_base_at(base, qptrs[qi], vi, dim, dt);
            topks[qi].consider(vi, s);
          }
        }
      }
    }

    // finalize batch and record stats per query
    auto t1 = std::chrono::steady_clock::now();
    const double batch_ms = ms_since(t0, t1);


    // Record batch-level latency (one sample per batch)
    lat_ms.push_back(batch_ms);

    // Still consume sink per query to prevent over-optimization
    for (int qi = 0; qi < b; ++qi) {
    auto res = topks[qi].finalize_sorted_desc();
    if (!res.empty()) sink += res[0].score;
    }
  }
}

static void batched_scan_pool_threads(
    const nvdb::VectorDataset& base,
    const nvdb::VectorDataset& query,
    uint32_t k,
    int threads,
    int batch_q,
    int tile_vecs,
    int prefetch_dist,
    std::vector<double>& lat_ms,
    volatile float& sink
) {
  nvdb::ensure_supported_base_dtype(base);

  const uint32_t dt  = base.dtype();
  const uint32_t dim = base.dim();
  const uint64_t N   = base.count();
  const uint64_t Q   = query.count();

  const int Tn = std::max(1, threads);
  const uint64_t T = (tile_vecs > 0) ? (uint64_t)tile_vecs : 1024ULL;

  for (uint64_t qb = 0; qb < Q; qb += (uint64_t)batch_q) {
    const int b = (int)std::min<uint64_t>((uint64_t)batch_q, Q - qb);

    // global topk buffers for this batch
    std::vector<nvdb::TopKBuffer> topks;
    topks.reserve(b);
    for (int i = 0; i < b; ++i) topks.emplace_back(k);

    // query pointers
    std::vector<const float*> qptrs(b);
    for (int i = 0; i < b; ++i) qptrs[i] = query.vector_ptr_f32(qb + i);

    auto t0 = std::chrono::steady_clock::now();

    // thread-local topk buffers: locals[tid][qi]
    std::vector<std::vector<nvdb::TopKBuffer>> locals(Tn);
    for (int t = 0; t < Tn; ++t) {
      locals[t].reserve(b);
      for (int i = 0; i < b; ++i) locals[t].emplace_back(k);
    }

    std::vector<std::thread> workers;
    workers.reserve(Tn);

    for (int tid = 0; tid < Tn; ++tid) {
      workers.emplace_back([&, tid]() {
        auto& ltopk = locals[tid];

        // Interleave tiles across threads for load balance:
        // tid handles tiles starting at base0 = tid*T, then += Tn*T
        for (uint64_t base0 = (uint64_t)tid * T; base0 < N; base0 += (uint64_t)Tn * T) {
          const uint64_t base1 = std::min<uint64_t>(base0 + T, N);
          for (uint64_t vi = base0; vi < base1; ++vi) {
            /*for (int qi = 0; qi < b; ++qi) {
              float s = nvdb::score_query_base_at(base, qptrs[qi], vi, dim, dt);
              ltopk[qi].consider(vi, s);
            }*/
            // with prefetch
            do_prefetch(base, vi, prefetch_dist);

            for (int qi = 0; qi < b; ++qi) {
                float s = nvdb::score_query_base_at(base, qptrs[qi], vi, dim, dt);
                ltopk[qi].consider(vi, s);
            }
          }
        }
      });
    }

    for (auto& th : workers) th.join();

    // merge locals -> global
    for (int qi = 0; qi < b; ++qi) {
      for (int t = 0; t < Tn; ++t) {
        topks[qi].merge_from(locals[t][qi].finalize_sorted_desc());
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    const double batch_ms = ms_since(t0, t1);
    // Record batch-level latency (one sample per batch)
    lat_ms.push_back(batch_ms);

    // Still consume sink per query to prevent over-optimization
    for (int qi = 0; qi < b; ++qi) {
    auto res = topks[qi].finalize_sorted_desc();
    if (!res.empty()) sink += res[0].score;
    }
  }
}


int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: nvdb_bench <base.vecbin> <query.vecbin> <k> [mode=st] [threads=0] [warmup=5] [batch_q=1] [tile_vecs=1024] [prefetch_dist=0]\n";

        return 1;
    }

    const std::string base_path  = argv[1];
    const std::string query_path = argv[2];
    const uint32_t k = static_cast<uint32_t>(std::stoul(argv[3]));

    const std::string mode = (argc >= 5) ? std::string(argv[4]) : "st";
    int threads = (argc >= 6) ? std::stoi(argv[5]) : 0;
    const int warmup = (argc >= 7) ? std::stoi(argv[6]) : 5;
    // [batch_q=1] [tile_vecs=1024] 
    const int batch_q= (argc >= 8) ? std::stoi(argv[7]) : 1;
    const int tile_vecs= (argc >= 9) ? std::stoi(argv[8]) : 1024;
    const int prefetch_dist = (argc >= 10) ? std::stoi(argv[9]) : 0;

    if (threads <= 0) threads = (int)std::thread::hardware_concurrency();
    int report_threads = threads;
    #if NVDB_HAS_OPENMP
    if (mode == "omp") report_threads = omp_get_max_threads();
    #endif
    std::cout << "mode=" << mode << " threads=" << report_threads << "\n";

    const char* fs = std::getenv("NVDB_FORCE_SCALAR");
    if (fs && fs[0] == '1') nvdb::set_force_scalar(true);


    nvdb::VectorDataset base, query;
    base.load(base_path);
    query.load(query_path);

    if (base.dim() != query.dim()) {
        std::cerr << "Dim mismatch: base.dim=" << base.dim()
                << ", query.dim=" << query.dim() << "\n";
        return 2;
    }

    nvdb::FlatIndex st_index(&base);
    nvdb::FlatIndexOMP omp_index(&base);
    nvdb::FlatIndexAsync async_index(&base);

    std::unique_ptr<nvdb::FlatIndexPool> pool_index;
    if (mode == "pool") {
        pool_index = std::make_unique<nvdb::FlatIndexPool>(&base, threads);
    }


    auto run_query = [&](const float* qvec) -> std::vector<nvdb::SearchResult> {
    if (mode == "st")    return st_index.search_topk_dot(qvec, k);
    if (mode == "omp")   return omp_index.search_topk_dot(qvec, k);
    if (mode == "async") return async_index.search_topk_dot(qvec, k, threads);
    if (mode == "pool")  return pool_index->search_topk_dot(qvec, k);
    throw std::runtime_error("Unknown mode: " + mode);
    };

    std::cout << "Base count=" << base.count() << " dim=" << base.dim()
                << " | Query count=" << query.count()
                << " | k=" << k << " | warmup=" << warmup << "\n";

    // warmup
    for (int i = 0; i < warmup; ++i) {
        const float* q0 = query.vector_ptr(0);
        //auto r = index.search_topk_dot(q0, k);
        auto r = run_query(q0);
        (void)r;
    }

    


    // record per-query latency
    std::vector<double> lat_ms;
    lat_ms.reserve(static_cast<size_t>(query.count()));


    volatile float sink = 0.0f;

    auto t_all0 = std::chrono::steady_clock::now();
    
    //  query batching
    if (batch_q > 1) {
    if (mode == "st" || mode == "omp") {
        //batched_scan_omp_or_st(base, query, k, mode, threads, batch_q, tile_vecs, lat_ms, sink);
        batched_scan_omp_or_st(base, query, k, mode, threads, batch_q, tile_vecs, prefetch_dist, lat_ms, sink);

    } else if (mode == "pool") {
        //batched_scan_pool_threads(base, query, k, threads, batch_q, tile_vecs, lat_ms, sink);
        batched_scan_pool_threads(base, query, k, threads, batch_q, tile_vecs, prefetch_dist, lat_ms, sink);

    } else {
        throw std::runtime_error("batch_q>1 supported only for mode=st/omp/pool (bench-side batching).");
    }
    } else {
    //  benchmark all queries 
    for (uint64_t qi = 0; qi < query.count(); ++qi) {
        auto t0 = std::chrono::steady_clock::now();
        const float* q = query.vector_ptr_f32(qi);
        auto topk = run_query(q);
        auto t1 = std::chrono::steady_clock::now();

        lat_ms.push_back(ms_since(t0, t1));
        if (!topk.empty()) sink += topk[0].score;
    }
    }


    auto t_all1 = std::chrono::steady_clock::now();
    const double total_ms = ms_since(t_all0, t_all1);

    

    // stats
    std::sort(lat_ms.begin(), lat_ms.end());
    auto pct = [&](double p) -> double {
    if (lat_ms.empty()) return 0.0;
    const double pos = (p / 100.0) * (lat_ms.size() - 1);
    const size_t i0 = (size_t)std::floor(pos);
    const size_t i1 = std::min(i0 + 1, lat_ms.size() - 1);
    const double frac = pos - double(i0);
    return lat_ms[i0] * (1.0 - frac) + lat_ms[i1] * frac;
    };

    if (batch_q > 1) std::cout << "batch_samples=" << lat_ms.size() << "\n";

    const double avg = total_ms / double(query.count());
    const double qps = (double(query.count()) * 1000.0) / total_ms;

    std::cout << std::fixed << std::setprecision(3);

    const double avg_query = total_ms / double(query.count());
    const double qps_query = (double(query.count()) * 1000.0) / total_ms;

    std::cout << "Total:     " << total_ms << " ms\n";
    std::cout << "Avg_query: " << avg_query << " ms/query  (" << qps_query << " QPS)\n";

    if (batch_q > 1) {
    const int num_batches = int((query.count() + uint64_t(batch_q) - 1) / uint64_t(batch_q));
    const double avg_batch = total_ms / double(num_batches);
    const double bps = (double(num_batches) * 1000.0) / total_ms;

    std::cout << "Avg_batch: " << avg_batch << " ms/batch  (" << bps << " batches/s)\n";

    // NOTE: lat_ms is batch-level latency samples in Route A
    std::cout << "batch_p50: " << pct(50) << " ms\n";
    std::cout << "batch_p95: " << pct(95) << " ms\n";
    std::cout << "batch_p99: " << pct(99) << " ms\n";
    } else {
    // NOTE: lat_ms is per-query latency samples
    std::cout << "p50:       " << pct(50) << " ms\n";
    std::cout << "p95:       " << pct(95) << " ms\n";
    std::cout << "p99:       " << pct(99) << " ms\n";
    }

    std::cout << "sink=" << sink << "\n";

    /*const double bytes_per_query =
    double(base.count()) * double(base.dim()) * double(nvdb::bytes_per_elem(base.dtype()));*/
    const uint32_t dtype = base.dtype();
    const double bytes_per_query =
        double(nvdb::bytes_for_payload_and_aux(base.count(), base.dim(), dtype));

    const double payload_equiv_bw = (avg_query > 0.0) ? (bytes_per_query * 1e-6 / avg_query) : 0.0;

    std::cout << std::setprecision(0);
    std::cout << "bytes_per_query=" << bytes_per_query << "\n";
    std::cout << std::setprecision(3);
    std::cout << "payload_equiv_bandwidth_GBps=" << payload_equiv_bw << "\n";
    if (batch_q > 1) std::cout << "(note) payload_equiv_bandwidth_GBps may exceed DRAM peak due to cache reuse\n";
    std::cout << "batch_q=" << batch_q << " tile_vecs=" << tile_vecs << " prefetch_dist=" << prefetch_dist << "\n";





    return 0;
}
