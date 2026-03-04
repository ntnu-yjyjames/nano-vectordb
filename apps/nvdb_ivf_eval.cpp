#include "nvdb/vector_dataset.h"
#include "nvdb/flat_index_omp.h"
#include "nvdb/to_f32_row.h"
#include "nvdb/gtbin_format.h"
#include "nvdb/mmap_file.h"
#include "nvdb/gtbin_format.h"
#include "nvdb/mmap_file.h"
#include "nvdb/cuda_refine.h"


#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_io.h>
#include <faiss/IndexPreTransform.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <memory>
#include <queue>


#include <cuda_runtime.h>


static double ms_since(const std::chrono::steady_clock::time_point& t0,
                       const std::chrono::steady_clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int getenv_int(const char* k, int defv) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : defv;
}

static std::string getenv_str(const char* k, const std::string& defv) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : defv;
}

static void cuda_ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
    std::exit(1);
  }
}


struct LatStats {
  double avg_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double qps = 0.0;
};

static LatStats compute_lat_stats(std::vector<double> v) {
  LatStats s;
  if (v.empty()) return s;

  std::sort(v.begin(), v.end());

  auto pct = [&](double p) -> double {
    if (v.empty()) return 0.0;
    const double pos = (p / 100.0) * (v.size() - 1);
    const size_t i0 = (size_t)pos;
    const size_t i1 = std::min(i0 + 1, v.size() - 1);
    const double frac = pos - double(i0);
    return v[i0] * (1.0 - frac) + v[i1] * frac;
  };

  double sum = 0.0;
  for (double x : v) sum += x;
  s.avg_ms = sum / (double)v.size();
  s.qps = (s.avg_ms > 0.0) ? (1000.0 / s.avg_ms) : 0.0;

  s.p50_ms = pct(50);
  s.p95_ms = pct(95);
  s.p99_ms = pct(99);
  return s;
}


struct PairIdDist {
  float dist;
  uint32_t id;
};

// 取出 pred top-k (k 很小)；buf 會被覆寫
static inline void select_topk_by_dist_inplace(
    PairIdDist* buf, int n, int k) {
  if (n <= 0 || k <= 0) return;
  if (k > n) k = n;
  auto cmp = [](const PairIdDist& a, const PairIdDist& b) { return a.dist < b.dist; };

  // 把第 k 小的元素放到 buf[k-1]，前 k 個為 k 個最小（未排序）
  std::nth_element(buf, buf + (k - 1), buf + n, cmp);
  // 前 k 個排序（方便後續 stable output / debug）
  std::sort(buf, buf + k, cmp);
}


static size_t intersection_size(const std::vector<uint64_t>& gt_ids,
                                const faiss::idx_t* ann_ids,
                                int k) {
  std::unordered_set<uint64_t> s(gt_ids.begin(), gt_ids.end());
  size_t hit = 0;
  for (int i = 0; i < k; ++i) {
    if (ann_ids[i] < 0) continue;
    hit += (s.find((uint64_t)ann_ids[i]) != s.end()) ? 1 : 0;
  }
  return hit;
}
static size_t intersection_size_vec(const std::vector<uint64_t>& gt_ids,
                                    const std::vector<uint64_t>& pred_ids) {
  std::unordered_set<uint64_t> s(gt_ids.begin(), gt_ids.end());
  size_t hit = 0;
  for (auto x : pred_ids) hit += (s.find(x) != s.end()) ? 1 : 0;
  return hit;
}

static inline size_t hit_k_small(const uint32_t* gt, const faiss::idx_t* ann, int k) {
  size_t hit = 0;
  for (int i = 0; i < k; ++i) {
    const faiss::idx_t a = ann[i];
    if (a < 0) continue;
    for (int j = 0; j < k; ++j) {
      if ((uint32_t)a == gt[j]) { hit++; break; }
    }
  }
  return hit;
}

struct NodeL2 {
  float dist;
  uint64_t id;
};

struct MaxCmp {
  bool operator()(const NodeL2& a, const NodeL2& b) const {
    return a.dist < b.dist; // max-heap by dist
  }
};

static inline float l2_sqr_f32(const float* a, const float* b, uint32_t d) {
  // naive; good enough for ground-truth
  double s = 0.0;
  for (uint32_t i = 0; i < d; ++i) {
    const double diff = double(a[i]) - double(b[i]);
    s += diff * diff;
  }
  return (float)s;
}

// Exact top-k under L2^2 (streaming scan; base may be fp16/fp32/int8 via base_row_to_f32)
static std::vector<uint64_t> exact_topk_l2_ids(const nvdb::VectorDataset& base,
                                               const float* q,
                                               uint32_t k) {
  const uint32_t d = base.dim();
  const uint64_t N = base.count();
  if (k == 0) return {};
  if (k > N) k = (uint32_t)N;

  // max-heap of size k (largest dist on top)
  std::priority_queue<NodeL2, std::vector<NodeL2>, MaxCmp> heap;
  std::vector<float> row(d);

  for (uint64_t i = 0; i < N; ++i) {
    nvdb::base_row_to_f32(base, i, row.data());
    float dist = l2_sqr_f32(q, row.data(), d);

    if (heap.size() < k) {
      heap.push({dist, i});
    } else if (dist < heap.top().dist) {
      heap.pop();
      heap.push({dist, i});
    }
  }

  std::vector<uint64_t> ids;
  ids.reserve(k);
  while (!heap.empty()) {
    ids.push_back(heap.top().id);
    heap.pop();
  }
  // heap pops worst->best (largest dist removed last), so reverse to best->worst
  std::reverse(ids.begin(), ids.end());
  return ids;
}

static std::vector<uint64_t> refine_topk_l2_ids(const nvdb::VectorDataset& base,
                                                const float* q,
                                                const faiss::idx_t* cand_ids,
                                                int cand_k,
                                                uint32_t k) {
  const uint32_t d = base.dim();
  if (k == 0) return {};
  if (cand_k <= 0) return {};
  if (k > (uint32_t)cand_k) k = (uint32_t)cand_k;

  // max-heap by dist (keep k best)
  std::priority_queue<NodeL2, std::vector<NodeL2>, MaxCmp> heap;
  std::vector<float> row(d);

  for (int i = 0; i < cand_k; ++i) {
    faiss::idx_t id = cand_ids[i];
    if (id < 0) continue;
    nvdb::base_row_to_f32(base, (uint64_t)id, row.data());
    float dist = l2_sqr_f32(q, row.data(), d);

    if (heap.size() < k) heap.push({dist, (uint64_t)id});
    else if (dist < heap.top().dist) { heap.pop(); heap.push({dist, (uint64_t)id}); }
  }

  std::vector<uint64_t> ids;
  ids.reserve(k);
  while (!heap.empty()) { ids.push_back(heap.top().id); heap.pop(); }
  std::reverse(ids.begin(), ids.end()); // best->worst
  return ids;
}


int main(int argc, char** argv) {
  if (argc < 5) {
      std::cerr
      << "Usage: nvdb_ivf_eval <base.vecbin> <ivf.faiss> <query.vecbin> <k>\n"
      << "Env:\n"
      << "  IVF_NPROBE=16\n"
      << "  WARMUP=5\n"
      << "  EVAL_MODE=full|ann_only   (default: full)\n";
    return 1;
  }

  const std::string base_path  = argv[1];
  const std::string index_path = argv[2];
  const std::string query_path = argv[3];
  const int k = std::atoi(argv[4]);

  const int nprobe = getenv_int("IVF_NPROBE", 16);
  const int warmup = getenv_int("WARMUP", 5);
  const int refine_k = getenv_int("REFINE_K", 0);
  const int k_search = (refine_k > 0) ? std::max(refine_k, k) : k;
  const bool cuda_refine = (getenv_int("CUDA_REFINE", 0) != 0);

  std::cout << "REFINE_K=" << refine_k << " (k_search=" << k_search << ")\n";


  nvdb::VectorDataset base, query;
  base.load(base_path);
  query.load(query_path);

  if (query.dtype() != (uint32_t)nvdb::DType::Float32) {
    std::cerr << "Query must be float32 vecbin\n";
    return 2;
  }
  if (base.dim() != query.dim()) {
    std::cerr << "Dim mismatch: base.dim=" << base.dim()
              << " query.dim=" << query.dim() << "\n";
    return 3;
  }

  const uint32_t d = base.dim();
  const uint64_t Q = query.count();
  const std::string eval_mode = getenv_str("EVAL_MODE", "full");
  const bool ann_only = (eval_mode == "ann_only");
  const std::string exact_metric = getenv_str("EXACT_METRIC", "L2"); // L2 (default) or DOT

  const std::string gt_path = getenv_str("GT_PATH", "");
  nvdb::MmapFile gt_mm;
  const nvdb::GtBinHeader* gt_h = nullptr;
  const uint32_t* gt_payload = nullptr;

  if (!gt_path.empty() && !ann_only) {
    gt_mm.open_readonly(gt_path);
    if (gt_mm.size() < sizeof(nvdb::GtBinHeader)) {
      std::cerr << "GT file too small: " << gt_path << "\n";
      return 7;
    }
    gt_h = reinterpret_cast<const nvdb::GtBinHeader*>(gt_mm.data());
    if (gt_h->magic != nvdb::kGtMagic || gt_h->version != nvdb::kGtVersion) {
      std::cerr << "Bad GT header: " << gt_path << "\n";
      return 8;
    }
    if (gt_h->k != (uint32_t)k || gt_h->dim != d || gt_h->Q != Q || gt_h->N != base.count()) {
      std::cerr << "GT mismatch: (k/d/Q/N) expected=(" << k << "/" << d << "/" << Q << "/" << base.count()
                << ") got=(" << gt_h->k << "/" << gt_h->dim << "/" << gt_h->Q << "/" << gt_h->N << ")\n";
      return 9;
    }
    const size_t expect = sizeof(nvdb::GtBinHeader) + nvdb::gt_payload_bytes(Q, (uint32_t)k);
    if (gt_mm.size() != expect) {
      std::cerr << "GT size mismatch: expected=" << expect << " got=" << gt_mm.size() << "\n";
      return 10;
    }
    gt_payload = reinterpret_cast<const uint32_t*>(gt_mm.data() + sizeof(nvdb::GtBinHeader));
    std::cout << "GT_PATH=" << gt_path << " (cached ground truth enabled)\n";
  }
  if (cuda_refine && refine_k > 0 && exact_metric != "L2") {
    std::cerr << "CUDA_REFINE=1 currently supports EXACT_METRIC=L2 only\n";
    return 11;
  }

  // Load FAISS index
  faiss::Index* idx = faiss::read_index(index_path.c_str());
  if (!idx) {
    std::cerr << "Failed to read index\n";
    return 4;
  }
  faiss::Index* top = idx; // the index we will call search() on

  // If OPQ/PreTransform: unwrap to reach the IVF layer for nprobe/nlist
  faiss::Index* inner = idx;
  if (auto* ipt = dynamic_cast<faiss::IndexPreTransform*>(idx)) {
    inner = ipt->index; // underlying index (e.g., IVFPQ)
  }

  auto* ivf = dynamic_cast<faiss::IndexIVF*>(inner);
  if (!ivf) {
    std::cerr << "Index is not IVF (even after unwrapping pretransform)\n";
    return 5;
  }
  ivf->nprobe = nprobe;

  std::cout << "IVF nlist=" << ivf->nlist << " nprobe=" << ivf->nprobe
            << " | d=" << d << " k=" << k << " Q=" << Q
            << (dynamic_cast<faiss::IndexPreTransform*>(idx) ? " | pretransform=OPQ" : "")
            << "\n";





  // Warmup ANN-only
  for (int i = 0; i < warmup && Q > 0; ++i) {
    const float* q0 = query.vector_ptr_f32(0);
    std::vector<float> Dd_w(k_search);
    std::vector<faiss::idx_t> I_w(k_search);
    idx->search(1, q0, k_search, Dd_w.data(), I_w.data());
  }





  std::unique_ptr<nvdb::FlatIndexOMP> exact;
  if (!ann_only && exact_metric == "DOT") {
    exact = std::make_unique<nvdb::FlatIndexOMP>(&base);
  }
  if (!ann_only && exact_metric != "DOT" && exact_metric != "L2") {
    std::cerr << "Unknown EXACT_METRIC=" << exact_metric << " (use L2 or DOT)\n";
    return 6;
  }


  //std::vector<double> lat_ms;
  //lat_ms.reserve((size_t)Q);
  std::vector<double> ann_lat_ms;
  std::vector<double> total_lat_ms;
  ann_lat_ms.reserve((size_t)Q);
  total_lat_ms.reserve((size_t)Q);

  double refine_ms_total = 0.0;
  double refine_ms_per_q = 0.0;

  double refine_h2d_ms = 0.0;
  double refine_kernel_ms = 0.0;
  double refine_d2h_ms = 0.0;

  double recall_sum = 0.0;
  double select_ms_total = 0.0;
  double select_ms_per_q = 0.0;

  auto t_sel0 = std::chrono::steady_clock::now();

  volatile float sink = 0.0f;
  std::vector<float> Dd(k_search);
  std::vector<faiss::idx_t> I(k_search);

  const std::string pipeline = getenv_str("PIPELINE", "staged");
  const bool staged = (pipeline == "staged");

  std::vector<faiss::idx_t> I_all;          // Q * k_search
  std::vector<float> queries_all;           // Q * d (for CUDA refine)
  std::vector<uint32_t> cand_ids_all;       // Q * refine_k (for CUDA refine)

  I_all.resize((size_t)Q * (size_t)k_search);
  nvdb::CudaRefineTiming t{};
  if (staged) {
    // -------------------------
    // Stage A: ANN only (collect candidates + per-query ANN latency)
    // -------------------------
    for (uint64_t qi = 0; qi < Q; ++qi) {
      const float* qv = query.vector_ptr_f32(qi);

      auto t0 = std::chrono::steady_clock::now();
      idx->search(1, qv, k_search, Dd.data(), I.data());
      auto t1 = std::chrono::steady_clock::now();

      ann_lat_ms.push_back(ms_since(t0, t1));
      std::memcpy(I_all.data() + (size_t)qi * (size_t)k_search,
                  I.data(),
                  (size_t)k_search * sizeof(faiss::idx_t));
    }

    // -------------------------
    // Stage B: refine (optional)
    //  - If ann_only: skip
    //  - If REFINE_K == 0: no refine, recall computed on raw top-k (optional)
    // -------------------------
    const bool cuda_refine_warmup = (getenv_int("CUDA_REFINE_WARMUP", 1) != 0);

    
    if (!ann_only && refine_k > 0) {
      // --- CUDA refine path ---
      if (cuda_refine) {
        // prepare batched buffers
        queries_all.resize((size_t)Q * (size_t)d);
        cand_ids_all.resize((size_t)Q * (size_t)refine_k, 0xFFFFFFFFu);

        for (uint64_t qi = 0; qi < Q; ++qi) {
          const float* qv = query.vector_ptr_f32(qi);
          std::memcpy(queries_all.data() + (size_t)qi * (size_t)d,
                      qv, (size_t)d * sizeof(float));

          const faiss::idx_t* cand = I_all.data() + (size_t)qi * (size_t)k_search;
          for (int r = 0; r < refine_k; ++r) {
            faiss::idx_t id = cand[r];
            cand_ids_all[(size_t)qi * (size_t)refine_k + (size_t)r] =
              (id >= 0) ? (uint32_t)id : 0xFFFFFFFFu;
          }
        }

        const void* base_ptr = base.data_ptr();
        if (!base_ptr ||
            (base.dtype() != (uint32_t)nvdb::DType::Float16 &&
            base.dtype() != (uint32_t)nvdb::DType::Float32)) {
          std::cerr << "CUDA refine supports base dtype fp16/fp32 only (got dtype=" << base.dtype() << ")\n";
          return 12;
        }

        // optional warmup
        if (cuda_refine_warmup) {
          std::vector<float> tmp_dist;
          std::vector<uint32_t> tmp_ids;
          nvdb::CudaRefineTiming tmp_t{};
          nvdb::cuda_l2_topk_batch(
            base_ptr, base.dtype(), base.count(), base.dim(),
            queries_all.data(), cand_ids_all.data(),
            (uint32_t)Q, (uint32_t)refine_k,(uint32_t)k,
            tmp_ids, tmp_dist,&tmp_t);
        }
        
        std::vector<uint32_t> topk_ids;
        std::vector<float> topk_dist;
        //nvdb::CudaRefineTiming t{};
        nvdb::cuda_l2_topk_batch(
          base_ptr, base.dtype(), base.count(), base.dim(),
          queries_all.data(), cand_ids_all.data(),
          (uint32_t)Q, (uint32_t)refine_k,
          (uint32_t)k,
          topk_ids, topk_dist,
          &t
        );

        refine_ms_total   = t.total_ms;
        refine_ms_per_q   = refine_ms_total / double(Q);

        refine_h2d_ms     = t.h2d_ms;
        refine_kernel_ms  = t.kernel_ms;
        refine_d2h_ms     = t.d2h_ms;


        std::cout << "CUDA_REFINE=1 refine_ms_total=" << t.total_ms
                  << " (h2d=" << t.h2d_ms
                  << " kernel=" << t.kernel_ms
                  << " d2h=" << t.d2h_ms << " ms)"
                  << " avg=" << (t.total_ms / double(Q)) << " ms/query\n";


        // recall
        recall_sum = 0.0;
        for (uint64_t qi = 0; qi < Q; ++qi) {
          std::vector<uint64_t> pred_ids;
          pred_ids.reserve(k);
          for (int i = 0; i < k; ++i) {
            uint32_t id = topk_ids[(size_t)qi * k + i];
            if (id != 0xFFFFFFFFu) pred_ids.push_back((uint64_t)id);
          }

          if (gt_payload) {
            const uint32_t* gt = gt_payload + (size_t)qi * (size_t)k;
            std::vector<uint64_t> gt_ids; gt_ids.reserve(k);
            for (int j = 0; j < k; ++j) gt_ids.push_back((uint64_t)gt[j]);
            recall_sum += double(intersection_size_vec(gt_ids, pred_ids)) / double(k);
            sink += (float)gt[0];
          }
        }

        total_lat_ms.resize(ann_lat_ms.size());
        for (size_t i = 0; i < ann_lat_ms.size(); ++i) total_lat_ms[i] = ann_lat_ms[i] + refine_ms_per_q;

      }

      // --- CPU refine path (staged, fair) ---
      else {
        auto t_ref0 = std::chrono::steady_clock::now();

        recall_sum = 0.0;

        for (uint64_t qi = 0; qi < Q; ++qi) {
          const float* qv = query.vector_ptr_f32(qi);
          const faiss::idx_t* cand = I_all.data() + (size_t)qi * (size_t)k_search;

          auto pred_ids = refine_topk_l2_ids(base, qv, cand, refine_k, (uint32_t)k);

          if (gt_payload) {
            const uint32_t* gt = gt_payload + (size_t)qi * (size_t)k;
            std::vector<uint64_t> gt_ids; gt_ids.reserve(k);
            for (int j = 0; j < k; ++j) gt_ids.push_back((uint64_t)gt[j]);

            const size_t hit = intersection_size_vec(gt_ids, pred_ids);
            recall_sum += double(hit) / double(k);
            sink += (float)gt[0];
          }
        }

        auto t_ref1 = std::chrono::steady_clock::now();
        refine_ms_total = ms_since(t_ref0, t_ref1);
        refine_ms_per_q = refine_ms_total / double(Q);



        std::cout << "CPU_REFINE staged refine_ms_total=" << refine_ms_total
                  << " (avg " << std::fixed << std::setprecision(3) << refine_ms_per_q << " ms/query)\n";
      }
    } else if (!ann_only && refine_k == 0) {
      // optional: compute recall on raw ANN top-k (k_search==k) if GT exists
      if (gt_payload) {
        recall_sum = 0.0;
        for (uint64_t qi = 0; qi < Q; ++qi) {
          const faiss::idx_t* cand = I_all.data() + (size_t)qi * (size_t)k_search;

          std::vector<uint64_t> pred_ids;
          pred_ids.reserve(k);
          for (int i = 0; i < k; ++i) if (cand[i] >= 0) pred_ids.push_back((uint64_t)cand[i]);

          const uint32_t* gt = gt_payload + (size_t)qi * (size_t)k;
          std::vector<uint64_t> gt_ids; gt_ids.reserve(k);
          for (int j = 0; j < k; ++j) gt_ids.push_back((uint64_t)gt[j]);

          recall_sum += double(intersection_size_vec(gt_ids, pred_ids)) / double(k);
          sink += (float)gt[0];
        }
      }
    }

    // -------------------------
    // TOTAL latency: per-query ANN + refine_ms_per_q (fair composition)
    // -------------------------
    total_lat_ms.resize(ann_lat_ms.size());
    for (size_t i = 0; i < ann_lat_ms.size(); ++i) {
      total_lat_ms[i] = ann_lat_ms[i] + ((!ann_only && refine_k > 0) ? refine_ms_per_q : 0.0);
    }

  } else {
    // ---------------------------------------------------------
    // interleaved (original behavior) for backward compatibility
    // ---------------------------------------------------------
    for (uint64_t qi = 0; qi < Q; ++qi) {
      const float* qv = query.vector_ptr_f32(qi);

      auto t0 = std::chrono::steady_clock::now();
      idx->search(1, qv, k_search, Dd.data(), I.data());
      auto t1 = std::chrono::steady_clock::now();
      ann_lat_ms.push_back(ms_since(t0, t1));

      std::vector<uint64_t> pred_ids;
      if (!ann_only && refine_k > 0) {
        pred_ids = refine_topk_l2_ids(base, qv, I.data(), refine_k, (uint32_t)k);
      } else {
        pred_ids.reserve(k);
        for (int i = 0; i < k; ++i) if (I[i] >= 0) pred_ids.push_back((uint64_t)I[i]);
      }

      auto t2 = std::chrono::steady_clock::now();
      total_lat_ms.push_back(ms_since(t0, t2));

      if (!ann_only && gt_payload) {
        const uint32_t* gt = gt_payload + (size_t)qi * (size_t)k;
        std::vector<uint64_t> gt_ids; gt_ids.reserve(k);
        for (int j = 0; j < k; ++j) gt_ids.push_back((uint64_t)gt[j]);
        recall_sum += double(intersection_size_vec(gt_ids, pred_ids)) / double(k);
        sink += (float)gt[0];
      }
    }
  }


  
  if (!cuda_refine){
    t.threads=0;
    t.nwarps=0;
    t.shmem_bytes=0;
  }

  // ---- stats (machine-readable) ----
  LatStats annS   = compute_lat_stats(ann_lat_ms);
  LatStats totalS = compute_lat_stats(total_lat_ms);

  // keep existing human-readable printing if you want
  if (!ann_only) {
    const double recall_at_k = recall_sum / double(Q);
    std::cout << "Recall@" << k << ": " << std::setprecision(4) << recall_at_k
              << std::setprecision(3) << "\n";
  } else {
    std::cout << "Recall@" << k << ": N/A (EVAL_MODE=ann_only)\n";
  }

  std::cout << "ANN Avg:   " << annS.avg_ms   << " ms/query  (" << annS.qps   << " QPS)\n";
  std::cout << "ANN p50:   " << annS.p50_ms   << " ms\n";
  std::cout << "ANN p95:   " << annS.p95_ms   << " ms\n";
  std::cout << "ANN p99:   " << annS.p99_ms   << " ms\n";

  std::cout << "TOTAL Avg:   " << totalS.avg_ms << " ms/query  (" << totalS.qps << " QPS)\n";
  std::cout << "TOTAL p50:   " << totalS.p50_ms << " ms\n";
  std::cout << "TOTAL p95:   " << totalS.p95_ms << " ms\n";
  std::cout << "TOTAL p99:   " << totalS.p99_ms << " ms\n";

  // ---- single-line RESULT for CSV parsing ----
  std::cout.setf(std::ios::fixed);
  std::cout << std::setprecision(6);

  auto kvi = [&](const char* key, long long v) {
    std::cout << " " << key << "=" << v;
  };
  auto kvd = [&](const char* key, double v) {
    std::cout << " " << key << "=" << v;
  };
  auto kvs = [&](const char* key, const std::string& v) {
    std::cout << " " << key << "=" << v;
  };

  std::cout << "RESULT";

  kvi("refine_k", refine_k);
  kvi("k_search", k_search);
  kvi("nprobe", nprobe);
  kvi("Q", (long long)Q);
  kvi("k", (long long)k);

  kvi("cuda_refine", cuda_refine ? 1 : 0);
  kvi("refine_enabled", (!ann_only && refine_k > 0) ? 1 : 0);
  kvs("refine_backend", ((!ann_only && refine_k > 0) ? (cuda_refine ? "cuda" : "cpu") : "none"));

  kvd("ann_avg_ms", annS.avg_ms);
  kvd("ann_p99_ms", annS.p99_ms);
  kvd("total_avg_ms", totalS.avg_ms);
  kvd("total_p99_ms", totalS.p99_ms);

  kvd("refine_ms_total", refine_ms_total);
  kvd("refine_ms_per_q", refine_ms_per_q);

  kvs("kernel_mode", getenv_str("CUDA_KERNEL_MODE", "baseline"));
  kvi("cuda_pinned", getenv_int("CUDA_PINNED", 0));
  kvi("cuda_return_dist", getenv_int("CUDA_RETURN_DIST", 1));
  kvs("git_rev", getenv_str("GIT_SHA", "NA"));

  kvd("refine_h2d_ms", refine_h2d_ms);
  kvd("refine_kernel_ms", refine_kernel_ms);
  kvd("refine_d2h_ms", refine_d2h_ms);
  kvd("refine_kernel_ms_per_q", (Q ? (refine_kernel_ms / double(Q)) : 0.0));

  kvi("cuda_threads", (long long)t.threads);
  kvi("cuda_nwarps", (long long)t.nwarps);
  kvi("cuda_shmem_bytes", (long long)t.shmem_bytes);
  kvi("cuda_forced_threads", (long long)getenv_int("CUDA_BLOCK_THREADS", 0));
  kvi("cuda_shmem_optin", (long long)getenv_int("CUDA_SHMEM_OPTIN", 0));

  std::cout << "\n";

  return 0;
}
