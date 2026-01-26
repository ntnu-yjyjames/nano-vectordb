#include "nvdb/vector_dataset.h"
#include "nvdb/flat_index.h"
#include "nvdb/flat_index_omp.h"
#include "nvdb/flat_index_pool.h"
#include "nvdb/score_dispatch.h"

#include <hnswlib/hnswlib.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#if NVDB_HAS_OPENMP
#include <omp.h>
#endif

static double ms_since(const std::chrono::steady_clock::time_point& t0,
                       const std::chrono::steady_clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static int getenv_int(const char* k, int defv) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : defv;
}

static std::vector<uint64_t> hnsw_topk_ids(hnswlib::HierarchicalNSW<float>& index,
                                           const float* q, int k) {
  auto res = index.searchKnn(q, k);
  std::vector<uint64_t> ids;
  ids.reserve(res.size());
  while (!res.empty()) {
    ids.push_back((uint64_t)res.top().second);
    res.pop();
  }
  // now from worst->best; reverse to best->worst
  std::reverse(ids.begin(), ids.end());
  return ids;
}

static size_t intersection_size(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
  std::unordered_set<uint64_t> s(a.begin(), a.end());
  size_t hit = 0;
  for (auto x : b) hit += (s.find(x) != s.end()) ? 1 : 0;
  return hit;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr
      << "Usage: nvdb_hnsw_eval <base.vecbin> <hnsw.index> <query.vecbin> <k>\n"
      << "Env:\n"
      << "  HNSW_EF_SEARCH=64\n"
      << "  EXACT_MODE=omp|st|pool   (default=omp if OpenMP available, else st)\n"
      << "  EXACT_THREADS=8          (pool threads; omp uses OMP_NUM_THREADS)\n"
      << "  WARMUP=5                 (default=5)\n";
    return 1;
  }

  const std::string base_path  = argv[1];
  const std::string index_path = argv[2];
  const std::string query_path = argv[3];
  const int k = std::atoi(argv[4]);

  const int efS = getenv_int("HNSW_EF_SEARCH", 64);
  const int warmup = getenv_int("WARMUP", 5);

  std::string exact_mode = "st";
#if NVDB_HAS_OPENMP
  exact_mode = "omp";
#endif
  if (const char* m = std::getenv("EXACT_MODE")) exact_mode = m;

  int exact_threads = getenv_int("EXACT_THREADS", 8);

  nvdb::VectorDataset base, query;
  base.load(base_path);
  query.load(query_path);

  if (query.dtype() != (uint32_t)nvdb::DType::Float32) {
    std::cerr << "Query must be float32 vecbin (got dtype=" << query.dtype() << ")\n";
    return 2;
  }
  if (base.dim() != query.dim()) {
    std::cerr << "Dim mismatch: base.dim=" << base.dim() << " query.dim=" << query.dim() << "\n";
    return 3;
  }

  const uint32_t D = base.dim();
  const uint64_t N = base.count();
  const uint64_t Q = query.count();

  // Load HNSW index (L2 space)
  hnswlib::L2Space space(D);
  hnswlib::HierarchicalNSW<float> hnsw(&space, index_path);
  hnsw.setEf(efS);

  // Exact engines
  nvdb::FlatIndex st_index(&base);
  nvdb::FlatIndexOMP omp_index(&base);
  std::unique_ptr<nvdb::FlatIndexPool> pool_index;
  if (exact_mode == "pool") pool_index = std::make_unique<nvdb::FlatIndexPool>(&base, exact_threads);

  std::cout << "Base: N=" << N << " D=" << D << " dtype=" << base.dtype()
            << " | Query: Q=" << Q
            << " | k=" << k
            << " | HNSW efSearch=" << efS
            << " | exact_mode=" << exact_mode
            << " | warmup=" << warmup
            << "\n";

  auto exact_topk = [&](const float* qvec) -> std::vector<nvdb::SearchResult> {
    if (exact_mode == "st")   return st_index.search_topk_dot(qvec, (uint32_t)k);
    if (exact_mode == "omp")  return omp_index.search_topk_dot(qvec, (uint32_t)k);
    if (exact_mode == "pool") return pool_index->search_topk_dot(qvec, (uint32_t)k);
    throw std::runtime_error("Unknown EXACT_MODE");
  };

  // Warmup (HNSW only)
  for (int i = 0; i < warmup && Q > 0; ++i) {
    const float* q0 = query.vector_ptr_f32(0);
    auto ids = hnsw_topk_ids(hnsw, q0, k);
    (void)ids;
  }

  // Evaluate
  std::vector<double> lat_ms;
  lat_ms.reserve((size_t)Q);

  double recall_sum = 0.0;
  volatile float sink = 0.0f;

  auto t_all0 = std::chrono::steady_clock::now();

  for (uint64_t qi = 0; qi < Q; ++qi) {
    const float* qv = query.vector_ptr_f32(qi);

    // ANN
    auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> ann_ids = hnsw_topk_ids(hnsw, qv, k);
    auto t1 = std::chrono::steady_clock::now();
    lat_ms.push_back(ms_since(t0, t1));

    // Exact (ground truth)
    auto gt = exact_topk(qv);
    std::vector<uint64_t> gt_ids;
    gt_ids.reserve(gt.size());
    for (auto& r : gt) gt_ids.push_back(r.id);

    const size_t hit = intersection_size(gt_ids, ann_ids);
    recall_sum += double(hit) / double(k);

    if (!gt.empty()) sink += gt[0].score;
  }

  auto t_all1 = std::chrono::steady_clock::now();
    const double total_ms = ms_since(t_all0, t_all1);

    // ---- Total (ANN + Exact + bookkeeping) ----
    const double total_avg_ms = total_ms / double(Q);
    const double total_qps = (double(Q) * 1000.0) / total_ms;

    // ---- ANN-only stats (from lat_ms) ----
    double ann_sum_ms = 0.0;
    for (double x : lat_ms) ann_sum_ms += x;
    const double ann_avg_ms = (lat_ms.empty()) ? 0.0 : (ann_sum_ms / double(lat_ms.size()));
    const double ann_qps = (ann_avg_ms > 0.0) ? (1000.0 / ann_avg_ms) : 0.0;

    std::sort(lat_ms.begin(), lat_ms.end());
    auto pct = [&](double p) -> double {
    if (lat_ms.empty()) return 0.0;
    const double pos = (p / 100.0) * (lat_ms.size() - 1);
    const size_t i0 = (size_t)pos;
    const size_t i1 = std::min(i0 + 1, lat_ms.size() - 1);
    const double frac = pos - double(i0);
    return lat_ms[i0] * (1.0 - frac) + lat_ms[i1] * frac;
    };

    const double recall_at_k = recall_sum / double(Q);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Recall@" << k << ": " << std::setprecision(4) << recall_at_k << std::setprecision(3) << "\n";

    // ANN-only (true ANN latency)
    std::cout << "ANN Avg:   " << ann_avg_ms << " ms/query  (" << ann_qps << " QPS)\n";
    std::cout << "ANN p50:   " << pct(50) << " ms\n";
    std::cout << "ANN p95:   " << pct(95) << " ms\n";
    std::cout << "ANN p99:   " << pct(99) << " ms\n";

    // Total end-to-end (includes exact for recall)
    std::cout << "TOTAL Avg: " << total_avg_ms << " ms/query  (" << total_qps << " QPS)\n";
    std::cout << "TOTAL:     " << total_ms << " ms\n";

    std::cout << "sink=" << sink << "\n";


  return 0;
}
