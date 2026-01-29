#include "nvdb/vector_dataset.h"
#include "nvdb/flat_index_omp.h"
#include "nvdb/to_f32_row.h"
#include "nvdb/gtbin_format.h"
#include "nvdb/mmap_file.h"



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
#include "nvdb/gtbin_format.h"
#include "nvdb/mmap_file.h"



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


  double recall_sum = 0.0;
  volatile float sink = 0.0f;
  std::vector<float> Dd(k_search);
  std::vector<faiss::idx_t> I(k_search);

  //std::vector<float> Dd(k);
  //std::vector<faiss::idx_t> I(k);

  for (uint64_t qi = 0; qi < Q; ++qi) {
    const float* qv = query.vector_ptr_f32(qi);

    auto t0 = std::chrono::steady_clock::now();
    idx->search(1, qv, k_search, Dd.data(), I.data());
    auto t1 = std::chrono::steady_clock::now();
    ann_lat_ms.push_back(ms_since(t0, t1));

    // Build predicted ids (either refined or raw ANN top-k)
    std::vector<uint64_t> pred_ids;
    if (!ann_only && refine_k > 0) {
      const int cand_k = std::min(refine_k, k_search);
      pred_ids = refine_topk_l2_ids(base, qv, I.data(), cand_k, (uint32_t)k);
    } else {
      pred_ids.reserve(k);
      for (int i = 0; i < k; ++i) {
        if (I[i] >= 0) pred_ids.push_back((uint64_t)I[i]);
      }
    }

    auto t2 = std::chrono::steady_clock::now();
    total_lat_ms.push_back(ms_since(t0, t2));

    // Recall (use cached GT if provided)
    if (!ann_only) {
      if (gt_payload) {
        const uint32_t* gt = gt_payload + (size_t)qi * (size_t)k;
        // compare to pred_ids (refined if enabled)
        std::vector<uint64_t> gt_ids;
        gt_ids.reserve(k);
        for (int j = 0; j < k; ++j) gt_ids.push_back((uint64_t)gt[j]);

        const size_t hit = intersection_size_vec(gt_ids, pred_ids);
        recall_sum += double(hit) / double(k);
        sink += (float)gt[0];
      } else {
        // fallback: slow exact path (avoid in sweeps)
        std::vector<uint64_t> gt_ids;
        if (exact_metric == "DOT") {
          auto gt = exact->search_topk_dot(qv, (uint32_t)k);
          for (auto& r : gt) gt_ids.push_back(r.id);
        } else {
          gt_ids = exact_topk_l2_ids(base, qv, (uint32_t)k);
        }
        const size_t hit = intersection_size_vec(gt_ids, pred_ids);
        recall_sum += double(hit) / double(k);
        if (!gt_ids.empty()) sink += (float)gt_ids[0];
      }
    }
  }

   

    
  
  


  

  // Stats

  /*
  double sum_ms = 0.0;
  for (double x : lat_ms) sum_ms += x;
  const double ann_avg_ms = sum_ms / double(lat_ms.size());
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


  std::cout << std::fixed << std::setprecision(3);
  std::cout << "EVAL_MODE=" << eval_mode << " EXACT_METRIC=" << exact_metric << "\n";

  if (!ann_only) {
    const double recall_at_k = recall_sum / double(Q);
    std::cout << "Recall@" << k << ": " << std::setprecision(4) << recall_at_k
              << std::setprecision(3) << "\n";
  } else {
    std::cout << "Recall@" << k << ": N/A (EVAL_MODE=ann_only)\n";
  }

  std::cout << "ANN Avg:   " << ann_avg_ms << " ms/query  (" << ann_qps << " QPS)\n";
  std::cout << "ANN p50:   " << pct(50) << " ms\n";
  std::cout << "ANN p95:   " << pct(95) << " ms\n";
  std::cout << "ANN p99:   " << pct(99) << " ms\n";
  std::cout << "sink=" << sink << "\n";


  delete idx;
  */
 std::cout << std::fixed << std::setprecision(3);

 auto summarize = [&](std::vector<double>& v, const char* name) {
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
    const double avg = sum / (double)v.size();
    const double qps = (avg > 0.0) ? (1000.0 / avg) : 0.0;

    std::cout << name << " Avg:   " << avg << " ms/query  (" << qps << " QPS)\n";
    std::cout << name << " p50:   " << pct(50) << " ms\n";
    std::cout << name << " p95:   " << pct(95) << " ms\n";
    std::cout << name << " p99:   " << pct(99) << " ms\n";
  };
  if (!ann_only) {
    const double recall_at_k = recall_sum / double(Q);
    std::cout << "Recall@" << k << ": " << std::setprecision(4) << recall_at_k
              << std::setprecision(3) << "\n";
  } else {
    std::cout << "Recall@" << k << ": N/A (EVAL_MODE=ann_only)\n";
  }

  summarize(ann_lat_ms,   "ANN");
  summarize(total_lat_ms, "TOTAL");
  std::cout << "sink=" << sink << "\n";

  return 0;
}
