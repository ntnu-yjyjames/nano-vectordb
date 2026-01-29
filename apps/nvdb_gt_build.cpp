#include "nvdb/vector_dataset.h"
#include "nvdb/flat_index.h"
#include "nvdb/flat_index_omp.h"
#include "nvdb/gtbin_format.h"
#include "nvdb/mmap_file.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>


static int getenv_int(const char* k, int defv) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : defv;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr
      << "Usage: nvdb_gt_build <base.vecbin> <query.vecbin> <k> <out.gtbin>\n"
      << "Notes:\n"
      << "  - Produces GT ids for L2 ranking by computing exact top-k dot products.\n"
      << "    This is valid because embeddings are assumed L2-normalized: L2^2 = 2 - 2*dot.\n"
      << "Env:\n"
      << "  GT_MODE=omp|st   (default=omp)\n"
      << "  WARMUP=2         (default=2)\n";
    return 1;
  }

  const std::string base_path  = argv[1];
  const std::string query_path = argv[2];
  const uint32_t k = static_cast<uint32_t>(std::stoul(argv[3]));
  const std::string out_path   = argv[4];

  const std::string gt_mode = []{
    const char* m = std::getenv("GT_MODE");
    return m ? std::string(m) : std::string("omp");
  }();
  const int warmup = getenv_int("WARMUP", 2);

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
  if (k == 0) {
    std::cerr << "k must be > 0\n";
    return 4;
  }

  const uint64_t N = base.count();
  const uint64_t Q = query.count();
  const uint32_t d = base.dim();

  std::cout << "GT build: N=" << N << " Q=" << Q << " d=" << d
            << " k=" << k << " mode=" << gt_mode << "\n";

  // Exact engines (dot). For normalized embeddings, dot top-k == L2 nearest top-k.
  nvdb::FlatIndex st(&base);
  nvdb::FlatIndexOMP omp(&base);

  auto exact_topk_ids = [&](const float* q) -> std::vector<uint32_t> {
    std::vector<nvdb::SearchResult> topk;
    if (gt_mode == "st") topk = st.search_topk_dot(q, k);
    else topk = omp.search_topk_dot(q, k);

    std::vector<uint32_t> ids;
    ids.reserve(topk.size());
    for (auto& r : topk) ids.push_back((uint32_t)r.id);
    return ids;
  };

  // Warmup
  for (int i = 0; i < warmup && Q > 0; ++i) {
    (void)exact_topk_ids(query.vector_ptr_f32(0));
  }

  // Compute GT ids
  std::vector<uint32_t> gt_ids;
  gt_ids.resize(static_cast<size_t>(Q) * static_cast<size_t>(k), 0);

  for (uint64_t qi = 0; qi < Q; ++qi) {
    const float* qv = query.vector_ptr_f32(qi);
    auto ids = exact_topk_ids(qv);
    if (ids.size() != k) {
      // In case base smaller than k; keep it strict
      std::cerr << "GT size mismatch at qi=" << qi << " got=" << ids.size() << " expected=" << k << "\n";
      return 5;
    }
    std::memcpy(gt_ids.data() + static_cast<size_t>(qi) * k, ids.data(), k * sizeof(uint32_t));
    if ((qi + 1) % 200 == 0) std::cout << "GT " << (qi + 1) << "/" << Q << "\n";
  }

  // Write gtbin
  nvdb::GtBinHeader h{};
  h.magic = nvdb::kGtMagic;
  h.version = nvdb::kGtVersion;
  h.metric = (uint32_t)nvdb::GtMetric::DotEquivalentL2;
  h.k = k;
  h.dim = d;
  h.Q = Q;
  h.N = N;

  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open output: " << out_path << "\n";
    return 6;
  }
  out.write(reinterpret_cast<const char*>(&h), sizeof(h));
  out.write(reinterpret_cast<const char*>(gt_ids.data()),
            static_cast<std::streamsize>(gt_ids.size() * sizeof(uint32_t)));
  out.close();

  std::cout << "Wrote GT: " << out_path
            << " (header=64B, payload=" << (gt_ids.size() * sizeof(uint32_t)) << " bytes)\n";
  return 0;
}
