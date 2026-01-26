#include "nvdb/vector_dataset.h"
#include "nvdb/to_f32_row.h"
#include <hnswlib/hnswlib.h>

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

static int getenv_int(const char* k, int defv) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : defv;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: nvdb_hnsw_build <base.vecbin> <out.index>\n"
              << "Env: HNSW_M=16 HNSW_EF_CONSTRUCT=200 HNSW_MAX_ELEMENTS=0 (0=>use base.count)\n";
    return 1;
  }
  const std::string base_path = argv[1];
  const std::string out_path  = argv[2];

  const int M = getenv_int("HNSW_M", 16);
  const int efC = getenv_int("HNSW_EF_CONSTRUCT", 200);
  const int maxE_env = getenv_int("HNSW_MAX_ELEMENTS", 0);

  nvdb::VectorDataset base;
  base.load(base_path);

  const uint32_t D = base.dim();
  const uint64_t N = base.count();
  const size_t max_elements = (maxE_env > 0) ? (size_t)maxE_env : (size_t)N;

  std::cout << "Base: N=" << N << " D=" << D << " dtype=" << base.dtype()
            << " | HNSW M=" << M << " efC=" << efC << " max_elements=" << max_elements << "\n";

  hnswlib::L2Space space(D);
  hnswlib::HierarchicalNSW<float> index(&space, max_elements, M, efC);

  std::vector<float> buf(D);

  for (uint64_t i = 0; i < N; ++i) {
    nvdb::base_row_to_f32(base, i, buf.data());
    index.addPoint(buf.data(), (hnswlib::labeltype)i);
    if ((i + 1) % 200000 == 0) {
      std::cout << "added " << (i + 1) << "/" << N << "\n";
    }
  }

  index.saveIndex(out_path);
  std::cout << "Saved index: " << out_path << "\n";
  return 0;
}
