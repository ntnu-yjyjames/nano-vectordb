#include "nvdb/vector_dataset.h"
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
  if (argc < 4) {
    std::cerr << "Usage: nvdb_hnsw_search <index> <query.vecbin> <k>\n"
              << "Env: HNSW_EF_SEARCH=64\n";
    return 1;
  }
  const std::string index_path = argv[1];
  const std::string query_path = argv[2];
  const int k = std::atoi(argv[3]);
  const int efS = getenv_int("HNSW_EF_SEARCH", 64);

  nvdb::VectorDataset query;
  query.load(query_path);
  const uint32_t D = query.dim();
  if (query.dtype() != (uint32_t)nvdb::DType::Float32) {
    std::cerr << "Query must be float32 vecbin\n";
    return 2;
  }

  hnswlib::L2Space space(D);
  hnswlib::HierarchicalNSW<float> index(&space, index_path);
  index.setEf(efS);

  std::cout << "Loaded index: " << index_path << " | D=" << D
            << " | k=" << k << " | efSearch=" << efS
            << " | Q=" << query.count() << "\n";

  for (uint64_t qi = 0; qi < query.count(); ++qi) {
    const float* q = query.vector_ptr_f32(qi);
    auto result = index.searchKnn(q, k);

    std::vector<hnswlib::labeltype> ids;
    ids.reserve(result.size());
    while (!result.empty()) {
      ids.push_back(result.top().second);
      result.pop();
    }
    // hnswlib returns closest first via min-heap; we reverse to print best first
    std::reverse(ids.begin(), ids.end());

    std::cout << "q" << qi << ":";
    for (auto id : ids) std::cout << " " << (uint64_t)id;
    std::cout << "\n";
  }

  return 0;
}
