#include "nvdb/vector_dataset.h"
#include "nvdb/flat_index.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: nvdb_search <base.vecbin> <query.vecbin> <k>\n";
    return 1;
  }

  const std::string base_path = argv[1];
  const std::string query_path = argv[2];
  const uint32_t k = static_cast<uint32_t>(std::stoul(argv[3]));

  nvdb::VectorDataset base, query;
  base.load(base_path);
  query.load(query_path);

  if (base.dim() != query.dim()) {
    std::cerr << "Dim mismatch: base.dim=" << base.dim()
              << ", query.dim=" << query.dim() << "\n";
    return 2;
  }

  nvdb::FlatIndex index(&base);

  std::cout << "Base count=" << base.count() << " dim=" << base.dim()
            << " | Query count=" << query.count() << "\n";

  const float* q0 = query.vector_ptr(0);
  auto topk = index.search_topk_dot(q0, k);

  std::cout << std::fixed << std::setprecision(6);
  for (size_t i = 0; i < topk.size(); ++i) {
    std::cout << "#" << (i + 1)
              << " row=" << topk[i].id
              << " score=" << topk[i].score << "\n";
  }
  return 0;
}
