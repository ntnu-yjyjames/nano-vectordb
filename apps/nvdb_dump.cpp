#include "nvdb/vector_dataset.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: nvdb_dump <vecbin_path> [num_vectors=3] [num_dims=8]\n";
    return 1;
  }

  const std::string path = argv[1];
  const uint64_t nvec = (argc >= 3) ? std::stoull(argv[2]) : 3ULL;
  const uint32_t nd = (argc >= 4) ? static_cast<uint32_t>(std::stoul(argv[3])) : 8U;

  nvdb::VectorDataset ds;
  ds.load(path);

  std::cout << "Loaded: count=" << ds.count() << ", dim=" << ds.dim() << "\n";

  const uint64_t show_vec = std::min<uint64_t>(nvec, ds.count());
  const uint32_t show_dim = std::min<uint32_t>(nd, ds.dim());

  std::cout << std::fixed << std::setprecision(6);
  for (uint64_t i = 0; i < show_vec; ++i) {
    const float* v = ds.vector_ptr(i);
    std::cout << "v[" << i << "]: ";
    for (uint32_t d = 0; d < show_dim; ++d) {
      std::cout << v[d] << (d + 1 == show_dim ? "" : ", ");
    }
    if (show_dim < ds.dim()) std::cout << " ...";
    std::cout << "\n";
  }
  return 0;
}
