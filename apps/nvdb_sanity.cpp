#include "nvdb/vector_dataset.h"
#include <iostream>
#include <random>
#include <cmath>

// calculate L2 Norm
static float l2norm(const float* v, uint32_t dim) {
  double s = 0.0;
  for (uint32_t i = 0; i < dim; ++i) s += double(v[i]) * double(v[i]);
  return float(std::sqrt(s));
}

//check for NAN
static bool has_nan_inf(const float* v, uint32_t dim) {
  for (uint32_t i = 0; i < dim; ++i) {
    if (!std::isfinite(v[i])) return true;
  }
  return false;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: nvdb_sanity <vecbin_path> [samples=10]\n";
    return 1;
  }
  const std::string path = argv[1];
  const int samples = (argc >= 3) ? std::stoi(argv[2]) : 10;

  nvdb::VectorDataset ds;
  ds.load(path);

  // Random Generator
  std::mt19937 rng(42);
  std::uniform_int_distribution<uint64_t> uid(0, ds.count() - 1);

  for (int s = 0; s < samples; ++s) {
    uint64_t i = uid(rng);
    const float* v = ds.vector_ptr(i);

    if (has_nan_inf(v, ds.dim())) {
      std::cout << "idx=" << i << " has NaN/Inf\n";
      return 2;
    }

    float n = l2norm(v, ds.dim());
    std::cout << "idx=" << i << " L2=" << n << "\n";
  }

  return 0;
}
