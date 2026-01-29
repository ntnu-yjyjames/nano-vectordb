#include "nvdb/vector_dataset.h"
#include "nvdb/to_f32_row.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static int getenv_int(const char* k, int defv) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : defv;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr
      << "Usage: nvdb_ivf_build <base.vecbin> <out.faiss>\n"
      << "Env:\n"
      << "  IVF_NLIST=4096\n"
      << "  IVF_TRAIN=50000\n"
      << "  IVF_METRIC=l2   (only l2 supported in this tool)\n";
    return 1;
  }

  const std::string base_path = argv[1];
  const std::string out_path  = argv[2];

  const int nlist  = getenv_int("IVF_NLIST", 4096);
  const int ntrain = getenv_int("IVF_TRAIN", 50000);

  nvdb::VectorDataset base;
  base.load(base_path);

  const uint32_t d = base.dim();
  const uint64_t N = base.count();

  const int train_n = (int)std::min<uint64_t>(N, (uint64_t)ntrain);

  std::cout << "Base: N=" << N << " d=" << d << " dtype=" << base.dtype()
            << " | IVF nlist=" << nlist << " ntrain=" << train_n << "\n";

  // Prepare training set (float32)
  std::vector<float> train((size_t)train_n * d);
  std::vector<float> buf(d);

  for (int i = 0; i < train_n; ++i) {
    nvdb::base_row_to_f32(base, (uint64_t)i, buf.data());
    std::memcpy(train.data() + (size_t)i * d, buf.data(), d * sizeof(float));
  }

  // Coarse quantizer + IVF-Flat (L2)
  faiss::IndexFlatL2 quantizer((int)d);
  faiss::IndexIVFFlat index(&quantizer, (int)d, nlist, faiss::METRIC_L2);

  // Train
  auto t0 = std::chrono::steady_clock::now();
  index.train(train_n, train.data());
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "Train: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";

  // Add vectors in blocks to reduce peak memory
  const uint64_t block = 2048;
  std::vector<float> add_block;
  add_block.reserve((size_t)block * d);

  auto t2 = std::chrono::steady_clock::now();
  for (uint64_t i0 = 0; i0 < N; i0 += block) {
    const uint64_t i1 = std::min<uint64_t>(i0 + block, N);
    const uint64_t nb = i1 - i0;

    add_block.assign((size_t)nb * d, 0.0f);

    for (uint64_t i = 0; i < nb; ++i) {
      nvdb::base_row_to_f32(base, i0 + i, buf.data());
      std::memcpy(add_block.data() + (size_t)i * d, buf.data(), d * sizeof(float));
    }

    index.add((faiss::idx_t)nb, add_block.data());

    if (i1 % 200000 == 0 || i1 == N) {
      std::cout << "added " << i1 << "/" << N << "\n";
    }
  }
  auto t3 = std::chrono::steady_clock::now();
  std::cout << "Add: " << std::chrono::duration<double>(t3 - t2).count() << " s\n";

  faiss::write_index(&index, out_path.c_str());
  std::cout << "Saved IVF-Flat index: " << out_path << "\n";

  return 0;
}
