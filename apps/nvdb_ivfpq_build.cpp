#include "nvdb/vector_dataset.h"
#include "nvdb/to_f32_row.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/VectorTransform.h>   // OPQMatrix


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
      << "Usage: nvdb_ivfpq_build <base.vecbin> <out.faiss>\n"
      << "Env:\n"
      << "  IVF_NLIST=4096\n"
      << "  IVF_TRAIN=200000\n"
      << "  PQ_M=48            (must divide dim=384)\n"
      << "  PQ_BITS=8          (typically 8)\n"
      << "  USE_OPQ=0          (0/1; optional)\n";
    return 1;
  }

  const std::string base_path = argv[1];
  const std::string out_path  = argv[2];

  const int nlist  = getenv_int("IVF_NLIST", 4096);
  const int ntrain = getenv_int("IVF_TRAIN", 200000);
  const int pq_m   = getenv_int("PQ_M", 48);
  const int pq_bits= getenv_int("PQ_BITS", 8);
  const int use_opq= getenv_int("USE_OPQ", 0);

  nvdb::VectorDataset base;
  base.load(base_path);

  const uint32_t d = base.dim();
  const uint64_t N = base.count();
  const int train_n = (int)std::min<uint64_t>(N, (uint64_t)ntrain);

  if ((int)d % pq_m != 0) {
    std::cerr << "PQ_M must divide dim. dim=" << d << " PQ_M=" << pq_m << "\n";
    return 2;
  }

  std::cout << "Base: N=" << N << " d=" << d << " dtype=" << base.dtype()
            << " | IVF-PQ nlist=" << nlist
            << " ntrain=" << train_n
            << " PQ_M=" << pq_m
            << " PQ_BITS=" << pq_bits
            << " USE_OPQ=" << use_opq
            << "\n";

  // Prepare training set (float32)
  std::vector<float> train((size_t)train_n * d);
  std::vector<float> buf(d);

  for (int i = 0; i < train_n; ++i) {
    nvdb::base_row_to_f32(base, (uint64_t)i, buf.data());
    std::memcpy(train.data() + (size_t)i * d, buf.data(), d * sizeof(float));
  }

  // Coarse quantizer (L2) — allocate on heap if we use PreTransform (ownership)
    faiss::Index* top_index = nullptr;

    if (use_opq) {
    // OPQMatrix(d, M) where M = pq_m (number of subquantizers)
    auto* opq = new faiss::OPQMatrix((int)d, pq_m);

    // (Optional) tune OPQ iterations
    opq->niter = getenv_int("OPQ_NITER", 25);
    opq->verbose = false;

    auto* quantizer = new faiss::IndexFlatL2((int)d);
    auto* ivfpq = new faiss::IndexIVFPQ(quantizer, (int)d, nlist, pq_m, pq_bits, faiss::METRIC_L2);

    // Wrap with pre-transform: OPQ -> IVFPQ
    // IndexPreTransform takes ownership of opq and ivfpq by default.
    top_index = new faiss::IndexPreTransform(opq, ivfpq);
    } else {
    auto* quantizer = new faiss::IndexFlatL2((int)d);
    auto* ivfpq = new faiss::IndexIVFPQ(quantizer, (int)d, nlist, pq_m, pq_bits, faiss::METRIC_L2);
    top_index = ivfpq; // plain IVFPQ
    }

    // Train (coarse + OPQ + PQ codebooks)
    auto t0 = std::chrono::steady_clock::now();
    top_index->train(train_n, train.data());
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "Train: " << std::chrono::duration<double>(t1 - t0).count() << " s\n";


     // Add vectors in blocks (streaming to reduce peak memory)
    const uint64_t block = 2048;
    std::vector<float> add_block;
    add_block.reserve((size_t)block * d);

    // Add vectors in blocks
      auto t2 = std::chrono::steady_clock::now();

    for (uint64_t i0 = 0; i0 < N; i0 += block) {
        const uint64_t i1 = std::min<uint64_t>(i0 + block, N);
        const uint64_t nb = i1 - i0;

        add_block.assign((size_t)nb * d, 0.0f);

        for (uint64_t i = 0; i < nb; ++i) {
        nvdb::base_row_to_f32(base, i0 + i, buf.data());
        std::memcpy(add_block.data() + (size_t)i * d, buf.data(), d * sizeof(float));
        }

        top_index->add((faiss::idx_t)nb, add_block.data());

        if (i1 % 200000 == 0 || i1 == N) {
        std::cout << "added " << i1 << "/" << N << "\n";
        }
    }

    auto t3 = std::chrono::steady_clock::now();
    std::cout << "Add: " << std::chrono::duration<double>(t3 - t2).count() << " s\n";


    faiss::write_index(top_index, out_path.c_str());
    std::cout << "Saved IVF-" << (use_opq ? "OPQ-PQ" : "PQ") << " index: " << out_path << "\n";

    delete top_index; // safe: IndexPreTransform will delete owned sub-objects

  std::cout << "Saved IVF-PQ index: " << out_path << "\n";

  return 0;
}
