#include "nvdb/vector_dataset.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <unordered_set>
#include <cstdint>
#include <algorithm>

#pragma pack(push, 1)
struct Raw12Header {
  uint32_t count;
  uint32_t reserved;
  uint32_t dim;
};
#pragma pack(pop)

static void write_all(std::ofstream& ofs, const char* p, size_t nbytes) {
  constexpr size_t kChunk = 64 * 1024 * 1024;
  size_t off = 0;
  while (off < nbytes) {
    size_t take = std::min(kChunk, nbytes - off);
    ofs.write(p + off, static_cast<std::streamsize>(take));
    if (!ofs) throw std::runtime_error("Write failed");
    off += take;
  }
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: nvdb_make_query <base.vecbin> <query.vecbin> <num_queries> [seed=42] [mode=random|first]\n";
    return 1;
  }

  const std::string base_path  = argv[1];
  const std::string query_path = argv[2];
  uint64_t qn = std::stoull(argv[3]);
  uint32_t seed = (argc >= 5) ? static_cast<uint32_t>(std::stoul(argv[4])) : 42U;
  std::string mode = (argc >= 6) ? std::string(argv[5]) : "random";

  nvdb::VectorDataset base;
  base.load(base_path);

  if (qn == 0) {
    std::cerr << "num_queries must be > 0\n";
    return 2;
  }
  if (qn > base.count()) qn = base.count();
  if (qn > 0xFFFFFFFFULL) {
    std::cerr << "Too many queries for raw12 header (uint32)\n";
    return 3;
  }

  const uint32_t dim = base.dim();

  // choose indices
  std::vector<uint64_t> idx;
  idx.reserve(static_cast<size_t>(qn));

  if (mode == "first") {
    for (uint64_t i = 0; i < qn; ++i) idx.push_back(i);
  } else if (mode == "random") {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint64_t> dist(0, base.count() - 1);
    std::unordered_set<uint64_t> used;
    used.reserve(static_cast<size_t>(qn) * 2);

    while (idx.size() < qn) {
      uint64_t r = dist(rng);
      if (used.insert(r).second) idx.push_back(r);
    }
  } else {
    std::cerr << "Unknown mode: " << mode << " (use random or first)\n";
    return 4;
  }

  // copy query vectors into contiguous buffer
  std::vector<float> Q(static_cast<size_t>(qn) * static_cast<size_t>(dim));
  for (size_t qi = 0; qi < idx.size(); ++qi) {
    const float* v = base.vector_ptr(idx[qi]);
    std::copy(v, v + dim, Q.begin() + qi * dim);
  }

  // write raw12 query file
  std::ofstream ofs(query_path, std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open output: " << query_path << "\n";
    return 5;
  }

  Raw12Header hdr{};
  hdr.count = static_cast<uint32_t>(qn);
  hdr.reserved = 0;
  hdr.dim = dim;

  ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  if (!ofs) {
    std::cerr << "Failed to write header\n";
    return 6;
  }

  write_all(ofs, reinterpret_cast<const char*>(Q.data()), Q.size() * sizeof(float));
  ofs.close();

  std::cout << "Wrote query: " << query_path
            << " count=" << qn << " dim=" << dim
            << " mode=" << mode << " seed=" << seed << "\n";

  // print chosen indices (for debugging/repro)
  std::cout << "Query indices (first up to 20): ";
  for (size_t i = 0; i < std::min<size_t>(idx.size(), 20); ++i) {
    std::cout << idx[i] << (i + 1 == std::min<size_t>(idx.size(), 20) ? "" : ",");
  }
  std::cout << "\n";
  return 0;
}
