#include "nvdb/vector_dataset.h"
#include <iostream>
#include <fstream>
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
  // chunked write to avoid huge single write
  constexpr size_t kChunk = 64 * 1024 * 1024; // 64MB
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
    std::cerr << "Usage: nvdb_slice <input.vecbin> <output.vecbin> <count>\n";
    return 1;
  }

  const std::string in_path = argv[1];
  const std::string out_path = argv[2];
  uint64_t want = std::stoull(argv[3]);

  nvdb::VectorDataset ds;
  ds.load(in_path);

  const uint64_t n = std::min<uint64_t>(want, ds.count());
  const uint32_t dim = ds.dim();

  if (n > 0xFFFFFFFFULL) {
    std::cerr << "Slice count too large for raw12 header (uint32)\n";
    return 2;
  }

  std::ofstream ofs(out_path, std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open output: " << out_path << "\n";
    return 3;
  }

  Raw12Header hdr{};
  hdr.count = static_cast<uint32_t>(n);
  hdr.reserved = 0;
  hdr.dim = dim;

  ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  if (!ofs) {
    std::cerr << "Failed to write header\n";
    return 4;
  }

  const float* base = ds.vector_ptr(0);
  const size_t total_floats = static_cast<size_t>(n) * static_cast<size_t>(dim);
  const size_t total_bytes  = total_floats * sizeof(float);

  write_all(ofs, reinterpret_cast<const char*>(base), total_bytes);
  ofs.close();

  std::cout << "Wrote slice: " << out_path
            << " count=" << n << " dim=" << dim
            << " bytes=" << (sizeof(hdr) + total_bytes) << "\n";
  return 0;
}
