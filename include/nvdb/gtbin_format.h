#pragma once
#include <cstdint>
#include <cstddef>

namespace nvdb {

// "NVDBGT01" (8 bytes)
static constexpr uint64_t kGtMagic = 0x4E56444247543031ULL;
static constexpr uint32_t kGtVersion = 1;

enum class GtMetric : uint32_t {
  // Ranking metric used for GT ids
  // For normalized embeddings: L2 ranking == DOT ranking.
  DotEquivalentL2 = 1,
};

#pragma pack(push, 1)
struct GtBinHeader {
  uint64_t magic;      // kGtMagic
  uint32_t version;    // kGtVersion
  uint32_t metric;     // (uint32_t)GtMetric
  uint32_t k;          // top-k
  uint32_t dim;        // embedding dim
  uint64_t Q;          // number of queries
  uint64_t N;          // number of base vectors
  uint8_t  reserved[64 - (8 + 4 + 4 + 4 + 4 + 8 + 8)];
};
#pragma pack(pop)

static_assert(sizeof(GtBinHeader) == 64, "GtBinHeader must be 64 bytes");

// Payload: uint32_t gt_ids[Q * k] (row ids in [0, N))
inline size_t gt_payload_bytes(uint64_t Q, uint32_t k) {
  return static_cast<size_t>(Q) * static_cast<size_t>(k) * sizeof(uint32_t);
}

} // namespace nvdb
