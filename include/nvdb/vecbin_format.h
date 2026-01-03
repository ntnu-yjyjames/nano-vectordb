#pragma once
#include <cstdint>
#include <cstddef>

namespace nvdb {

  static constexpr uint64_t kMagic = 0x4E56444256454331ULL; // "NVDBVEC1"
  static constexpr uint32_t kVersion = 1;

  enum class DType : uint32_t {
    Float32 = 1,
    Float16 = 2,
  };

  //Disabling memory alignment (Memory Packing)
  #pragma pack(push, 1)
  struct VecbinHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t dtype;
    uint32_t dim;
    uint32_t reserved0;
    uint64_t count;
    uint8_t  reserved[64 - (8 + 4 + 4 + 4 + 4 + 8)];
  };
  #pragma pack(pop)

  static_assert(sizeof(VecbinHeader) == 64, "VecbinHeader must be 64 bytes");

  // Float32-only helper (legacy)
  inline size_t bytes_for_vectors_f32(uint64_t count, uint32_t dim) {
    return static_cast<size_t>(count) * static_cast<size_t>(dim) * sizeof(float);
  }



  inline size_t bytes_per_elem(uint32_t dtype) {
    if (dtype == static_cast<uint32_t>(DType::Float32)) return sizeof(float);
    if (dtype == static_cast<uint32_t>(DType::Float16)) return sizeof(uint16_t);
    return 0;
  }

  inline size_t bytes_for_vectors_typed(uint64_t count, uint32_t dim, uint32_t dtype) {
    const size_t bpe = bytes_per_elem(dtype);
    if (bpe == 0) return 0;
    return static_cast<size_t>(count) * static_cast<size_t>(dim) * bpe;
  }


} // namespace nvdb
