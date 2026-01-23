#pragma once
#include "nvdb/mmap_file.h"
#include "nvdb/vecbin_format.h"

#include <cstdint>
#include <string>

namespace nvdb {

class VectorDataset {
public:
  VectorDataset() = default;

  void load(const std::string& path);

  uint64_t count() const { return count_; }
  uint32_t dim() const { return dim_; }
  uint32_t dtype() const { return dtype_; } // matches VecbinHeader::dtype, or Float32 for raw12

  // Typed accessors (preferred)
  const float* vector_ptr_f32(uint64_t i) const;
  const uint16_t* vector_ptr_f16(uint64_t i) const;


  const int8_t* vector_ptr_i8(uint64_t i) const;
  const float*  scale_ptr_i8(uint64_t i) const;

  // Backward compatible: only valid if dataset is float32
  const float* vector_ptr(uint64_t i) const { return vector_ptr_f32(i); }

private:
  MmapFile mm_;

  uint64_t count_ = 0;
  uint32_t dim_ = 0;
  uint32_t dtype_ = static_cast<uint32_t>(DType::Float32);

  size_t data_offset_ = 0;

  const float* vectors_f32_ = nullptr;
  const uint16_t* vectors_f16_ = nullptr;
  const int8_t*  vectors_i8_ = nullptr;
  const float*   scales_i8_  = nullptr;
};

} // namespace nvdb
