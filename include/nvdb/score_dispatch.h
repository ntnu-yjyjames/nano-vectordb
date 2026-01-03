#pragma once
#include "nvdb/vector_dataset.h"
#include "nvdb/simd_dot.h"
#include "nvdb/vecbin_format.h"

#include <cstdint>
#include <stdexcept>

namespace nvdb {

// Validate once in the caller (main thread) before parallel scan.
inline void ensure_supported_base_dtype(const VectorDataset& base) {
  const uint32_t dt = base.dtype();
  if (dt != static_cast<uint32_t>(DType::Float32) &&
      dt != static_cast<uint32_t>(DType::Float16)) {
    throw std::runtime_error("Unsupported base dtype (only Float32/Float16 supported)");
  }
}

// Compute similarity score between float32 query and base vector (f32 or f16).
// Caller should pass base.dtype() once to avoid repeated virtual/branch overhead.
inline float score_query_base_at(const VectorDataset& base,
                                 const float* q_f32,
                                 uint64_t i,
                                 uint32_t dim,
                                 uint32_t base_dtype) {
  if (base_dtype == static_cast<uint32_t>(DType::Float32)) {
    const float* v = base.vector_ptr_f32(i);
    return nvdb::dot_f32(q_f32, v, dim);
  } else { // Float16
    const uint16_t* v16 = base.vector_ptr_f16(i);
    return nvdb::dot_f32_f16base(q_f32, v16, dim);
  }
}

} // namespace nvdb
