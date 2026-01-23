#pragma once
#include "nvdb/vector_dataset.h"
#include "nvdb/simd_dot.h"
#include "nvdb/vecbin_format.h"

#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace nvdb {

// Validate dtype once in caller (main thread) before parallel scan.
inline void ensure_supported_base_dtype(const VectorDataset& base) {
  const uint32_t dt = base.dtype();
  if (dt != static_cast<uint32_t>(DType::Float32) &&
      dt != static_cast<uint32_t>(DType::Float16) &&
      dt != static_cast<uint32_t>(DType::Int8)) {
    throw std::runtime_error("Unsupported base dtype (Float32/Float16/Int8 only)");
  }
}


// Compute similarity score between float32 query and base row (f32/f16/i8).
// Pass base_dtype and dim to avoid repeated calls in hot loops.
inline float score_query_base_at(const VectorDataset& base,
                                 const float* q_f32,
                                 uint64_t row_id,
                                 uint32_t dim,
                                 uint32_t base_dtype) {
  if (base_dtype == static_cast<uint32_t>(DType::Float32)) {
    const float* v = base.vector_ptr_f32(row_id);
    return nvdb::dot_f32(q_f32, v, dim);
  }

  if (base_dtype == static_cast<uint32_t>(DType::Float16)) {
    const uint16_t* v16 = base.vector_ptr_f16(row_id);
    return nvdb::dot_f32_f16base(q_f32, v16, dim);
  }

  if (base_dtype == static_cast<uint32_t>(DType::Int8)) {
    const int8_t* v8 = base.vector_ptr_i8(row_id);
    const float s = *base.scale_ptr_i8(row_id);
    return nvdb::dot_f32_i8base(q_f32, v8, dim, s);
  }


  throw std::runtime_error("Unsupported base dtype in score_query_base_at");
}

} // namespace nvdb
