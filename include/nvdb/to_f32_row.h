#pragma once
#include "nvdb/vector_dataset.h"
#include "nvdb/f16_scalar.h"     
#include <cstdint>
#include <stdexcept>

namespace nvdb {

// Convert one base vector row to float32 into out[D].
inline void base_row_to_f32(const VectorDataset& base, uint64_t row, float* out) {
  const uint32_t D = base.dim();
  const uint32_t dt = base.dtype();

  if (dt == (uint32_t)DType::Float32) {
    const float* v = base.vector_ptr_f32(row);
    for (uint32_t j = 0; j < D; ++j) out[j] = v[j];
    return;
  }

  if (dt == (uint32_t)DType::Float16) {
    const uint16_t* v16 = base.vector_ptr_f16(row);
    for (uint32_t j = 0; j < D; ++j) out[j] = nvdb::f16_to_f32_scalar(v16[j]);
    return;
  }

  if (dt == (uint32_t)DType::Int8) {
    const int8_t* v8 = base.vector_ptr_i8(row);
    const float scale = *base.scale_ptr_i8(row);
    for (uint32_t j = 0; j < D; ++j) out[j] = float(v8[j]) * scale;
    return;
  }

  throw std::runtime_error("Unsupported dtype in base_row_to_f32");
}

} // namespace nvdb
