#pragma once
#include <cstdint>

namespace nvdb {

// Public entry: runtime-dispatched dot(a,b)
float dot_f32(const float* a, const float* b, uint32_t dim);

// Optional: force scalar (useful for A/B test)
void set_force_scalar(bool v);

float dot_f32_f16base(const float* q_f32, const uint16_t* x_f16, uint32_t dim);
float dot_f32_i8base(const float* q_f32, const int8_t* x_i8, uint32_t dim, float scale);

} // namespace nvdb


