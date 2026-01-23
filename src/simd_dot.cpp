#include "nvdb/simd_dot.h"

#include <atomic>
#include <cstdint>
#include <cstring>


#if defined(__GNUC__) || defined(__clang__)
  #include <immintrin.h>
#endif

namespace nvdb {

  static std::atomic<bool> g_force_scalar{false};

  void set_force_scalar(bool v) { g_force_scalar.store(v, std::memory_order_relaxed); }

  static inline float dot_scalar(const float* a, const float* b, uint32_t dim) {
    double s = 0.0;
    for (uint32_t i = 0; i < dim; ++i) s += double(a[i]) * double(b[i]);
    return float(s);
  }

  #if defined(__GNUC__) || defined(__clang__)
  // AVX2+FMA kernel (compiled only for this function)
  __attribute__((target("avx2,fma")))
  static float dot_avx2_fma(const float* a, const float* b, uint32_t dim) {
    __m256 acc = _mm256_setzero_ps(); //initialize to zero
    uint32_t i = 0;

    // 8 floats per loop
    for (; i + 8 <= dim; i += 8) {
      __m256 va = _mm256_loadu_ps(a + i);  // load a[i , i+1 ,..., i+7]
      __m256 vb = _mm256_loadu_ps(b + i);  // load b[i , i+1 ,..., i+7]
      acc = _mm256_fmadd_ps(va, vb, acc);   // preform FMA: acc = va * vb + acc
    }

    // horizontal sum acc
    __m128 lo = _mm256_castps256_ps128(acc); // low 128 digits (first 4 floats)
    __m128 hi = _mm256_extractf128_ps(acc, 1); // high 128 digits (last 4 floats)
    __m128 sum = _mm_add_ps(lo, hi);  // combine 8 floats into 4 (vertical sum)
    sum = _mm_hadd_ps(sum, sum);       // combine 4 floats into 2 (horizontal sum)
    sum = _mm_hadd_ps(sum, sum);       // combine 2 floats into 1
    float out = _mm_cvtss_f32(sum);

    // tail (dim not multiple of 8)
    for (; i < dim; ++i) out += a[i] * b[i];
    return out;
  }
  #endif

  float dot_f32(const float* a, const float* b, uint32_t dim) {
    if (g_force_scalar.load(std::memory_order_relaxed)) {
      return dot_scalar(a, b, dim);
    }

  #if defined(__GNUC__) || defined(__clang__)
    // runtime detect CPU features (GCC/Clang)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
      return dot_avx2_fma(a, b, dim);
    }
  #endif
    return dot_scalar(a, b, dim);
  }


  // half bits -> float scalar (IEEE754)
  static inline float f16_to_f32_scalar(uint16_t h) {
    const uint32_t sign = (uint32_t(h & 0x8000u) << 16);
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;

    uint32_t out;
    if (exp == 0) {
      if (mant == 0) {
        out = sign; // ±0
      } else {
        // subnormal half -> normalized float
        exp = 1;
        while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
        mant &= 0x3FFu;
        uint32_t fexp = (exp - 15 + 127) & 0xFFu;
        uint32_t fmant = mant << 13;
        out = sign | (fexp << 23) | fmant;
      }
    } else if (exp == 0x1F) {
      // Inf/NaN
      out = sign | 0x7F800000u | (mant << 13);
    } else {
      // normal
      uint32_t fexp = (exp - 15 + 127) & 0xFFu;
      uint32_t fmant = mant << 13;
      out = sign | (fexp << 23) | fmant;
    }

    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
  }

  #if defined(__GNUC__) || defined(__clang__)
  __attribute__((target("avx2,fma,f16c")))
  static float dot_f32_f16base_avx2(const float* q, const uint16_t* x, uint32_t dim) {
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;

    for (; i + 8 <= dim; i += 8) {
      __m128i hx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)); // 8x half
      __m256 xf  = _mm256_cvtph_ps(hx);                                      // -> 8x float
      __m256 qf  = _mm256_loadu_ps(q + i);
      acc = _mm256_fmadd_ps(qf, xf, acc);
    }

    // horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float out = _mm_cvtss_f32(sum);

    for (; i < dim; ++i) out += q[i] * f16_to_f32_scalar(x[i]);
    return out;
  }
  #endif

  float dot_f32_f16base(const float* q_f32, const uint16_t* x_f16, uint32_t dim) {
  #if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
      return dot_f32_f16base_avx2(q_f32, x_f16, dim);
    }
  #endif
    double s = 0.0;
    for (uint32_t i = 0; i < dim; ++i) s += double(q_f32[i]) * double(f16_to_f32_scalar(x_f16[i]));
    return float(s);
  }
  

} // namespace nvdb

#include <cmath>   // for std::lrintf (if you use), not required here
#include <cstring>

#if defined(__GNUC__) || defined(__clang__)
  #include <immintrin.h>
#endif

namespace nvdb {

// scalar fallback
static inline float dot_f32_i8_scalar(const float* q, const int8_t* x, uint32_t dim, float scale) {
  double acc = 0.0;
  for (uint32_t i = 0; i < dim; ++i) {
    acc += double(q[i]) * double(x[i]);
  }
  return float(acc * double(scale));
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
static float dot_f32_i8_avx2(const float* q, const int8_t* x, uint32_t dim, float scale) {
  __m256 acc = _mm256_setzero_ps();
  uint32_t i = 0;

  for (; i + 16 <= dim; i += 16) {
    // load 16 int8
    __m128i bx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i));

    // sign-extend 16x int8 -> 16x int16 (stored in 256-bit)
    __m256i x16 = _mm256_cvtepi8_epi16(bx);

    // lower 8 int16 -> 8 int32
    __m128i x16_lo = _mm256_castsi256_si128(x16);
    __m256i x32_lo = _mm256_cvtepi16_epi32(x16_lo);
    __m256 xf_lo = _mm256_cvtepi32_ps(x32_lo);
    __m256 qf_lo = _mm256_loadu_ps(q + i);
    acc = _mm256_fmadd_ps(qf_lo, xf_lo, acc);

    // upper 8 int16 -> 8 int32
    __m128i x16_hi = _mm256_extracti128_si256(x16, 1);
    __m256i x32_hi = _mm256_cvtepi16_epi32(x16_hi);
    __m256 xf_hi = _mm256_cvtepi32_ps(x32_hi);
    __m256 qf_hi = _mm256_loadu_ps(q + i + 8);
    acc = _mm256_fmadd_ps(qf_hi, xf_hi, acc);
  }

  // horizontal sum acc
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  float out = _mm_cvtss_f32(sum);

  // tail
  for (; i < dim; ++i) out += q[i] * float(x[i]);

  return out * scale;
}
#endif

float dot_f32_i8base(const float* q_f32, const int8_t* x_i8, uint32_t dim, float scale) {
  if (g_force_scalar.load(std::memory_order_relaxed)) {
    return dot_f32_i8_scalar(q_f32, x_i8, dim, scale);
  }

#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
    return dot_f32_i8_avx2(q_f32, x_i8, dim, scale);
  }
#endif
  return dot_f32_i8_scalar(q_f32, x_i8, dim, scale);
}

} // namespace nvdb

