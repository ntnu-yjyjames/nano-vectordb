#pragma once
#include <cstdint>
#include <cstring>

namespace nvdb {

// IEEE754 half (uint16) -> float32 (scalar, portable)
inline float f16_to_f32_scalar(uint16_t h) {
  const uint32_t sign = (uint32_t(h & 0x8000u) << 16);
  uint32_t exp  = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;

  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign; // ±0
    } else {
      exp = 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
      mant &= 0x3FFu;
      uint32_t fexp  = (exp - 15 + 127) & 0xFFu;
      uint32_t fmant = mant << 13;
      out = sign | (fexp << 23) | fmant;
    }
  } else if (exp == 0x1F) {
    out = sign | 0x7F800000u | (mant << 13); // Inf/NaN
  } else {
    uint32_t fexp  = (exp - 15 + 127) & 0xFFu;
    uint32_t fmant = mant << 13;
    out = sign | (fexp << 23) | fmant;
  }

  float f;
  std::memcpy(&f, &out, sizeof(f));
  return f;
}

} // namespace nvdb
