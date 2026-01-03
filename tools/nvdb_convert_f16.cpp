#include "nvdb/vector_dataset.h"
#include "nvdb/vecbin_format.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#if defined(__GNUC__) || defined(__clang__)
  #include <immintrin.h>
#endif

// ----------------------------
// Scalar float32 -> float16 bits (IEEE 754)
// Robust reference-style conversion (round-to-nearest-even).
// ----------------------------
static inline uint16_t f32_to_f16_scalar(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));

  const uint32_t sign = (x >> 31) & 0x1;
  int32_t exp = int32_t((x >> 23) & 0xFF) - 127;      // unbiased
  uint32_t mant = x & 0x7FFFFF;

  // NaN/Inf
  if (((x >> 23) & 0xFF) == 0xFF) {
    if (mant == 0) { // Inf
      return uint16_t((sign << 15) | (0x1F << 10));
    }
    // NaN: preserve some payload, ensure mantissa non-zero
    uint16_t nan_m = uint16_t((mant >> 13) & 0x3FF);
    if (nan_m == 0) nan_m = 1;
    return uint16_t((sign << 15) | (0x1F << 10) | nan_m);
  }

  // Zero / subnormal in f32
  if (((x >> 23) & 0xFF) == 0) {
    // treat as zero (we could handle f32 subnormals, but not needed typically)
    return uint16_t(sign << 15);
  }

  // Normalize mantissa (implicit 1)
  mant |= 0x800000;

  // Overflow => Inf
  if (exp > 15) {
    return uint16_t((sign << 15) | (0x1F << 10));
  }

  // Underflow => subnormal/zero
  if (exp < -14) {
    // shift such that exponent becomes -14 => half subnormal
    const int shift = (-14 - exp);
    if (shift > 24) {
      return uint16_t(sign << 15); // too small => 0
    }
    // round mantissa
    uint32_t mant_shifted = mant >> (shift + 13);
    uint32_t remainder = mant & ((1u << (shift + 13)) - 1u);
    uint32_t halfway = 1u << (shift + 12);

    // round-to-nearest-even
    if (remainder > halfway || (remainder == halfway && (mant_shifted & 1u))) {
      mant_shifted++;
    }

    return uint16_t((sign << 15) | (mant_shifted & 0x3FFu));
  }

  // Normal half
  uint16_t hexp = uint16_t(exp + 15);
  // take top 10 bits of mantissa (after implicit 1) with rounding
  uint32_t mant_rounded = mant + 0x1000; // rounding bias for >>13
  // round-to-nearest-even: if exactly halfway, clear LSB
  if ((mant & 0x1FFFu) == 0x1000u) {
    mant_rounded &= ~1u;
  }

  uint16_t hmant = uint16_t((mant_rounded >> 13) & 0x3FFu);

  // handle mantissa overflow from rounding
  if (hmant == 0x400) {
    hmant = 0;
    hexp += 1;
    if (hexp >= 0x1F) { // overflow to inf
      return uint16_t((sign << 15) | (0x1F << 10));
    }
  }

  return uint16_t((sign << 15) | (hexp << 10) | hmant);
}

#if defined(__GNUC__) || defined(__clang__)
// Vectorized float32 -> float16 (8 at a time) using F16C
__attribute__((target("f16c")))
static inline void f32_to_f16_f16c(const float* src, uint16_t* dst, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    __m128i h = _mm256_cvtps_ph(v, 0); // round-to-nearest-even
    _mm_storeu_si128((__m128i*)(dst + i), h);
  }
  for (; i < n; ++i) dst[i] = f32_to_f16_scalar(src[i]);
}
#endif

static inline void convert_f32_to_f16(const float* src, uint16_t* dst, size_t n) {
#if defined(__GNUC__) || defined(__clang__)
  // runtime detect f16c
  if (__builtin_cpu_supports("f16c")) {
    f32_to_f16_f16c(src, dst, n);
    return;
  }
#endif
  for (size_t i = 0; i < n; ++i) dst[i] = f32_to_f16_scalar(src[i]);
}

static void write_all(std::ofstream& ofs, const char* p, size_t nbytes) {
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
  if (argc < 3) {
    std::cerr << "Usage: nvdb_convert_f16 <input_f32.vecbin> <output_f16.vecbin> [batch_vecs=4096]\n";
    return 1;
  }

  const std::string in_path = argv[1];
  const std::string out_path = argv[2];
  const uint64_t batch_vecs = (argc >= 4) ? std::stoull(argv[3]) : 4096ULL;

  nvdb::VectorDataset ds;
  ds.load(in_path);

  if (ds.dtype() != static_cast<uint32_t>(nvdb::DType::Float32)) {
    std::cerr << "Input dtype must be Float32. Got dtype=" << ds.dtype() << "\n";
    return 2;
  }

  const uint64_t n = ds.count();
  const uint32_t dim = ds.dim();
  if (n == 0 || dim == 0) {
    std::cerr << "Invalid dataset: count or dim is zero\n";
    return 3;
  }

  // Write vecbin64 header (Float16)
  nvdb::VecbinHeader hdr{};
  hdr.magic = nvdb::kMagic;
  hdr.version = nvdb::kVersion;
  hdr.dtype = static_cast<uint32_t>(nvdb::DType::Float16);
  hdr.dim = dim;
  hdr.count = n;

  std::ofstream ofs(out_path, std::ios::binary);
  if (!ofs) {
    std::cerr << "Failed to open output: " << out_path << "\n";
    return 4;
  }

  ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  if (!ofs) {
    std::cerr << "Failed to write vecbin header\n";
    return 5;
  }

  // Batch convert: f32 -> f16 bits
  const uint64_t B = std::max<uint64_t>(1, batch_vecs);
  std::vector<uint16_t> out_buf;
  out_buf.resize(static_cast<size_t>(std::min<uint64_t>(B, n)) * static_cast<size_t>(dim));

  uint64_t i = 0;
  while (i < n) {
    const uint64_t take = std::min<uint64_t>(B, n - i);

    uint16_t* dst = out_buf.data();
    for (uint64_t bi = 0; bi < take; ++bi) {
      const float* v = ds.vector_ptr_f32(i + bi);
      convert_f32_to_f16(v, dst + static_cast<size_t>(bi) * dim, dim);
    }

    const size_t bytes = static_cast<size_t>(take) * static_cast<size_t>(dim) * sizeof(uint16_t);
    write_all(ofs, reinterpret_cast<const char*>(out_buf.data()), bytes);

    i += take;
  }

  ofs.close();
  std::cout << "Wrote FP16 vecbin64: " << out_path
            << " count=" << n << " dim=" << dim
            << " (payload bytes=" << (size_t)n * (size_t)dim * sizeof(uint16_t) << ")\n";
  return 0;
}
