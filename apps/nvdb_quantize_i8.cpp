// apps/nvdb_quantize_i8.cpp
#include "nvdb/vector_dataset.h"
#include "nvdb/vecbin_format.h"
#include "nvdb/f16_scalar.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>

static int8_t clamp_i8(int v) {
  if (v > 127) return 127;
  if (v < -127) return -127;
  return (int8_t)v;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: nvdb_quantize_i8 <in.vecbin> <out_i8.vecbin>\n";
    return 1;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];

  nvdb::VectorDataset base;
  base.load(in_path);

  const uint64_t N = base.count();
  const uint32_t D = base.dim();
  const uint32_t dt = base.dtype();
  if (dt != (uint32_t)nvdb::DType::Float32 && dt != (uint32_t)nvdb::DType::Float16) {
    std::cerr << "Input must be f32/f16. dtype=" << dt << "\n";
    return 2;
  }

  std::ofstream out(out_path, std::ios::binary);
  if (!out) { std::cerr << "Failed to open output\n"; return 3; }

  // write vecbin64 header (dtype=int8)
  nvdb::VecbinHeader h{};
  h.magic = nvdb::kMagic;
  h.version = nvdb::kVersion;
  h.dtype = (uint32_t)nvdb::DType::Int8;
  h.dim = D;
  h.count = N;
  out.write(reinterpret_cast<const char*>(&h), sizeof(h));

  // stream rows: write int8 payload then scales
  // We will write payload first, scales later => need two passes OR buffer scales.
  // Minimal approach: write payload, store scales in RAM (N floats).
  std::vector<float> scales(N);

  std::vector<int8_t> row8(D);
  std::vector<float>  rowf(D);

  for (uint64_t i = 0; i < N; ++i) {
    // load row into float
    if (dt == (uint32_t)nvdb::DType::Float32) {
      const float* v = base.vector_ptr_f32(i);
      std::copy(v, v + D, rowf.begin());
    } else {
      const uint16_t* v16 = base.vector_ptr_f16(i);
      
      for (uint32_t j = 0; j < D; ++j) {
        rowf[j] = nvdb::f16_to_f32_scalar(v16[j]);
       }

    }

    float max_abs = 0.f;
    for (uint32_t j = 0; j < D; ++j) max_abs = std::max(max_abs, std::fabs(rowf[j]));
    float scale = (max_abs > 0.f) ? (max_abs / 127.f) : 1.f;
    scales[i] = scale;

    const float inv = 1.0f / scale;
    for (uint32_t j = 0; j < D; ++j) {
      int q = (int)std::lrint(rowf[j] * inv);
      row8[j] = clamp_i8(q);
    }
    out.write(reinterpret_cast<const char*>(row8.data()), row8.size());
  }

  // write scales after payload
  out.write(reinterpret_cast<const char*>(scales.data()), scales.size() * sizeof(float));

  std::cerr << "Wrote int8 vecbin64: N=" << N << " D=" << D << "\n";
  return 0;
}
