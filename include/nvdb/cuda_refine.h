#pragma once
#include <cstdint>
#include <vector>

namespace nvdb {

struct CudaRefineTiming {
  double h2d_ms   = 0;
  double kernel_ms= 0;
  double d2h_ms   = 0;
  double total_ms = 0;
};

void cuda_l2_topk_batch(
  const void* base_ptr,
  uint32_t base_dtype,   // 1=fp32, 2=fp16
  uint64_t N,
  uint32_t D,
  const float* queries_f32,      // Q*D
  const uint32_t* cand_ids,      // Q*R
  uint32_t Q,
  uint32_t R,                    // <= 500
  uint32_t K,                    // final top-k, e.g. 10
  std::vector<uint32_t>& out_topk_ids, // Q*K
  std::vector<float>& out_topk_dist,   // Q*K
  CudaRefineTiming* timing = nullptr
);

} // namespace nvdb
