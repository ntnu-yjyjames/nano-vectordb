#pragma once
#include <cstdint>
#include <vector>

namespace nvdb {

struct CudaRefineTiming {
  float h2d_ms=0, kernel_ms=0, d2h_ms=0, total_ms=0;
  uint32_t threads=0;
  uint32_t nwarps=0;
  uint32_t K=0;
  uint32_t R=0;
  size_t shmem_bytes=0;
  // debug timing (clock64 sample)
  uint32_t dbg_q = 0;
  double dbg_dist_cycles_avg = 0.0;
  double dbg_write_cycles_avg = 0.0;
  double dbg_merge_cycles_avg = 0.0;
  double dbg_dist_pct = 0.0;
  double dbg_write_pct = 0.0;
  double dbg_merge_pct = 0.0;
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
