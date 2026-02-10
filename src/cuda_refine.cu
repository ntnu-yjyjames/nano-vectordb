#include "nvdb/cuda_refine.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>  
#include <string>

#ifndef NVDB_CUDA_KMAX
#define NVDB_CUDA_KMAX 64
#endif

namespace nvdb {

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

// ---- base cache (process-local) ----
static void*    g_base_dev   = nullptr;
static uint64_t g_base_N     = 0;
static uint32_t g_base_D     = 0;
static uint32_t g_base_dt    = 0;
static size_t   g_base_bytes = 0;

static void free_base_cache() {
  if (g_base_dev) {
    cudaFree(g_base_dev);
    g_base_dev = nullptr;
  }
  g_base_N = 0; g_base_D = 0; g_base_dt = 0; g_base_bytes = 0;
}

// ---- workspace cache (process-local) ----
static cudaStream_t g_stream = nullptr;

static float*    g_q_dev      = nullptr;   // Q*D
static uint32_t* g_cand_dev   = nullptr;   // Q*R
static uint32_t* g_out_id_dev = nullptr;   // Q*K
static float*    g_out_dist_dev = nullptr; // Q*K

static uint32_t g_cap_Q = 0;
static uint32_t g_cap_R = 0;
static uint32_t g_cap_D = 0;
static uint32_t g_cap_K = 0;

static void free_workspace_cache() {
  if (g_q_dev) cudaFree(g_q_dev), g_q_dev = nullptr;
  if (g_cand_dev) cudaFree(g_cand_dev), g_cand_dev = nullptr;
  if (g_out_id_dev) cudaFree(g_out_id_dev), g_out_id_dev = nullptr;
  if (g_out_dist_dev) cudaFree(g_out_dist_dev), g_out_dist_dev = nullptr;
  g_cap_Q = g_cap_R = g_cap_D = g_cap_K = 0;

  if (g_stream) cudaStreamDestroy(g_stream), g_stream = nullptr;
}

// ---- host pinned cache (process-local) ----
static float*    g_h_q_pinned        = nullptr; // Q*D
static uint32_t* g_h_cand_pinned     = nullptr; // Q*R
static uint32_t* g_h_out_id_pinned   = nullptr; // Q*K
static float*    g_h_out_dist_pinned = nullptr; // Q*K (optional)

static size_t g_h_q_bytes        = 0;
static size_t g_h_cand_bytes     = 0;
static size_t g_h_out_id_bytes   = 0;
static size_t g_h_out_dist_bytes = 0;

static void free_host_pinned_cache() {
  if (g_h_q_pinned)        cudaFreeHost(g_h_q_pinned), g_h_q_pinned = nullptr;
  if (g_h_cand_pinned)     cudaFreeHost(g_h_cand_pinned), g_h_cand_pinned = nullptr;
  if (g_h_out_id_pinned)   cudaFreeHost(g_h_out_id_pinned), g_h_out_id_pinned = nullptr;
  if (g_h_out_dist_pinned) cudaFreeHost(g_h_out_dist_pinned), g_h_out_dist_pinned = nullptr;

  g_h_q_bytes = g_h_cand_bytes = g_h_out_id_bytes = g_h_out_dist_bytes = 0;
}

// grow-only pinned buffers (realloc only when not enough)
static void ensure_host_pinned(uint32_t Q, uint32_t D, uint32_t R, uint32_t K, bool want_dist) {
  const size_t need_q   = (size_t)Q * (size_t)D * sizeof(float);
  const size_t need_c   = (size_t)Q * (size_t)R * sizeof(uint32_t);
  const size_t need_oid = (size_t)Q * (size_t)K * sizeof(uint32_t);
  const size_t need_od  = (size_t)Q * (size_t)K * sizeof(float);

  if (!g_h_q_pinned || g_h_q_bytes < need_q) {
    if (g_h_q_pinned) cudaFreeHost(g_h_q_pinned);
    ck(cudaHostAlloc((void**)&g_h_q_pinned, need_q, cudaHostAllocDefault), "cudaHostAlloc h_q");
    g_h_q_bytes = need_q;
  }
  if (!g_h_cand_pinned || g_h_cand_bytes < need_c) {
    if (g_h_cand_pinned) cudaFreeHost(g_h_cand_pinned);
    ck(cudaHostAlloc((void**)&g_h_cand_pinned, need_c, cudaHostAllocDefault), "cudaHostAlloc h_cand");
    g_h_cand_bytes = need_c;
  }
  if (!g_h_out_id_pinned || g_h_out_id_bytes < need_oid) {
    if (g_h_out_id_pinned) cudaFreeHost(g_h_out_id_pinned);
    ck(cudaHostAlloc((void**)&g_h_out_id_pinned, need_oid, cudaHostAllocDefault), "cudaHostAlloc h_out_id");
    g_h_out_id_bytes = need_oid;
  }

  if (want_dist) {
    if (!g_h_out_dist_pinned || g_h_out_dist_bytes < need_od) {
      if (g_h_out_dist_pinned) cudaFreeHost(g_h_out_dist_pinned);
      ck(cudaHostAlloc((void**)&g_h_out_dist_pinned, need_od, cudaHostAllocDefault), "cudaHostAlloc h_out_dist");
      g_h_out_dist_bytes = need_od;
    }
  }
}




static bool g_atexit_set = false;

static void ensure_stream() {
  if (!g_stream) ck(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking), "create stream");
  if (!g_atexit_set) {
    g_atexit_set = true;
    std::atexit(free_workspace_cache);
    std::atexit(free_base_cache);
    std::atexit(free_host_pinned_cache);
  }
}

// Ensure device workspace capacity for (Q,D,R,K) (grow-only)
static void ensure_workspace(uint32_t Q, uint32_t D, uint32_t R, uint32_t K) {
  ensure_stream();

  bool need_realloc = (Q > g_cap_Q) || (D > g_cap_D) || (R > g_cap_R) || (K > g_cap_K) ||
                      !g_q_dev || !g_cand_dev || !g_out_id_dev || !g_out_dist_dev;

  if (!need_realloc) return;

  // grow strategy: keep current buffers if possible; simplest is full free+realloc
  // (OK for MVP; later you can make it finer-grained)
  if (g_q_dev) cudaFree(g_q_dev), g_q_dev = nullptr;
  if (g_cand_dev) cudaFree(g_cand_dev), g_cand_dev = nullptr;
  if (g_out_id_dev) cudaFree(g_out_id_dev), g_out_id_dev = nullptr;
  if (g_out_dist_dev) cudaFree(g_out_dist_dev), g_out_dist_dev = nullptr;

  // round up capacities to reduce realloc frequency
  auto round_up = [](uint32_t x) {
    uint32_t p = 1;
    while (p < x) p <<= 1;
    return p;
  };
  g_cap_Q = round_up(Q);
  g_cap_D = D;          // D fixed 384; keep exact
  g_cap_R = round_up(R);
  //g_cap_K = K;          // K small; keep exact
  g_cap_K = std::max(g_cap_K, K);


  ck(cudaMalloc(&g_q_dev, (size_t)g_cap_Q * (size_t)g_cap_D * sizeof(float)), "cudaMalloc q(cache)");
  ck(cudaMalloc(&g_cand_dev, (size_t)g_cap_Q * (size_t)g_cap_R * sizeof(uint32_t)), "cudaMalloc cand(cache)");
  ck(cudaMalloc(&g_out_id_dev, (size_t)g_cap_Q * (size_t)g_cap_K * sizeof(uint32_t)), "cudaMalloc out_id(cache)");
  ck(cudaMalloc(&g_out_dist_dev, (size_t)g_cap_Q * (size_t)g_cap_K * sizeof(float)), "cudaMalloc out_dist(cache)");
}


static void ensure_base_on_gpu(const void* base_ptr, uint32_t base_dtype, uint64_t N, uint32_t D) {
  size_t need_bytes = 0;
  if (base_dtype == 1) need_bytes = (size_t)N * (size_t)D * sizeof(float);
  else if (base_dtype == 2) need_bytes = (size_t)N * (size_t)D * sizeof(uint16_t);
  else {
    std::fprintf(stderr, "ensure_base_on_gpu: unsupported base_dtype=%u\n", base_dtype);
    std::exit(2);
  }

  const bool same =
    (g_base_dev != nullptr) &&
    (g_base_N == N) && (g_base_D == D) &&
    (g_base_dt == base_dtype) &&
    (g_base_bytes == need_bytes);

  if (same) return;

  free_base_cache();
  ck(cudaMalloc(&g_base_dev, need_bytes), "cudaMalloc base(cache)");
  ck(cudaMemcpy(g_base_dev, base_ptr, need_bytes, cudaMemcpyHostToDevice), "H2D base(cache)");

  g_base_N = N;
  g_base_D = D;
  g_base_dt = base_dtype;
  g_base_bytes = need_bytes;
}

// ---- small helpers ----
__device__ __forceinline__ float load_f16(const __half* p) { return __half2float(*p); }

template<int KMAX>
__device__ __forceinline__ void topk_insert(float (&best_d)[KMAX], uint32_t (&best_id)[KMAX],
                                            float dist, uint32_t id, int K) {
  // keep best_d sorted ascending (smallest first), size K
  if (dist >= best_d[K-1]) return;
  int pos = K - 1;
  best_d[pos] = dist;
  best_id[pos] = id;
  while (pos > 0 && best_d[pos] < best_d[pos-1]) {
    float td = best_d[pos-1]; best_d[pos-1] = best_d[pos]; best_d[pos] = td;
    uint32_t ti = best_id[pos-1]; best_id[pos-1] = best_id[pos]; best_id[pos] = ti;
    --pos;
  }
}
#ifndef NVDB_WARP_SIZE
#define NVDB_WARP_SIZE 32
#endif

template<int KMAX>
__device__ __forceinline__ void topk_unsorted_init(float (&d)[KMAX], uint32_t (&id)[KMAX]) {
  #pragma unroll
  for (int i = 0; i < KMAX; ++i) { d[i] = 1e30f; id[i] = 0xFFFFFFFFu; }
}

template<int KMAX>
__device__ __forceinline__ void topk_unsorted_push(
    float (&d)[KMAX], uint32_t (&id)[KMAX],
    int& filled, int K, int& max_pos, float& max_val,
    float dist, uint32_t vid)
{
  if (filled < K) {
    d[filled]  = dist;
    id[filled] = vid;
    if (dist > max_val) { max_val = dist; max_pos = filled; }
    filled++;
    return;
  }
  if (dist >= max_val) return;

  d[max_pos]  = dist;
  id[max_pos] = vid;

  max_val = d[0];
  max_pos = 0;
  #pragma unroll
  for (int i = 1; i < KMAX; ++i) {
    if (i >= K) break;
    if (d[i] > max_val) { max_val = d[i]; max_pos = i; }
  }
}

// merge one K-list from shared into an unsorted topK
template<int KMAX>
__device__ __forceinline__ void topk_unsorted_merge_from_shared(
    float (&d)[KMAX], uint32_t (&id)[KMAX],
    int& filled, int K, int& max_pos, float& max_val,
    const float* src_d, const uint32_t* src_id)
{
  #pragma unroll
  for (int i = 0; i < KMAX; ++i) {
    if (i >= K) break;
    uint32_t vid = src_id[i];
    float dist   = src_d[i];
    if (vid == 0xFFFFFFFFu) continue;
    topk_unsorted_push<KMAX>(d, id, filled, K, max_pos, max_val, dist, vid);
  }
}

// sort K elements ascending by dist (tiny K, only used at the final output stage)
template<int KMAX>
__device__ __forceinline__ void topk_sort_small(float (&d)[KMAX], uint32_t (&id)[KMAX], int K) {
  for (int i = 0; i < K; ++i) {
    int best = i;
    for (int j = i + 1; j < K; ++j) {
      if (d[j] < d[best]) best = j;
    }
    if (best != i) {
      float td = d[i]; d[i] = d[best]; d[best] = td;
      uint32_t ti = id[i]; id[i] = id[best]; id[best] = ti;
    }
  }
}




/*__device__ __forceinline__ float l2_fp16_base(const __half* base, const float* qv, uint64_t id, uint32_t D) {
  const __half* bv = base + id * (uint64_t)D;
  float acc = 0.f;
  #pragma unroll 4
  for (uint32_t j = 0; j < D; ++j) {
    float diff = qv[j] - __half2float(bv[j]);
    acc += diff * diff;
  }
  return acc;
}*/
__device__ __forceinline__ float l2_fp16_base_half2(
    const __half* __restrict__ base,
    const float* __restrict__ qv,
    uint64_t id,
    uint32_t D)
{
  const __half* __restrict__ bv = base + id * (uint64_t)D;

  float acc0 = 0.f;
  float acc1 = 0.f; // ILP: 兩個 accumulator
  float acc2 = 0.f;
  float acc3 = 0.f;

  // 假設 D 是偶數（384）
  // 注意：half2 對齊通常 OK（bv 來自 cudaMalloc 的 base buffer）
  const half2* __restrict__ bv2 = reinterpret_cast<const half2*>(bv);

  // qv 是 float*，我們用 float2 來讀
  const float2* __restrict__ q2 = reinterpret_cast<const float2*>(qv);

  const uint32_t D2 = D / 2;

  // 手動 unroll：每次處理 4 個 half2（=8 維）
  for (uint32_t j = 0; j + 3 < D2; j += 4) {
    float2 qb0 = q2[j + 0];
    float2 qb1 = q2[j + 1];
    float2 qb2 = q2[j + 2];
    float2 qb3 = q2[j + 3];

    float2 b0 = __half22float2(bv2[j + 0]);
    float2 b1 = __half22float2(bv2[j + 1]);
    float2 b2 = __half22float2(bv2[j + 2]);
    float2 b3 = __half22float2(bv2[j + 3]);

    float dx0 = qb0.x - b0.x; float dy0 = qb0.y - b0.y;
    float dx1 = qb1.x - b1.x; float dy1 = qb1.y - b1.y;
    float dx2 = qb2.x - b2.x; float dy2 = qb2.y - b2.y;
    float dx3 = qb3.x - b3.x; float dy3 = qb3.y - b3.y;

    acc0 = fmaf(dx0, dx0, acc0); acc0 = fmaf(dy0, dy0, acc0);
    acc1 = fmaf(dx1, dx1, acc1); acc1 = fmaf(dy1, dy1, acc1);
    acc2 = fmaf(dx2, dx2, acc2); acc2 = fmaf(dy2, dy2, acc2);
    acc3 = fmaf(dx3, dx3, acc3); acc3 = fmaf(dy3, dy3, acc3);
  }

  // tail（理論上 D2=192，j=0..188 step4，剛好無尾；保險留著）
  for (uint32_t j = (D2 & ~3u); j < D2; ++j) {
    float2 qb = q2[j];
    float2 b  = __half22float2(bv2[j]);
    float dx = qb.x - b.x;
    float dy = qb.y - b.y;
    acc0 = fmaf(dx, dx, acc0);
    acc0 = fmaf(dy, dy, acc0);
  }

  return (acc0 + acc1) + (acc2 + acc3);
}
__device__ __forceinline__ float l2_fp32_base(const float* base, const float* qv, uint64_t id, uint32_t D) {
  const float* bv = base + id * (uint64_t)D;
  float acc = 0.f;
  #pragma unroll 4
  for (uint32_t j = 0; j < D; ++j) {
    float diff = qv[j] - bv[j];
    acc += diff * diff;
  }
  return acc;
}

/*
 * Kernel: one block per query.
 * - blockDim threads compute distances for their share of candidates (R <= 500)
 * - each thread keeps local topK in registers
 * - write local topK to shared
 * - thread0 merges all local topKs -> final topK
 *
 * Shared memory usage: blockDim*K*(dist+id)
 * with blockDim=256, K=10 => 2560*(4+4)=~20 KB, OK.
 */

template<int KMAX>
__global__ void l2_topk_fp16_kernel(
    const __half* base, uint64_t N, uint32_t D,
    const float* queries, uint32_t Q,
    const uint32_t* cand, uint32_t R,
    uint32_t K,
    uint32_t* out_ids, float* out_dist)
{
  const uint32_t q = blockIdx.x;
  if (q >= Q) return;

  extern __shared__ unsigned char smem[];
  float* s_dist = (float*)smem;
  uint32_t* s_id = (uint32_t*)(s_dist + (size_t)blockDim.x * (size_t)K);

  // ---- per-thread unsorted topK ----
  float best_d[KMAX];
  uint32_t best_id[KMAX];
  topk_unsorted_init<KMAX>(best_d, best_id);

  int filled = 0;
  int max_pos = 0;
  float max_val = -1.0f; // while filling, track current max

  const float* qv = queries + (uint64_t)q * D;

  for (uint32_t r = threadIdx.x; r < R; r += blockDim.x) {
    uint32_t id = cand[(uint64_t)q * R + r];
    if (id >= N) continue;

    float dist = l2_fp16_base_half2(base, qv, (uint64_t)id, D);
    topk_unsorted_push<KMAX>(best_d, best_id, filled, (int)K, max_pos, max_val, dist, id);
  }

  // write per-thread K items to shared
  const size_t off = (size_t)threadIdx.x * (size_t)K;
  #pragma unroll
  for (int i = 0; i < KMAX; ++i) {
    if (i >= (int)K) break;
    s_dist[off + i] = best_d[i];
    s_id[off + i]   = best_id[i];
  }
  __syncthreads();

  // ---- thread0 merges (still ok) ----
  if (threadIdx.x == 0) {
    float g_d[KMAX];
    uint32_t g_id[KMAX];
    topk_unsorted_init<KMAX>(g_d, g_id);

    int g_filled = 0;
    int g_max_pos = 0;
    float g_max_val = -1.0f;

    for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
      const size_t toff = (size_t)t * (size_t)K;
      #pragma unroll
      for (int i = 0; i < KMAX; ++i) {
        if (i >= (int)K) break;
        uint32_t id = s_id[toff + i];
        float dist  = s_dist[toff + i];
        if (id == 0xFFFFFFFFu) continue;
        topk_unsorted_push<KMAX>(g_d, g_id, g_filled, (int)K, g_max_pos, g_max_val, dist, id);
      }
    }

    // sort final K (K<=KMAX<=64, small)
    // simple selection sort / partial sort in registers:
    for (uint32_t i = 0; i < K; ++i) {
      uint32_t best = i;
      for (uint32_t j = i + 1; j < K; ++j) {
        if (g_d[j] < g_d[best]) best = j;
      }
      if (best != i) {
        float td = g_d[i]; g_d[i] = g_d[best]; g_d[best] = td;
        uint32_t ti = g_id[i]; g_id[i] = g_id[best]; g_id[best] = ti;
      }
    }

    const size_t out_off = (size_t)q * (size_t)K;
    for (uint32_t i = 0; i < K; ++i) {
      out_ids[out_off + i]  = g_id[i];
      out_dist[out_off + i] = g_d[i];
    }
  }
}

//*/
template<int KMAX>
__global__ void l2_topk_fp16_kernel_warpmerge(
    const __half* base, uint64_t N, uint32_t D,
    const float* queries, uint32_t Q,
    const uint32_t* cand, uint32_t R,
    uint32_t K,
    uint32_t* out_ids, float* out_dist)
{
  const uint32_t q = blockIdx.x;
  if (q >= Q) return;

  const uint32_t tid   = threadIdx.x;
  const uint32_t lane  = tid & 31u;
  const uint32_t warp  = tid >> 5;               // /32
  const uint32_t nwarps = (blockDim.x + 31) >> 5;

  // Shared layout:
  // per-thread: [blockDim * K] dist + id
  // per-warp:   [nwarps  * K] dist + id
  extern __shared__ unsigned char smem[];
  float*    s_dist_t = reinterpret_cast<float*>(smem);
  uint32_t* s_id_t   = reinterpret_cast<uint32_t*>(s_dist_t + (size_t)blockDim.x * (size_t)K);

  float*    s_dist_w = reinterpret_cast<float*>(s_id_t + (size_t)blockDim.x * (size_t)K);
  uint32_t* s_id_w   = reinterpret_cast<uint32_t*>(s_dist_w + (size_t)nwarps * (size_t)K);

  // -----------------------------
  // Stage 1: per-thread topK (unsorted + maxslot)
  // -----------------------------
  float best_d[KMAX];
  uint32_t best_id[KMAX];
  topk_unsorted_init<KMAX>(best_d, best_id);

  int filled = 0;
  int max_pos = 0;
  float max_val = -1.0f;

  const float* qv = queries + (uint64_t)q * (uint64_t)D;

  for (uint32_t r = tid; r < R; r += blockDim.x) {
    uint32_t id = cand[(uint64_t)q * (uint64_t)R + r];
    if (id >= N) continue;

    float dist = l2_fp16_base_half2(base, qv, (uint64_t)id, D);
    topk_unsorted_push<KMAX>(best_d, best_id, filled, (int)K, max_pos, max_val, dist, id);
  }

  // write per-thread K items to shared (unsorted order ok)
  {
    const size_t base_off = (size_t)tid * (size_t)K;
    #pragma unroll
    for (int i = 0; i < KMAX; ++i) {
      if (i >= (int)K) break;
      s_dist_t[base_off + i] = best_d[i];
      s_id_t[base_off + i]   = best_id[i];
    }
  }
  __syncthreads();

  // -----------------------------
  // Stage 2: warp leader merges 32 threads -> warp topK (unsorted + maxslot)
  // -----------------------------
  if (lane == 0) {
    float w_d[KMAX];
    uint32_t w_id[KMAX];
    topk_unsorted_init<KMAX>(w_d, w_id);

    int w_filled = 0;
    int w_max_pos = 0;
    float w_max_val = -1.0f;

    const uint32_t t0 = warp * 32u;
    for (uint32_t t = 0; t < 32u; ++t) {
      const uint32_t th = t0 + t;
      if (th >= (uint32_t)blockDim.x) break;
      const size_t off = (size_t)th * (size_t)K;
      topk_unsorted_merge_from_shared<KMAX>(
        w_d, w_id, w_filled, (int)K, w_max_pos, w_max_val,
        &s_dist_t[off], &s_id_t[off]
      );
    }

    // store warp topK to shared
    const size_t woff = (size_t)warp * (size_t)K;
    #pragma unroll
    for (int i = 0; i < KMAX; ++i) {
      if (i >= (int)K) break;
      s_dist_w[woff + i] = w_d[i];
      s_id_w[woff + i]   = w_id[i];
    }
  }
  __syncthreads();

  // -----------------------------
  // Stage 3: thread0 merges nwarps lists -> global topK, then sort once
  // -----------------------------
  if (tid == 0) {
    float g_d[KMAX];
    uint32_t g_id[KMAX];
    topk_unsorted_init<KMAX>(g_d, g_id);

    int g_filled = 0;
    int g_max_pos = 0;
    float g_max_val = -1.0f;

    for (uint32_t w = 0; w < nwarps; ++w) {
      const size_t woff = (size_t)w * (size_t)K;
      topk_unsorted_merge_from_shared<KMAX>(
        g_d, g_id, g_filled, (int)K, g_max_pos, g_max_val,
        &s_dist_w[woff], &s_id_w[woff]
      );
    }

    // final sort (K small)
    topk_sort_small<KMAX>(g_d, g_id, (int)K);

    const size_t out_off = (size_t)q * (size_t)K;
    #pragma unroll
    for (int i = 0; i < KMAX; ++i) {
      if (i >= (int)K) break;
      out_ids[out_off + i]  = g_id[i];
      out_dist[out_off + i] = g_d[i];
    }
  }
}


template<int KMAX>
__global__ void l2_topk_fp32_kernel(
    const float* base, uint64_t N, uint32_t D,
    const float* queries, uint32_t Q,
    const uint32_t* cand, uint32_t R,
    uint32_t K,
    uint32_t* out_ids, float* out_dist)
{
  const uint32_t q = blockIdx.x;
  if (q >= Q) return;

  extern __shared__ unsigned char smem[];
  float* s_dist = (float*)smem;
  uint32_t* s_id = (uint32_t*)(s_dist + (size_t)blockDim.x * (size_t)K);

  float best_d[KMAX];
  uint32_t best_id[KMAX];
  #pragma unroll
  for (int i = 0; i < KMAX; ++i) { best_d[i] = 1e30f; best_id[i] = 0xFFFFFFFFu; }

  const float* qv = queries + (uint64_t)q * D;

  for (uint32_t r = threadIdx.x; r < R; r += blockDim.x) {
    uint32_t id = cand[(uint64_t)q * R + r];
    if (id >= N) continue;
    float dist = l2_fp32_base(base, qv, (uint64_t)id, D);
    topk_insert<KMAX>(best_d, best_id, dist, id, (int)K);
  }

  for (uint32_t i = 0; i < K; ++i) {
    const size_t off = (size_t)threadIdx.x * (size_t)K + i;
    s_dist[off] = best_d[i];
    s_id[off]   = best_id[i];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float g_d[KMAX];
    uint32_t g_id[KMAX];
    #pragma unroll
    for (int i = 0; i < KMAX; ++i) { g_d[i] = 1e30f; g_id[i] = 0xFFFFFFFFu; }

    const uint32_t nT = blockDim.x;
    for (uint32_t t = 0; t < nT; ++t) {
      const size_t base_off = (size_t)t * (size_t)K;
      for (uint32_t i = 0; i < K; ++i) {
        float dist = s_dist[base_off + i];
        uint32_t id = s_id[base_off + i];
        if (id == 0xFFFFFFFFu) continue;
        topk_insert<KMAX>(g_d, g_id, dist, id, (int)K);
      }
    }

    const size_t out_off = (size_t)q * (size_t)K;
    for (uint32_t i = 0; i < K; ++i) {
      out_ids[out_off + i]  = g_id[i];
      out_dist[out_off + i] = g_d[i];
    }
  }
}

void cuda_l2_topk_batch(
  const void* base_ptr,
  uint32_t base_dtype,
  uint64_t N,
  uint32_t D,
  const float* queries_f32,
  const uint32_t* cand_ids,
  uint32_t Q,
  uint32_t R,
  uint32_t K,
  std::vector<uint32_t>& out_topk_ids,
  std::vector<float>& out_topk_dist,
  CudaRefineTiming* timing)
{
  if (K == 0 || Q == 0 || R == 0) {
    out_topk_ids.clear(); out_topk_dist.clear();
    if (timing) *timing = {};
    return;
  }
  if (K > NVDB_CUDA_KMAX) {
    std::fprintf(stderr,
      "cuda_l2_topk_batch: K=%u not supported (K<=%d)\n", K, NVDB_CUDA_KMAX);
    std::exit(3);
  }



  const int pinned = std::getenv("CUDA_PINNED") ? std::atoi(std::getenv("CUDA_PINNED")) : 0;
  const int return_dist_env = std::getenv("CUDA_RETURN_DIST") ? std::atoi(std::getenv("CUDA_RETURN_DIST")) : 1;
  const bool use_pinned = (pinned != 0);
  const bool want_dist = (return_dist_env != 0);

  const char* km = std::getenv("CUDA_KERNEL_MODE");
  const bool use_warpmerge = (km && std::string(km) == "warpmerge"); // default baseline


    ensure_base_on_gpu(base_ptr, base_dtype, N, D);
    ensure_workspace(Q, D, R, K);

    // output host vectors (ids always; dist optional)
    out_topk_ids.assign((size_t)Q * (size_t)K, 0xFFFFFFFFu);
    if (want_dist) out_topk_dist.assign((size_t)Q * (size_t)K, 1e30f);
    else out_topk_dist.clear();

    // pinned host buffers (optional)
    const float*    h_q_src   = queries_f32;
    const uint32_t* h_c_src   = cand_ids;
    uint32_t*       h_oid_dst = out_topk_ids.data();
    float*          h_od_dst  = want_dist ? out_topk_dist.data() : nullptr;

    if (use_pinned) {
      ensure_host_pinned(Q, D, R, K, want_dist);

      // pack into pinned
      std::memcpy(g_h_q_pinned, queries_f32, (size_t)Q * (size_t)D * sizeof(float));
      std::memcpy(g_h_cand_pinned, cand_ids, (size_t)Q * (size_t)R * sizeof(uint32_t));

      h_q_src = g_h_q_pinned;
      h_c_src = g_h_cand_pinned;

      h_oid_dst = g_h_out_id_pinned;
      h_od_dst  = want_dist ? g_h_out_dist_pinned : nullptr;
    }

    // events
    cudaEvent_t ev0, ev1, ev2, ev3;
    ck(cudaEventCreate(&ev0), "event create");
    ck(cudaEventCreate(&ev1), "event create");
    ck(cudaEventCreate(&ev2), "event create");
    ck(cudaEventCreate(&ev3), "event create");

    ck(cudaEventRecord(ev0, g_stream), "event record 0");

    // H2D async (from pinned if enabled)
    ck(cudaMemcpyAsync(g_q_dev, h_q_src, (size_t)Q * (size_t)D * sizeof(float),
                      cudaMemcpyHostToDevice, g_stream), "H2D queries async");
    ck(cudaMemcpyAsync(g_cand_dev, h_c_src, (size_t)Q * (size_t)R * sizeof(uint32_t),
                      cudaMemcpyHostToDevice, g_stream), "H2D cand async");

    ck(cudaEventRecord(ev1, g_stream), "event record 1");

    // threads selection (initial)
    int threads = 256;
    if (R < 256) threads = 128;
    if (R < 128) threads = 64;
    if (R < 64)  threads = 32;
    if (threads < 32) threads = 32;

    // Get device shmem limit (default 48KB on many GPUs)
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "getDeviceProperties");
    const size_t SHMAX = (size_t)prop.sharedMemPerBlock;

    // helper to compute shmem for a given threads
    auto shmem_for = [&](int th)->size_t {
      const uint32_t nwarps = (uint32_t)((th + 31) / 32);
      const size_t shmem_base =
          (size_t)th * (size_t)K * (sizeof(float) + sizeof(uint32_t));
      const size_t shmem_warpmerge =
          shmem_base + (size_t)nwarps * (size_t)K * (sizeof(float) + sizeof(uint32_t));
      return use_warpmerge ? shmem_warpmerge : shmem_base;
    };

    // clamp threads down until shmem fits
    size_t shmem_bytes = shmem_for(threads);
    while (threads > 32 && shmem_bytes > SHMAX) {
      threads >>= 1;                 // 256->128->64->32
      shmem_bytes = shmem_for(threads);
    }

    // final safety check
    if (shmem_bytes > SHMAX) {
      std::fprintf(stderr,
                  "[cuda] shmem too large even at threads=%d (K=%u R=%u use_warpmerge=%d): "
                  "need=%zuB > limit=%zuB\n",
                  threads, (unsigned)K, (unsigned)R, (int)use_warpmerge,
                  shmem_bytes, SHMAX);
      std::exit(3);
    }

    const dim3 grid(Q, 1, 1);
    const dim3 block(threads, 1, 1);



    if (base_dtype == 2) {
      if (!use_warpmerge) {
        l2_topk_fp16_kernel<NVDB_CUDA_KMAX><<<grid, block, shmem_bytes, g_stream>>>(
          (const __half*)g_base_dev, N, D,
          g_q_dev, Q,
          g_cand_dev, R,
          K,
          g_out_id_dev, g_out_dist_dev
        );
      } else {
        l2_topk_fp16_kernel_warpmerge<NVDB_CUDA_KMAX><<<grid, block, shmem_bytes, g_stream>>>(
          (const __half*)g_base_dev, N, D,
          g_q_dev, Q,
          g_cand_dev, R,
          K,
          g_out_id_dev, g_out_dist_dev
        );
      }
    } else {
      l2_topk_fp32_kernel<NVDB_CUDA_KMAX><<<grid, block, shmem_bytes, g_stream>>>(
        (const float*)g_base_dev, N, D,
        g_q_dev, Q,
        g_cand_dev, R,
        K,
        g_out_id_dev, g_out_dist_dev
      );
    }

  static bool once=true;
  if (once) {
    once=false;
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "getDeviceProperties");
    std::fprintf(stderr,
      "[cuda_launch_dbg] K=%u R=%u Q=%u threads=%d shmem=%zu bytes "
      "sharedMemPerBlock=%zu sharedMemOptin=%zu\n",
      K, R, Q, threads, shmem_bytes,
      (size_t)prop.sharedMemPerBlock,
      (size_t)prop.sharedMemPerBlockOptin
    );
  }

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaEventRecord(ev2, g_stream), "event record 2");

    // D2H async (ids always; dist optional)
    ck(cudaMemcpyAsync(h_oid_dst, g_out_id_dev, (size_t)Q * (size_t)K * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost, g_stream), "D2H ids async");
    if (want_dist) {
      ck(cudaMemcpyAsync(h_od_dst, g_out_dist_dev, (size_t)Q * (size_t)K * sizeof(float),
                        cudaMemcpyDeviceToHost, g_stream), "D2H dist async");
    }

    ck(cudaEventRecord(ev3, g_stream), "event record 3");
    ck(cudaEventSynchronize(ev3), "event sync");

    // timing
    float h2d_ms=0, kern_ms=0, d2h_ms=0;
    ck(cudaEventElapsedTime(&h2d_ms, ev0, ev1), "elapsed h2d");
    ck(cudaEventElapsedTime(&kern_ms, ev1, ev2), "elapsed kernel");
    ck(cudaEventElapsedTime(&d2h_ms, ev2, ev3), "elapsed d2h");

    if (timing) {
      timing->h2d_ms = h2d_ms;
      timing->kernel_ms = kern_ms;
      timing->d2h_ms = d2h_ms;
      timing->total_ms = h2d_ms + kern_ms + d2h_ms;
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    cudaEventDestroy(ev3);

    // copy back from pinned to std::vector outputs
    if (use_pinned) {
      std::memcpy(out_topk_ids.data(), g_h_out_id_pinned, (size_t)Q * (size_t)K * sizeof(uint32_t));
      if (want_dist) {
        out_topk_dist.resize((size_t)Q * (size_t)K);
        std::memcpy(out_topk_dist.data(), g_h_out_dist_pinned, (size_t)Q * (size_t)K * sizeof(float));
      }
    }

  }

} // namespace nvdb
