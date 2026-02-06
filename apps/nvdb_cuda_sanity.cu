#include <cstdio>
#include <cuda_runtime.h>

__global__ void add1(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += 1.0f;
}

static void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[CUDA ERROR] %s: %s\n", msg, cudaGetErrorString(e));
    std::exit(1);
  }
}

int main() {
  int dev = 0;
  cudaDeviceProp p{};
  ck(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");

  std::printf("GPU: %s | cc=%d.%d | globalMem=%.1f GiB\n",
              p.name, p.major, p.minor,
              double(p.totalGlobalMem) / (1024.0*1024.0*1024.0));

  const int n = 1 << 20;
  float* d = nullptr;
  ck(cudaMalloc(&d, n * sizeof(float)), "cudaMalloc");
  ck(cudaMemset(d, 0, n * sizeof(float)), "cudaMemset");

  add1<<<(n + 255) / 256, 256>>>(d, n);
  ck(cudaGetLastError(), "kernel launch");
  ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  float h = -1.f;
  ck(cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
  std::printf("sanity: first element = %.1f (expect 1.0)\n", h);

  cudaFree(d);
  return 0;
}
