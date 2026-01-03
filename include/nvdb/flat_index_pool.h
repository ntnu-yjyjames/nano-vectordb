#pragma once
#include "nvdb/vector_dataset.h"
#include "nvdb/topK.h"
#include <cstdint>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace nvdb {

class FlatIndexPool {
public:
  FlatIndexPool(const VectorDataset* base, int threads);
  ~FlatIndexPool();

  FlatIndexPool(const FlatIndexPool&) = delete;
  FlatIndexPool& operator=(const FlatIndexPool&) = delete;

  std::vector<SearchResult> search_topk_dot(const float* q, uint32_t k);

private:
  const VectorDataset* base_;
  int threads_;

  std::vector<std::thread> workers_;
  std::vector<std::vector<SearchResult>> locals_;

  // per-query shared state
  const float* cur_q_ = nullptr;
  uint32_t cur_k_ = 0;

  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  bool stop_ = false;

  int epoch_ = 0;                // 每次 query +1，讓 worker 知道有新工作
  int done_count_ = 0;

  //static float dot_f32(const float* a, const float* b, uint32_t dim);
  std::vector<SearchResult> scan_range(const float* q, uint32_t k, uint64_t begin, uint64_t end) const;

  void worker_loop(int tid);
};

} // namespace nvdb
