#pragma once
#include "nvdb/vector_dataset.h"
#include "nvdb/topK.h"
#include <cstdint>
#include <vector>

namespace nvdb {

class FlatIndexAsync {
public:
  explicit FlatIndexAsync(const VectorDataset* base) : base_(base) {}

  
  std::vector<SearchResult> search_topk_dot(const float* q, uint32_t k, int threads) const;

private:
  const VectorDataset* base_;

  //std::vector<SearchResult> scan_range(const float* q, uint32_t k, uint64_t begin, uint64_t end) const;
  std::vector<SearchResult> scan_range(const float* q, uint32_t k,uint64_t begin, uint64_t end,uint32_t dt, uint32_t dim) const;

};

} // namespace nvdb
