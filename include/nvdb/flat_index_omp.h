#pragma once
#include "nvdb/vector_dataset.h"
#include "nvdb/topK.h"
#include <cstdint>
#include <vector>

namespace nvdb {



class FlatIndexOMP {
public:
  explicit FlatIndexOMP(const VectorDataset* base) : base_(base) {}
  std::vector<SearchResult> search_topk_dot(const float* q, uint32_t k) const;

private:
  const VectorDataset* base_;
  //static float dot_f32(const float* a, const float* b, uint32_t dim);
};

} // namespace nvdb
