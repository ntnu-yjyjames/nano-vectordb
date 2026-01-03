#include "nvdb/flat_index.h"
#include <queue>
#include <algorithm>
#include <stdexcept>
#include "nvdb/flat_index.h"
#include "nvdb/simd_dot.h"   
#include "nvdb/score_dispatch.h"

namespace nvdb {

//  dot calculation
float FlatIndex::dot_f32(const float* a, const float* b, uint32_t dim) {
  return nvdb::dot_f32(a, b, dim);
}

std::vector<SearchResult> FlatIndex::search_topk_dot(const float* q, uint32_t k) const {
  if (!base_ || base_->count() == 0) throw std::runtime_error("Empty base");
  if (k == 0) return {};
  nvdb::ensure_supported_base_dtype(*base_);
  const uint32_t dt = base_->dtype();
  const uint32_t dim = base_->dim();
  
  const uint64_t n = base_->count();
  if (k > n) k = static_cast<uint32_t>(n);

  //Using Min-Heap to find top_k
  struct Node { float score; uint64_t id; };
  struct Cmp { bool operator()(const Node& x, const Node& y) const { return x.score > y.score; } };
  std::priority_queue<Node, std::vector<Node>, Cmp> heap;

  for (uint64_t i = 0; i < n; ++i) {
    //const float* v = base_->vector_ptr(i);
    //float s = dot_f32(q, v, dim); 
    float s = nvdb::score_query_base_at(*base_, q, i, dim, dt);

    if (heap.size() < k) heap.push({s, i});
    else if (s > heap.top().score) { heap.pop(); heap.push({s, i}); }
  }

  std::vector<SearchResult> out;
  out.reserve(k); // reserve k slots for top_k
  while (!heap.empty()) {
    out.push_back({heap.top().id, heap.top().score});
    heap.pop();
  }
  std::reverse(out.begin(), out.end());
  return out;
}

} // namespace nvdb
