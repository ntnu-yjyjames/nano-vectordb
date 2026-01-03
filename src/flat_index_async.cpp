#include "nvdb/flat_index_async.h"
#include "nvdb/score_dispatch.h"
#include <future>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace nvdb {

std::vector<SearchResult> FlatIndexAsync::scan_range(const float* q, uint32_t k,
                                                     uint64_t begin, uint64_t end,
                                                     uint32_t dt, uint32_t dim) const {
  TopKBuffer topk(k);

  for (uint64_t i = begin; i < end; ++i) {
    float s = nvdb::score_query_base_at(*base_, q, i, dim, dt);
    topk.consider(i, s);
  }
  return topk.finalize_sorted_desc();
}

std::vector<SearchResult> FlatIndexAsync::search_topk_dot(const float* q, uint32_t k, int threads) const {
  if (!base_ || base_->count() == 0) throw std::runtime_error("Empty base");
  if (k == 0) return {};
  if (threads < 1) threads = 1;

  //  validate once on main thread (avoid throwing inside async tasks)
  nvdb::ensure_supported_base_dtype(*base_);

  const uint32_t dt = base_->dtype();
  const uint32_t dim = base_->dim();
  const uint64_t n = base_->count();
  const uint64_t block = (n + uint64_t(threads) - 1) / uint64_t(threads);

  std::vector<std::future<std::vector<SearchResult>>> futs;
  futs.reserve(threads);

  for (int t = 0; t < threads; ++t) {
    uint64_t begin = uint64_t(t) * block;
    uint64_t end = std::min<uint64_t>(begin + block, n);
    if (begin >= end) break;

    auto fut = std::async(std::launch::async,
                          [this, q, k, begin, end, dt, dim]() -> std::vector<SearchResult> {
                            return this->scan_range(q, k, begin, end, dt, dim);
                          });
    futs.emplace_back(std::move(fut));
  }

  TopKBuffer global(k);
  for (auto& f : futs) {
    global.merge_from(f.get());
  }
  return global.finalize_sorted_desc();
}

} // namespace nvdb
