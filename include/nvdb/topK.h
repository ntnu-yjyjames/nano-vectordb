#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>

namespace nvdb {

struct SearchResult {
  uint64_t id;
  float score;
};

//  O(k) insertion is good for k around 10 to 20
class TopKBuffer {
  public:
    explicit TopKBuffer(uint32_t k) : k_(k) {
      buf_.reserve(k_);
      worst_score_ = std::numeric_limits<float>::infinity(); // min-heap concept
      worst_idx_ = 0;
    }

    void consider(uint64_t id, float score) {
      if (k_ == 0) return;

      if (buf_.size() < k_) {
        buf_.push_back({id, score});
        if (buf_.size() == k_) recompute_worst();
        return;
      }

      // Compare with worst
      if (score <= worst_score_) return;

      // replace worst
      buf_[worst_idx_] = {id, score};
      recompute_worst();
    }

    // merge  topks（usually size <=k）
    void merge_from(const std::vector<SearchResult>& other) {
      for (const auto& x : other) consider(x.id, x.score);
    }

    std::vector<SearchResult> finalize_sorted_desc() {
      std::sort(buf_.begin(), buf_.end(),
                [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
      return buf_;
    }

    const std::vector<SearchResult>& raw() const { return buf_; }

  private:
    void recompute_worst() {
      worst_score_ = buf_[0].score;
      worst_idx_ = 0;
      for (size_t i = 1; i < buf_.size(); ++i) {
        if (buf_[i].score < worst_score_) {
          worst_score_ = buf_[i].score;
          worst_idx_ = i;
        }
      }
    }

    uint32_t k_;
    std::vector<SearchResult> buf_;
    float worst_score_;
    size_t worst_idx_;
  };

} // namespace nvdb
