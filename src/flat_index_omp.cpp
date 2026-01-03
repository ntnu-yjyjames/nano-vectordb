#include "nvdb/flat_index_omp.h"
#include <queue>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "nvdb/simd_dot.h"  
#include "nvdb/score_dispatch.h"
#if NVDB_HAS_OPENMP
#include <omp.h>
#endif


namespace nvdb {


    std::vector<SearchResult> FlatIndexOMP::search_topk_dot(const float* q, uint32_t k) const {
        if (!base_ || base_->count() == 0) throw std::runtime_error("Empty base");
        if (k == 0) return {};
        nvdb::ensure_supported_base_dtype(*base_);
        const uint32_t dt = base_->dtype();
        const uint32_t dim = base_->dim();

        const uint64_t n = base_->count();
        if (k > n) k = static_cast<uint32_t>(n);

        struct Node { float score; uint64_t id; };
        struct Cmp { bool operator()(const Node& x, const Node& y) const { return x.score > y.score; } };

        #if NVDB_HAS_OPENMP
        const int nthreads = omp_get_max_threads();
        #else
        const int nthreads = 1;
        #endif

        // 每個 thread 一個 heap
        std::vector<std::priority_queue<Node, std::vector<Node>, Cmp>> heaps(nthreads);

        #if NVDB_HAS_OPENMP
        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& heap = heaps[tid];

        #pragma omp for schedule(static)
            for (uint64_t i = 0; i < n; ++i) {
                //const float* v = base_->vector_ptr(i);
                //float s = nvdb::dot_f32(q, v, dim);
                float s = nvdb::score_query_base_at(*base_, q, i, dim, dt);
                if (heap.size() < k) heap.push({s, i});
                else if (s > heap.top().score) { heap.pop(); heap.push({s, i}); }
            }
            
        }
        #else
        // fallback single-thread
        auto& heap = heaps[0];
        for (uint64_t i = 0; i < n; ++i) {
            //const float* v = base_->vector_ptr(i);
            //float s = dot_f32(q, v, dim);
            float s = nvdb::dot_f32(q, v, dim);
            if (heap.size() < k) heap.push({s, i});
            else if (s > heap.top().score) { heap.pop(); heap.push({s, i}); }
        }
        #endif

        // 合併各 thread 的 top-k → 全域 top-k
        std::priority_queue<Node, std::vector<Node>, Cmp> global;
        for (int t = 0; t < nthreads; ++t) {
            auto& h = heaps[t];
            while (!h.empty()) {
            Node x = h.top(); h.pop();
            if (global.size() < k) global.push(x);
            else if (x.score > global.top().score) { global.pop(); global.push(x); }
            }
        }

        std::vector<SearchResult> out;
        out.reserve(k);
        while (!global.empty()) {
            out.push_back({global.top().id, global.top().score});
            global.pop();
        }
        std::reverse(out.begin(), out.end());
        return out;
    }

} // namespace nvdb
