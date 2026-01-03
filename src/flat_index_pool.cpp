#include "nvdb/flat_index_pool.h"
#include "nvdb/simd_dot.h"  
#include "nvdb/score_dispatch.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <atomic>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <fstream>
#include <unordered_map>
#endif

// --- affinity helpers (Linux) ---
#ifdef __linux__
#include <fstream>
#include <unordered_map>

static int read_int_file(const std::string& path) {
  std::ifstream ifs(path);
  int v = -1;
  ifs >> v;
  return v;
}

static void pin_this_thread_to_cpu(int cpu_id) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu_id, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

static std::vector<int> get_one_cpu_per_core_allowed() {
  cpu_set_t aff;
  CPU_ZERO(&aff);
  if (sched_getaffinity(0, sizeof(aff), &aff) != 0) {
    return {0};
  }

  // key = (package_id, core_id) -> smallest cpu id
  std::unordered_map<long long, int> core2cpu;

  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &aff)) continue;

    const std::string topo =
      "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    const int pkg  = read_int_file(topo + "physical_package_id");
    const int core = read_int_file(topo + "core_id");
    if (pkg < 0 || core < 0) continue;

    const long long key =
      (static_cast<long long>(pkg) << 32) | static_cast<unsigned>(core);

    auto it = core2cpu.find(key);
    if (it == core2cpu.end() || cpu < it->second) core2cpu[key] = cpu;
  }

  std::vector<int> cpus;
  cpus.reserve(core2cpu.size());
  for (auto& kv : core2cpu) cpus.push_back(kv.second);
  std::sort(cpus.begin(), cpus.end());
  if (cpus.empty()) cpus.push_back(0);
  return cpus;
}
#endif








namespace nvdb {


    std::vector<SearchResult> FlatIndexPool::scan_range(const float* q, uint32_t k,uint64_t begin, uint64_t end) const {
    TopKBuffer topk(k);

    const uint32_t dt  = base_->dtype();
    const uint32_t dim = base_->dim();

    for (uint64_t i = begin; i < end; ++i) {
      float s = nvdb::score_query_base_at(*base_, q, i, dim, dt);
      topk.consider(i, s);
    }
    return topk.finalize_sorted_desc();
  }

    FlatIndexPool::FlatIndexPool(const VectorDataset* base, int threads)
        : base_(base), threads_(threads) {
        if (!base_ || base_->count() == 0) throw std::runtime_error("Empty base");
        if (threads_ < 1) threads_ = 1;

        locals_.resize(threads_);
        workers_.reserve(threads_);

        for (int t = 0; t < threads_; ++t) {
            workers_.emplace_back([this, t]() { this->worker_loop(t); });
        }
    }

    FlatIndexPool::~FlatIndexPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            epoch_++;
        }
        cv_start_.notify_all();
        for (auto& th : workers_) th.join();
    }

    void FlatIndexPool::worker_loop(int tid) {


        int seen_epoch = 0;
#ifdef __linux__
        static std::vector<int> core_list = get_one_cpu_per_core_allowed();
        const int cpu_id = core_list[tid % (int)core_list.size()];
        pin_this_thread_to_cpu(cpu_id);

#      ifdef NVDB_DEBUG_POOL
        // using mutex to protect print from multiple threads
        static std::mutex print_mu;
        {
            std::lock_guard<std::mutex> g(print_mu);
            std::cerr << "pool worker " << tid
                              << " pinned to cpu " << cpu_id
                              << " | core_list_size=" << core_list.size()
                              << "\n";

        }
#        endif

#endif

        while (true) {
                const float* q = nullptr;
                uint32_t k = 0;

                {
                std::unique_lock<std::mutex> lk(mu_);
                cv_start_.wait(lk, [&]() { return stop_ || epoch_ != seen_epoch; });
                if (stop_) return;

                seen_epoch = epoch_;
                q = cur_q_;
                k = cur_k_;
                }

                const uint64_t n = base_->count();
                const uint64_t block = (n + uint64_t(threads_) - 1) / uint64_t(threads_);
                uint64_t begin = uint64_t(tid) * block;
                uint64_t end = std::min<uint64_t>(begin + block, n);
                
                //if (begin < end) locals_[tid] = scan_range(q, k, begin, end);
                //else locals_[tid].clear();
                try {
                  if (begin < end) locals_[tid] = scan_range(q, k, begin, end);
                  else locals_[tid].clear();
                } catch (const std::exception& e) {
                  // Print the real cause instead of recursive terminate
                  std::cerr << "[POOL] worker " << tid
                            << " exception: " << e.what()
                            << " | base_dtype=" << base_->dtype()
                            << " | dim=" << base_->dim()
                            << " | begin=" << begin << " end=" << end
                            << "\n";
                  std::terminate();
                }
#ifdef NVDB_DEBUG_POOL
                // (debug) first few prints across all workers
                static std::atomic<int> prints{0};
                int p = prints.fetch_add(1);
                if (p < 8) {
                std::cerr << "pool worker " << tid
                            << " range computed, local_size=" << locals_[tid].size()
                            << "\n";
                }
#endif

                {
                std::lock_guard<std::mutex> lk(mu_);
                done_count_++;
                if (done_count_ == threads_) cv_done_.notify_one();
                }
         }
}

std::vector<SearchResult> FlatIndexPool::search_topk_dot(const float* q, uint32_t k) {
    if (k == 0) return {};
    if (!q) throw std::runtime_error("Null query");
    nvdb::ensure_supported_base_dtype(*base_);
    {
        std::lock_guard<std::mutex> lk(mu_);
        cur_q_ = q;
        cur_k_ = k;
        done_count_ = 0;
        epoch_++;
    }
    cv_start_.notify_all();

    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&]() { return done_count_ == threads_; });
    }

    TopKBuffer global(k);
    for (int t = 0; t < threads_; ++t) global.merge_from(locals_[t]);
    return global.finalize_sorted_desc();
    }

} // namespace nvdb
