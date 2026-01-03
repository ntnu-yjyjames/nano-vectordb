#include "nvdb/vector_dataset.h"
#include "nvdb/flat_index.h"
#include "nvdb/flat_index_omp.h"
#include "nvdb/flat_index_async.h"
#include "nvdb/flat_index_pool.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <algorithm>
#include <thread>
#include <memory>

#if NVDB_HAS_OPENMP
#include <omp.h>
#endif

#include <cstdlib>
#include "nvdb/simd_dot.h"



static double ms_since(const std::chrono::steady_clock::time_point& t0,
                       const std::chrono::steady_clock::time_point& t1) {
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: nvdb_bench <base.vecbin> <query.vecbin> <k> [mode=st] [threads=0] [warmup=5]\n";
        return 1;
    }

    const std::string base_path  = argv[1];
    const std::string query_path = argv[2];
    const uint32_t k = static_cast<uint32_t>(std::stoul(argv[3]));

    const std::string mode = (argc >= 5) ? std::string(argv[4]) : "st";
    int threads = (argc >= 6) ? std::stoi(argv[5]) : 0;
    const int warmup = (argc >= 7) ? std::stoi(argv[6]) : 5;

    if (threads <= 0) threads = (int)std::thread::hardware_concurrency();
    int report_threads = threads;
    #if NVDB_HAS_OPENMP
    if (mode == "omp") report_threads = omp_get_max_threads();
    #endif
    std::cout << "mode=" << mode << " threads=" << report_threads << "\n";

    const char* fs = std::getenv("NVDB_FORCE_SCALAR");
    if (fs && fs[0] == '1') nvdb::set_force_scalar(true);


    nvdb::VectorDataset base, query;
    base.load(base_path);
    query.load(query_path);

    if (base.dim() != query.dim()) {
        std::cerr << "Dim mismatch: base.dim=" << base.dim()
                << ", query.dim=" << query.dim() << "\n";
        return 2;
    }

    nvdb::FlatIndex st_index(&base);
    nvdb::FlatIndexOMP omp_index(&base);
    nvdb::FlatIndexAsync async_index(&base);

    std::unique_ptr<nvdb::FlatIndexPool> pool_index;
    if (mode == "pool") {
        pool_index = std::make_unique<nvdb::FlatIndexPool>(&base, threads);
    }


    auto run_query = [&](const float* qvec) -> std::vector<nvdb::SearchResult> {
    if (mode == "st")    return st_index.search_topk_dot(qvec, k);
    if (mode == "omp")   return omp_index.search_topk_dot(qvec, k);
    if (mode == "async") return async_index.search_topk_dot(qvec, k, threads);
    if (mode == "pool")  return pool_index->search_topk_dot(qvec, k);
    throw std::runtime_error("Unknown mode: " + mode);
    };

    std::cout << "Base count=" << base.count() << " dim=" << base.dim()
                << " | Query count=" << query.count()
                << " | k=" << k << " | warmup=" << warmup << "\n";

    // warmup
    for (int i = 0; i < warmup; ++i) {
        const float* q0 = query.vector_ptr(0);
        //auto r = index.search_topk_dot(q0, k);
        auto r = run_query(q0);
        (void)r;
    }

    // benchmark all queries, record per-query latency
    std::vector<double> lat_ms;
    lat_ms.reserve(static_cast<size_t>(query.count()));

    volatile float sink = 0.0f; // prevent over-optimization
    auto t_all0 = std::chrono::steady_clock::now();

    for (uint64_t qi = 0; qi < query.count(); ++qi) {
        auto t0 = std::chrono::steady_clock::now();
        const float* q = query.vector_ptr(qi);
        //auto topk = index.search_topk_dot(q, k);
        auto topk = run_query(q);
        auto t1 = std::chrono::steady_clock::now();

        lat_ms.push_back(ms_since(t0, t1));
        if (!topk.empty()) sink += topk[0].score;
    }

    auto t_all1 = std::chrono::steady_clock::now();
    const double total_ms = ms_since(t_all0, t_all1);

    // stats
    std::sort(lat_ms.begin(), lat_ms.end());
    auto pct = [&](double p) -> double {
        if (lat_ms.empty()) return 0.0;
        const double idx = (p / 100.0) * (lat_ms.size() - 1);
        const size_t i = static_cast<size_t>(idx);
        return lat_ms[i];
    };

    const double avg = total_ms / double(query.count());
    const double qps = (double(query.count()) * 1000.0) / total_ms;
    // --- bandwidth metrics ---
    //const double bytes_per_query = double(base.count()) * double(base.dim()) * double(sizeof(float));
    const double bytes_per_query =double(base.count()) * double(base.dim()) * double(nvdb::bytes_per_elem(base.dtype()));

    const double effective_bandwidth_GBps = (avg > 0.0) ? (bytes_per_query * 1e-6 / avg) : 0.0;


    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total: " << total_ms << " ms\n";
    std::cout << "Avg:   " << avg << " ms/query  (" << qps << " QPS)\n";
    std::cout << "p50:   " << pct(50) << " ms\n";
    std::cout << "p95:   " << pct(95) << " ms\n";
    std::cout << "p99:   " << pct(99) << " ms\n";
    std::cout << "sink=" << sink << "\n";
    std::cout << std::setprecision(0);
    std::cout << "bytes_per_query=" << bytes_per_query << "\n";
    std::cout << std::setprecision(3);
    std::cout << "effective_bandwidth_GBps=" << effective_bandwidth_GBps << "\n";


    return 0;
}
