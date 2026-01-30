# REPRO.md — Reproducibility Guide (Nano-VectorDB)

This document describes how to reproduce key results in `performance.md` on Linux (Arch-like environment).
For a short overview, see `performance_summary.md`.

---

## 0) Requirements

### Toolchain
- CMake ≥ 3.20
- Ninja
- GCC (or Clang) with C++17 support
- OpenMP runtime (e.g., `libgomp` via GCC)
- Python 3.10+ (only for data pipeline / optional plotting scripts)

### Optional dependencies
- **FAISS (C++ library)** is required for IVF / IVFPQ / OPQ-PQ experiments (Phase 6B/6C).
  - If you only reproduce flat-scan, FP16/INT8, batching, and HNSW, you can build Nano-VectorDB without FAISS.

---

## 1) Repository build

### 1.1 Build (core; without FAISS)
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
1.2 Build (with FAISS enabled)

You need a FAISS install prefix containing:

headers: <prefix>/include/faiss/...

library: <prefix>/lib/libfaiss.a or <prefix>/lib/libfaiss.so

Then configure Nano-VectorDB with:
```bash
# Option A: use environment variable
export FAISS_ROOT=/path/to/faiss/prefix

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```