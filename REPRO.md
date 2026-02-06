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

If your CMake uses find_library(FAISS_LIB faiss ...), make sure it searches:
* `${FAISS_ROOT}/lib `(or `${FAISS_ROOT}/lib64`)
* and includes `${FAISS_ROOT}/include`

> Tip: If your system has multiple users (local vs CRD), install FAISS into a shared readable prefix like `/srv/shared/faiss`.

## 2) Installing FAISS (recommended: from source)
### 2.1 Build FAISS (CPU)

Example (generic; adjust to your environment):
```bash
git clone --recursive https://github.com/facebookresearch/faiss.git
cd faiss

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_CXX_STANDARD=17

cmake --build build
cmake --install build --prefix /srv/shared/faiss
```

Validate:
```bash
ls -la /srv/shared/faiss/include/faiss | head
ls -la /srv/shared/faiss/lib | grep -i faiss
```
### 2.2 (Optional) Build FAISS with CUDA

Only needed if you later compare GPU FAISS or want CUDA baselines:
```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_CXX_STANDARD=17

cmake --build build
cmake --install build --prefix /srv/shared/faiss
```

> If you see errors like “Thrust requires at least C++17”, confirm` -DCMAKE_CXX_STANDARD=17`and that NVCC uses a compatible host compiler.

## 3) Dataset preparation (`.vecbin`)

Nano-VectorDB benchmarks operate on `.vecbin` files:
* Base embeddings: FP32 / FP16 / INT8(+scale)
* Query embeddings: FP32

### 3.1 Create embeddings via Python pipeline (example)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_vecbin_chunked.py \
  --csv-path ./arxiv_data/arxiv_cornell_title_abstract.csv \
  --out ./vecbin_full/embeddings_full.vecbin \
  --export-metadata
```
### 3.2 Slice exact vector counts (recommended)
```bash
./build/nvdb_slice ./vecbin_full/embeddings_full.vecbin ./vecbin_full/embeddings_500k.vecbin 500000
./build/nvdb_slice ./vecbin_full/embeddings_full.vecbin ./vecbin_full/embeddings_1m.vecbin   1000000
```
### 3.3 Convert base dtype (FP16 / INT8)
```bash
./build/nvdb_convert_f16 ./vecbin_full/embeddings_500k.vecbin ./vecbin_full/embeddings_500k_f16.vecbin
./build/nvdb_quantize_i8 ./vecbin_full/embeddings_500k.vecbin ./vecbin_full/embeddings_500k_i8.vecbin
```
### 3.4 Build queries
```bash
./build/nvdb_make_query ./vecbin_full/embeddings_500k.vecbin ./vecbin_full/query_1000.vecbin 1000 42 random
```
## 4) Reproducing key experiments
### 4.1 Flat scan benchmark (exact)
```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=8

./build/nvdb_bench ./vecbin_full/embeddings_500k_f16.vecbin ./vecbin_full/query_1000.vecbin 10 omp 0 5
```
### 4.2 Batching / tiling (example)
```bash
# args: ... warmup batch_q tile_vecs prefetch_dist
./build/nvdb_bench ./vecbin_full/embeddings_full_i8.vecbin ./vecbin_full/query_1000.vecbin 10 omp 0 5 8 512 0
```
### 4.3 HNSW baseline (ANN)
```bash
HNSW_M=16 HNSW_EF_CONSTRUCT=200 \
./build/nvdb_hnsw_build ./vecbin_full/embeddings_500k_f16.vecbin ./hnsw_index/hnsw_500k.index

export HNSW_EF_SEARCH=64
./build/nvdb_hnsw_eval ./vecbin_full/embeddings_500k_f16.vecbin ./hnsw_index/hnsw_500k.index ./vecbin_full/query_1000.vecbin 10
```
### 4.4 IVF-Flat baseline (FAISS required)
```bash
export IVF_NLIST=4096
export IVF_TRAIN=200000

./build/nvdb_ivf_build ./vecbin_full/embeddings_500k_f16.vecbin ./ivf_index/ivf_500k_f16_n4096_t200k.faiss

for np in 1 4 8 16 32 64 128 256; do
  export IVF_NPROBE=$np
  ./build/nvdb_ivf_eval ./vecbin_full/embeddings_500k_f16.vecbin ./ivf_index/ivf_500k_f16_n4096_t200k.faiss ./vecbin_full/query_1000.vecbin 10
done
```
### 4.5 OPQ-PQ + cached GT + refinement

1. Build GT (one-time)
```bash
mkdir -p gt
./build/nvdb_gt_build ./vecbin_full/embeddings_500k_f16.vecbin ./vecbin_full/query_1000.vecbin 10 gt/gt_500k_q1000_k10_l2.gtbin
```
2. Run eval with refinement
```bash
export EVAL_MODE=full
export EXACT_METRIC=L2
export GT_PATH=gt/gt_500k_q1000_k10_l2.gtbin
export IVF_NLIST=4096
export IVF_NPROBE=64

# sweep refine depth
for rk in 10 20 50 100; do
  export REFINE_K=$rk
  ./build/nvdb_ivf_eval ./vecbin_full/embeddings_500k_f16.vecbin ./ivf_index/ivfopqpq_500k_f16_n4096_t200k_opq_m64_b8.faiss ./vecbin_full/query_1000.vecbin 10
done
```
## 5) FAISS attribution 
### 5.1 Academic citation (recommended)

If you mention FAISS in a report/paper, cite the canonical FAISS reference:
> Johnson, Douze, Jégou. “Billion-scale similarity search with GPUs.” (FAISS reference)

Example BibTeX:
```bibtex
@article{johnson2017billion,
  title={Billion-scale similarity search with GPUs},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  year={2017}
}
```
5.2 Repository license/notice

* Do not copy FAISS source into this repo unless you plan to track its license in t`hird_party/`.
* Preferred: treat FAISS as an **external dependency** (installed under a prefix).
* Add a short **NOTICE** section in README or a separate NOTICE file:

Example:
```text
This project optionally links against FAISS (facebookresearch/faiss) for IVF/IVFPQ baselines.
FAISS is distributed under its own license. See the FAISS repository for details.
```
### 5.3 If you vendor FAISS (not recommended unless necessary)

If you use `git submodule` or vendor code under `third_party/faiss/,` you should:
* keep it as a submodule pinned to a commit
* include the FAISS license text in `third_party/faiss/` (as upstream provides)
* mention it in `NOTICE` / `THIRD_PARTY_NOTICES.md`