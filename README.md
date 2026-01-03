# Nano-VectorDB
## Highlights (TL;DR)
Nano-VectorDB is a research-grade C++ vector search engine built to study **memory-bandwidth–bound behavior** in dense retrieval systems.

Key results:
- **5.4× speedup** over single-thread flat scan via pinned thread pool
- **~44.4 GB/s sustained memory bandwidth**, reaching hardware limits
- **2.7× single-thread speedup** from AVX2+FMA SIMD
- FP16 bases reduce bytes/query and reach bandwidth saturation at lower thread counts

See `performance.md` for full experimental analysis.

## Background
Nano-VectorDB is a lightweight, embedded **C++ flat-scan vector search engine**
built to study **system-level performance bottlenecks** in dense retrieval
workloads such as RAG pipelines.

Rather than optimizing model accuracy, this project focuses on **hardware-visible
constraints**, including:

- memory bandwidth and cache behavior  
- data movement (disk → RAM → cache)  
- CPU parallelism and instruction-level utilization (SIMD)  
- performance ceilings across dataset sizes  

**Key idea:** For large-scale flat scan, performance rapidly becomes
**data-movement bound**.

Once memory bandwidth is saturated, increasing thread count yields diminishing
returns. Meaningful speedups come from **reducing bytes per query**
(e.g., FP16/INT8) or avoiding full scans—not from additional parallelism.



---

## Project Overview (Original Goal)

Nano-VectorDB was designed as a **clean experimental platform** for understanding
the fundamental performance limits of dense vector retrieval at scale.

Rather than building a full-featured vector database, the system intentionally
focuses on a minimal set of retrieval primitives to make system bottlenecks
**explicit, measurable, and reproducible**.

The design goals are:

1. **Infrastructure**: zero-copy loading of large embedding matrices via `mmap`
   to minimize I/O and allocation overhead.
2. **Baseline retrieval**: correctness-first flat-scan Top-k search as a
   ground-truth reference.
3. **Bottleneck isolation**: controlled experiments to separately study threading,
   SIMD acceleration, and memory-bandwidth limits.
4. **Hardware-aware optimization**: explicit control over memory layout, CPU
   affinity/pinning, and instruction-level parallelism.
5. **Data reduction**: reduce bytes per query (FP16 implemented; INT8/PQ planned)
   to push throughput beyond bandwidth ceilings.


---

## Current Status (Checklist)

### ✅ Phase 1 — Infrastructure & Zero-Copy I/O
- [x] CMake/Ninja project structure
- [x] `mmap`-based dataset loader
- [x] Flat binary formats supported:
- [x] legacy **raw12** (float32)
 - [x] **vecbin64** header (dtype-aware: FP32/FP16)
- [x] Correctness checks (`nvdb_dump`, `nvdb_sanity`)

### ✅ Phase 2 — Baseline Retrieval
- [x] Flat-scan Top-k baseline (ST)
- [x] Benchmark harness with Avg/QPS/p95/p99
- [x] Derived metrics: `bytes_per_query`, `effective_bandwidth_GBps`

### ✅ Phase 3 — Parallelism & SIMD
- [x] OpenMP implementation (OMP)
- [x] `std::async` implementation (ASYNC)
- [x] Pinned thread-pool implementation (POOL) with hybrid-aware affinity handling
- [x] Thread scaling experiments (sweet spot discovery)
- [x] AVX2+FMA dot kernel (compute-bound → bandwidth-bound transition)

### ✅ Phase 4 — Reduce Data Movement (FP16)
- [x] FP16 base conversion tool (`nvdb_convert_f16`)
- [x] FP16 flat scan support (ST/OMP/ASYNC/POOL)
- [x] Cross-size FP16 results (500K / 1M / 2.9M)
- [x] Hybrid (Alder Lake i7-12700) analysis and affinity pitfalls documented

### ⏳ Next (Planned)
- [ ] Phase 4B: INT8 base + scaling (per-vector/per-block scale)
- [ ] Phase 5: Query batching / cache tiling / prefetch experiments
- [ ] Phase 6: ANN baseline (HNSW/IVF) with recall@k vs latency trade-offs

---
> Note: The following section is provided for **reproducibility and experimental rigor**.
> For a high-level summary of results, see `performance.md`.

## Dataset & Embeddings

### Dataset source (official)
All experiments use a dataset derived from the **arXiv metadata snapshot released by Cornell University** (`arxiv-metadata-oai-snapshot.json`), distributed on Kaggle:

- Kaggle dataset (Cornell University — arXiv): https://www.kaggle.com/datasets/Cornell-University/arxiv

This repository does **not** include the raw dataset or generated embeddings due to size constraints.

---

### Data preparation pipeline (chunking + embedding → `.vecbin`)
Nano-VectorDB does **not** perform chunking or embedding natively.  
All preprocessing, **chunking**, and embedding are performed in a standalone Python pipeline and exported to `.vecbin` for the C++ engine.

This repo includes:
- `scripts/build_vecbin_chunked.py` — builds **chunk-level** embeddings from the CSV and exports `.vecbin` + row metadata.

**Input (repo-local):**
- `./arxiv_data/arxiv_cornell_title_abstract.csv`  
  Required columns: `title`, `abstract`  
  Recommended: `id`, `categories`  
  Optional full-text columns: `full_text`, `body_text`, `paper_text`, or `text` (if present, sections are detected and chunked)

**Embedding model:**
- `sentence-transformers/all-MiniLM-L6-v2` → **384-D**, L2-normalized

---

### Important note on “500K” (documents vs vectors)
The Python pipeline parameter `--max-docs 500000` limits the number of **documents (CSV rows)** processed, **not** the number of output vectors.  
Because each document may produce **multiple chunks**, the resulting embedding matrix typically contains **more than 500K vectors**.

In `performance.md`, reported datasets such as “500K / 1M vectors” refer to the **number of vectors (rows in the embedding matrix)**.  
To make experiments comparable, we generate a larger (ideally full) embedding matrix and then slice the first **N vectors** using the C++ tool:

- `./build/nvdb_slice <input.vecbin> <output.vecbin> <N>`

This ensures the benchmark input size is **exactly N vectors**, regardless of how many chunks were produced per document.

---
### Python environment (`.venv`)
We use a project-local virtual environment for the data pipeline:

```bash
cd Nano-vectorDB
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```
Deactivate with:
```bash
deactivate
```
### Recommended reproduction workflow (exact vector counts)
If you want to reproduce the benchmark results with consistent vector counts:

1) **Build a full embedding set** (or a large enough set):
```bash
python scripts/build_vecbin_chunked.py \
  --csv-path ./arxiv_data/arxiv_cornell_title_abstract.csv \
  --out ./vecbin_full/embeddings_full.vecbin \
  --format raw12 \
  --prefer-fulltext \
  --export-metadata
```
2) Slice to the first N vectors (e.g., 500K / 1M):
```bash
./build/nvdb_slice ./vecbin_full/embeddings_full.vecbin ./vecbin_full/embeddings_500k.vecbin 500000
./build/nvdb_slice ./vecbin_full/embeddings_full.vecbin ./vecbin_full/embeddings_1m.vecbin   1000000
```

3) (Optional) Convert FP32 → FP16 base:
```bash
./build/nvdb_convert_f16 ./vecbin_full/embeddings_500k.vecbin ./vecbin_full/embeddings_500k_f16.vecbin
```
#### Faster workflow (approximate experiments)
If you do not require strict comparability by exact vector count, you may run experiments directly on the output of:
```bash
python scripts/build_vecbin_chunked.py --max-docs <N_docs> ...
```

Be aware that the resulting .vecbin will contain N_vectors ≥ N_docs due to chunking, so results may not align exactly with the “N vectors” benchmarks.

#### Row-level metadata (optional)

When `--export-metadata` is enabled, the pipeline writes:
* `<out>.rowmeta.jsonl`

Each line maps a vector row back to its source:
```json
{"row":0,"doc_idx":0,"id":"...","title":"...","section":"abstract","chunk_index":0}
```

This is optional for pure performance benchmarking, but useful for end-to-end inspection (mapping row id → paper/chunk).


---

## Build

```bash
# Arch Linux
sudo pacman -S --needed base-devel cmake ninja

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
## Quickstart
1) Inspect a vecbin file
```bash
./build/nvdb_dump ./vecbin_full/embeddings.vecbin 3 8
./build/nvdb_sanity ./vecbin_full/embeddings.vecbin 10
```
2) Make a smaller base (e.g., 100k) and queries
```bash
./build/nvdb_slice  ./vecbin_full/embeddings.vecbin ./out/embeddings_100k.vecbin 100000
./build/nvdb_make_query ./out/embeddings_100k.vecbin ./out/query_100.vecbin 100 42 random
```
3) Run baseline benchmark
```bash
./build/nvdb_bench ./out/embeddings_100k.vecbin ./out/query_100.vecbin 10 st 1 5
```
4) Run OMP benchmark (thread count via env)
```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=8
./build/nvdb_bench ./out/embeddings_100k.vecbin ./out/query_100.vecbin 10 omp 0 5
```
5) Convert FP32 base → FP16 base and benchmark
```bash
./build/nvdb_convert_f16 ./out/embeddings_100k.vecbin ./out/embeddings_100k_f16.vecbin
./build/nvdb_bench ./out/embeddings_100k_f16.vecbin ./out/query_100.vecbin 10 pool 8 5
```
## Results & Performance Report

All experimental results, figures, and methodology are documented in:
➡️ performance.md

* parallel flat-scan benchmarks (500K / 1M / 2.9M)
* thread scaling and bandwidth ceiling analysis
* AVX2 compute-bound → bandwidth-bound transition
* FP16 data-movement optimization + hybrid CPU effects (Alder Lake)

### Hardware / Environment (Reference)
* CPU: Intel Core i7-12700 (8P + 4E / 20 threads), Alder Lake hybrid
* Memory: 32GB DDR4-3200 dual-channel
* OS: Arch Linux (Kernel 6.12.63-lts)
* Compiler: GCC 15.2.1 with -O3 -mavx2 -mfma -pthread

