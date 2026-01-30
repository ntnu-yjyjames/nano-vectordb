# Nano-VectorDB — Performance Summary

This page is a **high-level, low-friction summary** of Nano-VectorDB’s performance findings.
For full methodology, figures, and complete sweeps, see **`performance.md`**.

---

## TL;DR (What matters)

**Nano-VectorDB shows that flat-scan dense retrieval on modern CPUs quickly becomes memory-bandwidth bound.**
Once bandwidth is saturated, performance gains come mainly from:
1) **reducing bytes/query** (FP16 → INT8, PQ/OPQ), or  
2) **switching to ANN** (HNSW / IVF-family) with explicit accuracy tradeoffs.

---

## Key results (headline numbers)

### 1) Flat-scan scaling: parallelism saturates at the memory ceiling
- Parallel flat-scan (OpenMP / pinned pool) reaches a stable speedup envelope of roughly **~5×** over single-thread, across **500K → 2.9M** vectors.
- Effective bandwidth saturates around **~44–45 GB/s**, indicating a **bandwidth-limited regime**.

**Implication:** past moderate thread counts, adding threads mainly increases contention/tail variance; it does not buy proportional throughput.

### 2) SIMD helps *before* bandwidth saturation
- AVX2+FMA provides a large single-thread latency reduction (compute-bound regime),
- but provides **negligible improvement** when the scan is already bandwidth-bound (parallel regime).

**Implication:** SIMD is essential to reach saturation with fewer threads; after that, bytes/query dominates.

### 3) Data movement reductions: FP16 and INT8
- FP16 halves bytes/query vs FP32 and pushes the scan to hit bandwidth saturation at lower thread counts.
- INT8(+scale) reduces bytes/query further and yields consistent **~1.8–1.9× throughput gains vs FP16** under the same parallel configuration.

**Implication:** quantization behaves like a “bandwidth multiplier” once the kernel is streaming-bound.

### 4) ANN baselines: HNSW and IVF
- **HNSW** demonstrates the classic recall–latency–memory tradeoff; **efSearch** is the primary runtime knob, while **M** dominates footprint.
- **IVF-Flat** provides a clean “surface” view (nlist × nprobe): recall rises with nprobe, while tail latency (p99) grows rapidly with probing.
- **Compressed IVF (IVF-PQ / OPQ-PQ) + small exact refinement** is an effective “quality ladder”:
  - PQ/OPQ shrink index footprint dramatically,
  - refinement recovers near-exact recall with modest incremental tail cost when the candidate set is good.

---

## Practical recommendations 

### CPU flat-scan (exact)
- Use the pinned pool or OpenMP until you hit bandwidth saturation; beyond that, focus on **bytes/query** (FP16/INT8) rather than more threads.

### Compressed IVF (quality under small footprint)
- For this workload, a strong pattern is:
  - **OPQ-PQ candidate generation**
    - **small L2 refine** (re-rank top-R candidates)
- Refinement shows diminishing returns; small R often captures most of the gain.

---

## How to read the full report (`performance.md`)
If you only read a few sections:
1) **Flat-scan bandwidth saturation** (why ~5× is “the ceiling” on this platform)  
2) **FP16 / INT8** (bytes/query as the dominant lever)  
3) **IVF / HNSW baselines** (accuracy–latency–memory surfaces)  
4) **OPQ-PQ + refine** (compressed index + small exact rerank as a robust recipe)

---

## What's next (documentation structure)

To keep the repo readable:
- `performance.md` remains the **full experimental notebook** (all figures + tables).
- `performance_summary.md` (this file) is the **short executive summary**.
- Repro steps will live in a separate **`REPRO.md`** (exact commands, env, dataset prep, and “how to reproduce figures/tables”).

