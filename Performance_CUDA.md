# CUDA Phase — Step 1: CUDA Refinement (GPU L2 re-ranking)

This phase is the first **CUDA integration milestone**: we offload only the **refinement step** (exact L2 re-ranking over top-`REFINE_K` ANN candidates) to GPU, while keeping **candidate generation unchanged** (FAISS OPQ-PQ on CPU).
The goal is to establish a **correct, measurable CUDA path** with clean timing boundaries before CUDA-izing additional pipeline components (e.g., GPU-resident query/candidate buffers, overlapping transfers, or eventually moving more of scoring/top-k to GPU).

**Fixed setup (unless noted)**: OPQ-PQ (m=64, b=8), `nlist=4096`, `nprobe=64`, base=500K FP16, metric=L2 with cached GT.  
**Pipeline accounting**: `PIPELINE=staged` (ANN candidate generation and refinement measured as separate stages; **TOTAL = ANN + refine**).  
**Pinned memory policy**: `CUDA_PINNED=0` is the **baseline**. `CUDA_PINNED=1` is treated as an **optional optimization** (reported separately only when needed).

Baseline knobs used for the figures below (unless explicitly stated):
- `CUDA_KERNEL_MODE=baseline`
- `CUDA_RETURN_DIST=0` (ids-only D2H)
- `k=10` (unless explicitly varied)

---

## CU1 — End-to-end impact (TOTAL Avg) when only refinement is CUDA-accelerated

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU1_Total_Avg_Latency_dark.png">
  <img alt="Total average latency vs REFINE_K (CPU vs CUDA, Q=1000/Q=10000)" src="performance_images/CU1_Total_Avg_Latency_light.png">
</picture>

**Figure CU1.** Total average latency (ANN + refine) vs `REFINE_K`, comparing **CPU refine** vs **CUDA refine** (ANN remains CPU/FAISS).  
> Note: `Q=1000` is a **short-run** measurement and is more sensitive to run-to-run variance; `Q=10000` better reflects **steady-state** throughput.

**Reading the curve.** With ANN fixed on CPU, CUDA refinement mainly matters when `REFINE_K` becomes large (e.g., 200–500), where CPU refine begins to dominate TOTAL Avg, while CUDA keeps TOTAL Avg comparatively flat.

---

## CU2 — Refinement stage only: per-query cost (CPU vs CUDA)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU1_Refine_CPU_vs_CUDA_dark.png">
  <img alt="Refine stage only cost per query (CPU vs CUDA, Q=1000/Q=10000)" src="performance_images/CU1_Refine_CPU_vs_CUDA_light.png">
</picture>

**Figure CU2.** **Refine stage only (distance + top-k)** latency per query, derived from `refine_ms_per_q`.  
This intentionally excludes ANN time and should not be interpreted as end-to-end latency.

**CUDA timing scope.** CUDA refine includes **H2D (queries + candidate IDs)**, **kernel**, and **D2H (top-k ids/dist)**. Base vectors are **cached on GPU** and excluded from per-run H2D.

---

## CU2b — Kernel-level optimization: half2 + ILP vectorization (FP16 base)

Before evaluating alternative merge strategies (e.g., warp-level merge), we first optimized the **FP16 L2 distance kernel** by introducing **half2-based loading/compute and instruction-level parallelism (ILP)**. This targets the dominant portion of CUDA refinement—the distance accumulation over `D=384`—and primarily reduces **kernel time** rather than transfer overhead.

Under the same workload (`R=500`, `Q=10000`, `k=10`, ids-only D2H; base vectors cached on GPU), the optimized kernel reduces CUDA refinement time as follows:

| Variant       | refine_ms_total (ms) | H2D (ms) | kernel (ms) | D2H (ms) | avg per query (ms/query) |
| ------------- | -------------------: | -------: | ----------: | -------: | -----------------------: |
| Pre-half2/ILP |              44.0914 |   1.4663 |     42.6056 |   0.0194 |                 0.004409 |
| half2 + ILP   |              29.8578 |   1.4494 |     28.3889 |   0.0194 |                 0.002986 |

**Table CU2b-1.** Illustrative single-run snapshots for the FP16 refine kernel.  
**Interpretation.** The improvement is overwhelmingly explained by the **kernel reduction** (42.61 → 28.39 ms), while H2D and D2H remain essentially unchanged, consistent with a compute-side speedup from **vectorized FP16 loads** and **higher ILP** in the inner loop.

> Note: These example measurements were collected with `CUDA_PINNED=1` and `CUDA_RETURN_DIST=0` (ids-only). Base vectors are cached on GPU and excluded from per-run H2D. Paired statistical comparisons are reported in CU4.

---

## CU3 — Latency breakdown (Avg): ANN (CPU) + Refinement (CPU vs CUDA), Q=10000

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU1_Breakdown_Bar_dark.png">
  <img alt="Latency breakdown: ANN vs refinement cost (Q=10000)" src="performance_images/CU1_Breakdown_Bar_light.png">
</picture>

**Figure CU3.** Average-latency decomposition at representative `REFINE_K` values.  
Refinement portion is computed as **TOTAL Avg − ANN Avg** (from the same run), highlighting that in this phase the **only GPU acceleration is refinement**.

---

## CU4 — Warp-merge vs. baseline: paired Δ (nPairs=30, 95% CI)

To evaluate whether **warp-level merging** improves the CUDA refinement kernel beyond the baseline block-level merge, we run a paired A/B test under identical conditions (same Q, same candidates, same `R=500`, same build), alternating `CUDA_KERNEL_MODE=baseline` and `CUDA_KERNEL_MODE=warpmerge`. We report paired mean differences with 95% CI. Negative Δ indicates warp-merge is faster.

We track two metrics:
- **Total refine Δ (incl. H2D/D2H)**: paired Δ on `refine_ms_per_q` (H2D + kernel + D2H).
- **Kernel-only Δ**: paired Δ on `refine_kernel_ms_per_q` (kernel time only), to isolate compute/merge behavior from transfer effects.

**Test scope (baseline policy):** `CUDA_PINNED=0`, `CUDA_RETURN_DIST=0` (ids-only), `PIPELINE=staged`, `R=500`, `D=384`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU4_Q10000_dark.png">
  <img alt="CU4 paired delta (warpmerge - baseline), Q=10000" src="performance_images/CU4_Q10000_light.png">
</picture>

**Figure CU4a.** Paired Δ (warp-merge − baseline), `Q=10000`, `nPairs=30`, 95% CI.  
Y-axis is **µs/query**. Blue is total refine Δ (incl. H2D/D2H); red is kernel-only Δ.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU4_Q1000_dark.png">
  <img alt="CU4 paired delta (warpmerge - baseline), Q=1000" src="performance_images/CU4_Q1000_light.png">
</picture>

**Figure CU4b.** Same paired Δ analysis under `Q=1000` (short-run). This setting is more sensitive to run-to-run variance, but is useful to confirm whether kernel-only direction matches steady-state runs.

|     Q |  K | Metric      | meanDiff (ms/query) | 95% CI (ms/query)      | Interpretation   |
| ----: | -: | ----------- | ------------------: | ---------------------- | ---------------- |
|  1000 | 10 | kernel-only |           +0.000063 | [+0.000059, +0.000067] | warpmerge slower |
|  1000 | 20 | kernel-only |           −0.001367 | [−0.001374, −0.001360] | warpmerge faster |
|  1000 | 30 | kernel-only |           −0.000377 | [−0.000382, −0.000372] | warpmerge faster |
| 10000 | 10 | kernel-only |           +0.000296 | [+0.000295, +0.000297] | warpmerge slower |
| 10000 | 20 | kernel-only |           −0.001034 | [−0.001036, −0.001031] | warpmerge faster |
| 10000 | 30 | kernel-only |           −0.000302 | [−0.000304, −0.000300] | warpmerge faster |

**Table CU4-1.** CU4 summary table (kernel-only)

**Interpretation.** Warp-merge shows a K-dependent behavior: it is slightly slower at `K=10`, but becomes measurably faster at `K=20/30`. Importantly, the same pattern appears in the **kernel-only** Δ, indicating the effect is attributable to the kernel’s merge/compute path rather than transfer variability.

---

## Takeaway (baseline `CUDA_PINNED=0`)

- **Correctness preserved:** CUDA refinement matches CPU-refined recall under the same candidates (verified via GT-cached evaluation).
- **Where CUDA helps:** As `REFINE_K` grows, CPU refinement starts to dominate end-to-end latency; CUDA keeps refinement cost small, improving TOTAL Avg especially in steady-state runs (`Q=10000`).
- **What remains CPU-bound:** Candidate generation (FAISS OPQ-PQ search) still determines most of the “floor” latency at moderate `REFINE_K`.
- **Kernel strategy insight (CU4):** Warp-merge is not universally better; it depends on **K**. In this setup, warp-merge is slower at `K=10` but faster at `K=20/30`, and the effect is visible in kernel-only timings.

---

## Optional: pinned host memory (`CUDA_PINNED=1`)

Pinned memory can reduce **H2D** time (queries + candidate IDs). In this phase, it is treated as an optional improvement rather than baseline. If enabled, report pinned results side-by-side with baseline for the same (`Q`, `REFINE_K`) to avoid conflating “algorithmic gain” with “transfer optimization.”

## CU5 — Warp-merge vs. baseline: paired Δ vs. K (kernel-only, Q=10000, R=500)

We investigate the hypothesis that the **warp-merge benefit is merge-bound and sensitive to block parallelism**, by repeating the paired A/B test while **forcing CUDA block threads** (`CUDA_BLOCK_THREADS ∈ {128, 256}`) and sweeping `k ∈ {10,20,30}` under a fixed refinement budget (`REFINE_K=500`, `R=500`, `D=384`).
We report `paired mean differences` (warpmerge − baseline) with **95% CI** over `nPairs=30 `runs. **Negative Δ means warp-merge is faster.** Baseline settings: `CUDA_PINNED=0`, `CUDA_RETURN_DIST=0 (ids-only)`, `PIPELINE=staged`, OPQ-PQ (`nlist=4096`, `nprobe=64`), 500K FP16 base, GT-cached L2.

### CU5-A — Kernel-only Δ vs K (primary evidence)
<picture> <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU5_A_Kernel_Dark.png"> <img alt="CU5 kernel-only paired delta vs K (threads=128/256, Q=10000, R=500)" src="performance_images/CU5_A_Kernel_Light.png"> </picture>

**Figure CU5-A.** Paired Δ on **kernel-only** time (`refine_kernel_ms_per_q`) vs `k`, comparing forced `CUDA_BLOCK_THREADS=128` vs `256` (Q=10000, R=500). This isolates compute/merge behavior from H2D/D2H transfer variability.

|     Q | Threads |  K | Metric      | meanDiff (ms/query) | 95% CI (ms/query)      | Interpretation                      |
| ----: | ------: | -: | ----------- | ------------------: | ---------------------- | ----------------------------------- |
| 10000 |     128 | 10 | kernel-only |           +0.000706 | [+0.000681, +0.000730] | warpmerge slower                    |
| 10000 |     128 | 20 | kernel-only |           +0.000095 | [+0.000094, +0.000097] | warpmerge slower                    |
| 10000 |     128 | 30 | kernel-only |           −0.000303 | [−0.000305, −0.000301] | warpmerge faster                    |
| 10000 |     256 | 10 | kernel-only |           +0.000294 | [+0.000279, +0.000309] | warpmerge slower                    |
| 10000 |     256 | 20 | kernel-only |           −0.001038 | [−0.001041, −0.001036] | **warpmerge faster (largest gain)** |
| 10000 |     256 | 30 | kernel-only |           −0.000302 | [−0.000304, −0.000300] | warpmerge faster                    |

**Table CU5-1.** Kernel-only summary (paired Δ, Q=10000, nPairs=30)

**Interpretation.** The improvement peaks at **K=20 with Threads=256**, where warp-merge reduces kernel time by **~1.04 μs/query**. In contrast, at **Threads=128**, warp-merge is slower for K=10/20 and only becomes beneficial at K=30. This supports a thread/parallelism interaction: **warp-merge amortizes best when enough thread-level work exists to make the baseline’s block-level merge comparatively expensive, but not so large that additional shared-memory traffic dominates.**

### CU5-B — Total refine Δ sanity check (incl. H2D + kernel + D2H)

Kernel-only is the primary evidence (mechanism-level). For completeness, we also compute paired Δ on **total refine** time (`refine_ms_per_q`, includes H2D + kernel + D2H). As expected, effect sizes are typically **smaller** due to transfer components, but the **direction** generally matches the kernel-only results.

|     Q | Threads |  K | Metric       | meanDiff (ms/query) | 95% CI (ms/query)      | Interpretation   |
| ----: | ------: | -: | ------------ | ------------------: | ---------------------- | ---------------- |
| 10000 |     128 | 10 | total refine |           +0.000705 | [+0.000678, +0.000732] | warpmerge slower |
| 10000 |     128 | 20 | total refine |           +0.000107 | [+0.000086, +0.000128] | warpmerge slower |
| 10000 |     128 | 30 | total refine |           −0.000306 | [−0.000327, −0.000285] | warpmerge faster |
| 10000 |     256 | 10 | total refine |           +0.000294 | [+0.000274, +0.000313] | warpmerge slower |
| 10000 |     256 | 20 | total refine |           −0.001043 | [−0.001069, −0.001016] | warpmerge faster |
| 10000 |     256 | 30 | total refine |           −0.000275 | [−0.000308, −0.000242] | warpmerge faster |

**Table CU5-2.**  Total refine Δ  summary (incl. H2D + kernel + D2H)

**Takeaway.** The strongest gain at **K=20** is not a transfer artifact: it appears in **kernel-only** and persists (though diluted) in **total refine**. The forced-thread experiments show that warp-merge’s advantage is conditional on **block parallelism**, consistent with a merge-cost vs occupancy/shared-memory tradeoff.