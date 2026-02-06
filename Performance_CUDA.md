# CUDA Phase — Step 1: CUDA Refinement (GPU L2 re-ranking)

This phase is the first **CUDA integration milestone**: we offload only the **refinement step** (exact L2 re-ranking over top-`REFINE_K ANN` candidates) to GPU, while keeping **candidate generation unchanged** (FAISS OPQ-PQ on CPU).
The goal is to establish a **correct, measurable CUDA path** with clean timing boundaries before CUDA-izing additional pipeline components (e.g., GPU-resident query/candidate buffers, overlapping transfers, or eventually moving more of scoring/top-k to GPU).

**Fixed setup (unless noted)**: OPQ-PQ (m=64, b=8), `nlist=4096`, `nprobe=64`, `k=10`, base=500K FP16, metric=L2 with cached GT.
**Pipeline accounting**: `PIPELINE=staged` (ANN candidate generation and refinement measured as separate stages; TOTAL = ANN + refine).
**Pinned memory policy**: `CUDA_PINNED=0` is the **baseline**. `CUDA_PINNED=1` is treated as an **optional optimization** (reported separately only when needed).

## CU1 — End-to-end impact (TOTAL Avg) when only refinement is CUDA-accelerated
<picture> <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU1_Total_Avg_Latency_dark.png"> <img alt="Total average latency vs REFINE_K (CPU vs CUDA, Q=1000/Q=10000)" src="performance_images/CU1_Total_Avg_Latency_light.png"> </picture>

**Figure CU1**. Total average latency (ANN + refine) vs `REFINE_K`, comparing **CPU refine** vs **CUDA refine** (ANN remains CPU/FAISS).
> Note: `Q=1000` is a **short-run** measurement and is more sensitive to run-to-run variance; `Q=10000` better reflects **steady-state** throughput.

**Reading the curve.** With ANN fixed on CPU, CUDA refinement mainly matters when `REFINE_K` becomes large (e.g., 200–500), where CPU refine begins to dominate TOTAL Avg, while CUDA keeps TOTAL Avg comparatively flat.

## CU2 — Refinement stage only: per-query cost (CPU vs CUDA)
<picture> <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU1_Refine_CPU_vs_CUDA_dark.png"> <img alt="Refine stage only cost per query (CPU vs CUDA, Q=1000/Q=10000)" src="performance_images/CU1_Refine_CPU_vs_CUDA_light.png"> </picture>

**Figure CU2.** **Refine stage only (distance + top-k)** latency per query, derived from `refine_ms_per_q`.
This intentionally excludes ANN time and should not be interpreted as end-to-end latency.

**CUDA timing scope.** CUDA refine includes **H2D (queries + candidate IDs)**, **kernel**, and **D2H (top-k ids/dist)**. Base vectors are **cached on GPU** and excluded from per-run H2D.

## CU3 — Latency breakdown (Avg): ANN (CPU) + Refinement (CPU vs CUDA), Q=10000
<picture> <source media="(prefers-color-scheme: dark)" srcset="performance_images/CU1_Breakdown_Bar_dark.png"> <img alt="Latency breakdown: ANN vs refinement cost (Q=10000)" src="performance_images/CU1_Breakdown_Bar_light.png"> </picture>

**Figure CU3.** Average-latency decomposition at representative `REFINE_K` values.
Refinement portion is computed as **TOTAL Avg − ANN Avg** (from the same run), highlighting that in this phase the **only GPU acceleration is refinement**.

#### Takeaway (baseline `CUDA_PINNED=0`)

* **Correctness preserved:** CUDA refinement matches the CPU-refined Recall@10 under the same candidates (same refined top-k set, verified via GT-cached evaluation).
* **Where CUDA helps:** As `REFINE_K` grows, CPU refinement starts to dominate end-to-end latency; CUDA keeps refinement cost small, improving TOTAL Avg especially in **steady-state runs** (`Q=10000`).
* **What remains CPU-bound:** Candidate generation (FAISS OPQ-PQ search) still determines most of the “floor” latency at moderate `REFINE_K`.

#### Optional: pinned host memory (`CUDA_PINNED=1`)

Pinned memory can reduce **H2D** time (queries + candidate IDs). In this phase, it is treated as an optional improvement rather than baseline. If enabled, report pinned results side-by-side with baseline for the same (`Q`, `REFINE_K`) to avoid conflating “algorithmic gain” with “transfer optimization.”