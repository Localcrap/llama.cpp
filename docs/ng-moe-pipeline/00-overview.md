# Next-gen MoE CPU pipeline - overview

**Status:** planning | **Owner:** this project | **Date:** 2026-10-04
**Predecessor:** [CPU/MoE/PCIe investigation findings](../findings/2026-10-cpu-moe-and-pcie-investigation.md)

## Why this project exists

Dense pp on this box is PCIe-bound: the GPU streams ~78 GiB of host-resident
expert weights per 2048-token ubatch over x4 Gen4 (~7 GB/s) because the CPU
MoE path is 2.2x slower than the wire (3.0 GB/s tiled vs 7 GB/s PCIe). The
CPU pipeline's measured ceiling is ~5x higher than its current output
(~6 TMAC/s VNNI + ~45 GB/s RAM => ~500-550 t/s pp), gated entirely on
pipeline quality. This plan is the engineering path to close that gap.

## Goal

**Beat the streaming path by >= 1.7x on dense pp with CPU-executed MoE
experts, without changing model quality** (bitwise-identical quantization,
same math semantics, tolerance-level numeric drift only):

| metric | today (streaming) | target (NG pipeline) |
|---|---|---|
| pp8192 (llama-bench, ncmoe 39) | 147-161 t/s | **>= 270 t/s** (stretch 350+) |
| isolated MUL_MAT_ID op | 207 ms (3.0 GB/s) | **<= 90 ms** (>= 7 GB/s) |
| aggregate VNNI efficiency | 917 GMAC/s (15% peak) | >= 2.2 TMAC/s (37% peak) |
| tg with MTP | 14-15.4 t/s | not worse than 13 (CPU experts run decode too) |

Secondary goals: keep the engine runtime-selectable (`GGML_CPU_MOE_NG=1`),
keep decode unaffected or better, and structure the work so every milestone
is independently bankable.

## Non-goals

- No model changes (no requant; Q2_XXS stays).
- No GPU-side work (the streaming path is already at the PCIe ceiling; it
  remains the fallback config).
- No speculative prefill / MTP changes (separate track, already shipped).
- No attempt to overlap CPU and GPU expert compute: transformer layers are
  a dependency chain; per-ubatch execution is strictly serial across layers
  (candidate D in the design doc, formally rejected).

## Approach

Incremental, gate-driven. Three candidate architectures (A: fix the existing
tiled pipeline; B: fused dequant-GEMM - the main bet; C: operand-swapped
register blocking - research fallback), executed as milestone ladders where
**each milestone has a measurable go/no-go gate** and everything that passes
is committed and servable immediately. The first milestone that fails its
gate stops the ladder; the streaming config stays default until the NG path
beats it end-to-end.

## Documents

| file | content |
|---|---|
| [01-measured-baseline.md](01-measured-baseline.md) | every number the design leans on, with provenance |
| [02-design-candidates.md](02-design-candidates.md) | the pipeline designs, cost models, tradeoffs |
| [03-milestones.md](03-milestones.md) | the incremental ladder with gates |
| [04-validation.md](04-validation.md) | correctness + performance methodology |

## TL;DR of the bet

The current tiled microkernel is load-port bound spending 2 of every 3 cycles
feeding `vpdpbusd` with scalar 4-byte loads. Fixing *what feeds the math*
(not the math, not the LUTs) is worth ~2-3x. Removing the 4x data
amplification through scratch stages is worth another ~1.3-1.5x. Together
that clears the PCIe bar with margin. All of it is measurable in isolation
before touching the serving path.
