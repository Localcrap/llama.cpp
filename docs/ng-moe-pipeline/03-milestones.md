# 03 - Milestones

Incremental ladder. Each milestone: scope, gate (go/no-go, measured),
deliverable, est. effort. Gates use `bench-moe-cpu/mmid.cpp` (isolated op)
and llama-bench pp8192 (in-situ, `GGML_CUDA_MMI_ON_CPU=1`, ncmoe 39,
12 threads). **A failed gate stops the ladder** - the state remains
servable via the default streaming config.

## M0 - Harness hardening (prereq, ~half day)

- [ ] `mmid.cpp`: add golden-output capture (reference run -> md5 of dst
      per (seed, shape)) + tolerance compare mode; add per-type variants
      (iq3_xxs, iq2_s, iq4_xs shapes from the model)
- [ ] one-key repro script: build + run matrix + append results table to
      `docs/ng-moe-pipeline/results.md` (committed history of every gate)
- [ ] baseline row recorded (current: 207 ms/op, 3.0 GB/s, 917 GMAC/s)

**Gate:** golden compare catches a deliberately-injected off-by-one;
results table committed.

## M1 - Bankable repairs (X1 + X2 + X3 + A2) (~1-2 evenings)

- [ ] X1 hoist activation staging out of the weight-group loop
- [ ] X2 work-stealing over (expert, group) tickets
- [ ] X3 prefetch of next weight group
- [ ] A2 8x32 microtile (register-budget audit first)

**Gate:** isolated op <= 150 ms (>= 4.1 GB/s) AND llama-bench pp >= 118 t/s
with correctness green. Expected actual: 120-135 ms / 150-165 t/s.
**On fail:** bisect the four changes, keep what passed, proceed to M2
anyway (M2 does not depend on M1's internals).

## M2 - Fused kernel prototype, isolated (~2-4 evenings, the decisive gate)

- [ ] `fused_bench.cpp`: standalone inner-loop prototype (no ggml
      integration): staged activations + raw iq2_xxs weights -> dst,
      compared against the tiled path via M0 goldens
- [ ] iterate on expansion ILP (8+ interleaved groups, scalar-LUT loads,
      L1-pinned tables); 8x16 tile first, widen under measurement
- [ ] if green: repeat for iq3_xxs (4 KB grid - watch L1)

**Gate:** >= 2.0 TMAC/s aggregate (>= 33% VNNI peak) on the [4096x2048x288]
iq2_xxs shape with correctness green, i.e. op <= ~86 ms.
**Stretch gate:** >= 2.5 TMAC/s.
**On fail:** record the measured expansion cost; evaluate C (operand swap)
with the same prototype harness; do not integrate.

## M3 - Integration (~1-2 evenings)

- [ ] new `ggml_compute_forward_mul_mat_id_ng` behind
      `GGML_CPU_MOE_NG=1` (runtime env, no build flag), reusing the
      scheduler-facing surfaces of the tiled path
- [ ] narrow-path (few routed rows) falls back to the existing tiled code
- [ ] work-stealing from M1 carried over

**Gate:** llama-bench pp >= 200 t/s (beats streaming's 147-161 bench
number) with green goldens + end-to-end server sanity. Server flips its
documented default to NG with streaming as fallback.
**On fail:** keep NG behind the env; streaming stays default; document the
delta.

## M4 - Polish to target (~open-ended, evening-sized chunks)

- [ ] per-type kernels (iq2_s, iq4_xs; iq3_xxs if not done in M2)
- [ ] per-core tile-size selection (Zen5 vs Zen5c)
- [ ] decode-path validation: tg >= 13 t/s with NG experts (MTP on);
      if decode regresses, route decode-sized batches (nrows <= 16) to
      the incumbent path automatically
- [ ] skew handling: experts with 1-5 rows
- [ ] prefetch tuning pass

**Gate (final):** server pp >= 270 t/s sustained on the 8k/19k prompt
suite, tg >= 13 t/s, quality suite green (04-validation). Stretch 350+.

## Reporting

Every gate run appends to `docs/ng-moe-pipeline/results.md`:
date, commit, config, table (ms/op, GB/s, GMAC/s, pp, tg), decision.
Failed gates get a paragraph on why - the negative results are the
expensive part.
