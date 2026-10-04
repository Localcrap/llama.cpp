# 04 - Validation

## Correctness

Three layers, cheap to expensive:

1. **Golden op outputs** (M0 harness): reference tiled path over fixed
   seeds x the model's real shapes (iq2_xxs/iq3_xxs/iq2_s/iq4_xs x
   up/gate/down). New engines must match within tolerance:
   `|x - ref| <= 1e-4 * (1 + |ref|)` per element AND md5 equality of the
   activation-side quantization (the q8_K staging must be bit-identical;
   drift may only come from fp32 accumulation reordering inside the MAC
   tree). A deliberately injected off-by-one must be caught (harness
   self-test).
2. **End-to-end server sanity** (after M3): the standard suite -
   `17*23 -> 391`; a fixed 6-prompt quality set (arithmetic, recall from
   context, summary vs source facts, code edit) eyeballed against outputs
   from the streaming config; token-count drift < 20% on temp-0 prompts.
3. **Soak**: 2 h mixed pp/tg traffic through the tailscale endpoint before
   any default flip (catches scheduler/threading races in work-stealing).

Numeric-drift budget rationale: MUL_MAT_ID sums 4096 products per output;
reordering changes results at ~1e-6 relative. Anything beyond 1e-4 relative
indicates a real bug, not drift.

## Performance methodology

- **Isolated op** (`bench-moe-cpu/mmid.cpp`): 5 iterations after 2 warmups,
  report median; pinned threads (`-C` masks when needed); record
  GB/s (weight bytes) and GMAC/s.
- **In-situ** `llama-bench -p 8192 -n 32 -r 3` with
  `GGML_CUDA_MMI_ON_CPU=1 --n-cpu-moe 39 -ub 2048 -t 12`, warm page cache
  (one discarded pp32768 pass first). Server numbers via the existing
  warmup-then-measure scripts (see git history of run-glm-q2-spf.sh docs).
- **Environment discipline**: no builds or other jobs concurrent with
  measurement; `perf` only for attribution, never during gates; every
  results-table row notes the binary commit.
- Known noise sources (from the investigation): page-cache state (~+/-10%
  on first pass), governor=performance unavailable (no sudo) => accept
  +/-3% between runs, gates set with margin.

## Regression protection for the serving path

The default config only changes when an M-gate explicitly says so; the
streaming config must remain one env-var away:
`GGML_CUDA_MMI_ON_CPU` (unset) = streaming, `=1` (+`GGML_CPU_MOE_NG=1`
once it exists) = NG CPU path. Launcher gains a `PP_ENGINE` env that maps
to these and documents them.
