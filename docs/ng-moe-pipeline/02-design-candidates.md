# 02 - Design candidates

Three candidate architectures, ordered by risk. A is defensive (fix the
incumbent), B is the main bet (fused kernel), C is the research fallback.
They share cross-cutting work (X1-X3) and are *not* mutually exclusive: B
subsumes A's stages as it lands.

---

## X. Cross-cutting (apply to every candidate)

### X1 - Activation staging hoist (fixes the 32x redundancy)
Today `tiled_unpack_src1_q8_K` + `tiled_repack_src1` run inside the
weight-group loop: the activation tile for the same routed rows is rebuilt
for every one of the ~32 weight-row groups per expert. Hoist to
once-per-(expert, k-window); the 64 KB tile then stays L2-hot across all
groups. Expected: +8-12% op throughput, trivial diff, zero risk. **Do this
first regardless of anything else.**

### X2 - Work-stealing over expert-row groups
Replace the static `g0..g1` split with an atomic ticket counter over
(weight-row groups x experts) so Zen5 cores pull more work instead of
idling at the gomp barrier. Expected: +10-20% on heterogeneous cores
(measured flatness 12->24 threads says static split leaves ~30% idle).
Must keep per-thread scratch (tiled_ws) semantics.

### X3 - Weight-stream prefetch
`__builtin_prefetch(next_group_rows, 0, 1)` at chunk granularity ahead of
the unpack/consume loop; also consider `prefetchw` for the accumulator
window. Cheap, measured-MLP-bound patterns benefit.

---

## A. Incumbent repair (defensive ladder)

Keep the tiled architecture (unpack -> repack -> microkernel -> scatter).

**A2 - 8x32 microtile.** Process two `codes` registers (two 16-column
activation bands) per weight broadcast: the 512 scalar u32 loads + 512
broadcasts per call then serve 2x the MACs. Register budget: 16 acc zmm
(8 rows x 2 bands) + 8 s1_acc + 2 codes + ~4 aux = 30/32 - fits, tight.
Expected: microkernel ~1.5-1.7x => op 207 -> ~150 ms (4.1 GB/s).
Combined with X1/X2/X3: **~120-135 ms/op (4.7-5 GB/s) => pp ~150-165 t/s.
Reaches, does not beat, streaming decisively.** Bankable, low risk, and
the fallback if B fails.

**A3 - unpack->repack fusion.** Emit `tiled_unpack_src0` output directly
in the microkernel's interleaved layout, deleting `tiled_repack_src0` and
one scratch round-trip (saves ~4-8% op time, moderate diff risk: the
interleave layout is subtle - see the `kg%16/kg/16/row@4` mapping in
`tiled-kernel.cpp`).

Ceiling of the A ladder: ~5 GB/s. **A alone does not justify the project;
it de-risks and banks.**

---

## B. Fused dequant-GEMM (the main bet)

One pass over raw weight bytes; dequantize in registers; feed VNNI
directly. Deletes unpack_src0, repack_src0, and the tile scratch
round-trip for weights entirely.

### Structure (per expert, per k-window of 256 routed rows)

```
stage:  activations -> q8 tile layout (X1 hoist, reuse existing code)   [once]
loop g (weight-row groups, e.g. 64 rows):
  loop K-slab (256 of 4096):
    load:  raw weight bytes for 64 rows x slab                          [vector loads, prefetch X3]
    expand: per 32-value group -> 8x(4B u4 groups) in registers         [inline LUT, see below]
    mac:   dpbusd against the staged activation tile, accumulate        [VNNI]
  epilogue: scales/d/min application + scatter store                    [existing]
```

### The core question: can inline expansion feed VNNI?

Budget: to keep 2 dpbusd/cycle/core, the pipeline must produce one 64-lane
u4 vector (64 codes = 2 x 32-value groups) in <= ~4-6 cycles of added work,
with LUT tables L1-resident. Decomposition per 32-value group:

1. load 8 raw bytes (vector, amortized across groups)
2. grid magnitude expansion: 32 codes from `iq2xxs_grid[4 indices]`
3. sign application: keven-sign bits + bias-128
4. pack to 8 lanes x 4B groups

Steps 2-3 are the same work the unpack does today at ~27 cy/32 codes
(L2-resident scalar path). To hit ~5-6 cy/32 codes we need the *vector*
path to actually win - which Finding 4 says it currently doesn't
(v2 = 75 cy). **Resolution: don't expand per-32-group serially; expand
per-256-slab with 8+ independent groups in flight and scalar-LUT loads
(interleaved, ILP-hidden) instead of vpermi2b.** The reference's 27 cy is a
*latency-chain* number, not a throughput number: 8 interleaved chains at
~4-6 loads/cycle throughput => ~6-9 cy/32 codes sustained. This is the
central claim of candidate B and **Milestone M2 exists to prove it in
isolation before any integration** (gate: >= 2 TMAC/s on the fused
prototype).

### Why not keep two phases with better layouts?

Measured: the two-phase design moves ~4x weight bytes through scratch
(unpacked tiles are 4x the raw bytes) and pays two L1 round-trips. Fusion
removes both. The cost is register pressure and code complexity in the
inner loop - acceptable for a 2-3x target.

### B-side risk register

- register pressure starves the MAC stream -> mitigation: 8x16 tile first,
  widen later; measure at each step
- expansion ILP insufficient on Zen5c (narrower) -> per-core tile-size
  selection at runtime
- iq3_xxs grid is 4 KB (vs 2 KB): L1 pressure -> per-type table residency
  check; if needed, promote iq3 to A-ladder only
- skew (experts with 1-5 routed rows) -> narrow-path special case retained
  from today's tiled code

---

## C. Operand-swapped register blocking (research fallback)

Flip the dpbusd roles: `dpbusd(acc, weights_zmm, acts_broadcast)` where the
*weights* arrive as full-width vectors (16 lanes x 4B groups, loaded
normally) and the *activations* are the broadcast side. Then:
- weight feed is pure vector loads (the thing we cannot do today)
- but each lane accumulates a *different* k-group => per-output-element
  results need a horizontal reduction (16-lane tree) once per K-slab
  instead of per K-step - amortized 256x, acceptable
- activations must be pre-transposed into broadcast-ready 4B groups;
  they are small (64 KB tile) and reused across ~2048 weight rows per
  expert, so the transpose cost amortizes ~2000x

Highest ceiling (no scalar loads at all, no scratch amplification), highest
complexity (new activation layout + reduction epilogue + per-type dequant
of vector-loaded raw bytes feeding lanes directly... note: raw quant bytes
cannot feed dpbusd without expansion, so C still needs B's inline expansion
or an unpack pass). **Only attempted if B's M2 gate shows the expansion
budget cannot be met but MAC efficiency is otherwise reachable.**

---

## D. Rejected: CPU/GPU expert overlap

Transformer layers are a serial dependency chain within and across
ubatches (KV state). Per-layer expert placement alternation adds sync
without overlap opportunity. Formally dead; do not revisit.

---

## Selection logic

```
M1 (X1+X2+X3+A2)  --gate 4.5 GB/s-->  M2 (B prototype in isolation)
                                         |--gate 2 TMAC/s--> M3 (integrate B) --> M4 (polish, 350+)
                                         '--fail------------> keep M1 + A3, re-evaluate C
```

Every arrow produces a bankable, servable state; nothing speculative gets
merged.
