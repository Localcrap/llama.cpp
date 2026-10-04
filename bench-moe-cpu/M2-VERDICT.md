# M2 fused dequant-GEMM prototype: verdict

**Gate: FAIL.** The fused kernel (inline scalar-LUT expansion feeding vpdpbusd)
measures **17.5 GMAC/s per core** where the gate needed >= 167/core
(2.0 TMAC/s aggregate over 12 threads). Short by ~10x.

## Mechanism (perf-verified)

The hot loop executes ~1100 scalar x86 ops per 256 weight values:
byte extraction (movzbl/movw/movl dominate the profile), sign math, and
u32 packing, against only 2 vpdpbusd-equivalents of useful math per
group. The expansion dependency chains + LUT loads cannot be hidden by
ILP: the compiler scalarizes the whole expansion because the byte-lane
math IS scalar. AVX-512 vector LUTs were already shown to lose on Zen5
(vpermi2b multi-uop; see findings doc); scalar-LUT chains lose by worse.

## Design cards considered after the failure

1. **Weight-side codebook pre-expansion** (kindex-style: precompute all
   (grid entry, sign mask) pairs into a table addressed by 11 bits, 64K
   entries x 16B = 1 MB shared): replaces ~30 scalar ops with one 16B
   load per 8 values. Estimated ceiling ~80-100 GMAC/s/core - still
   short of 167. Also the table is shared L2 pressure across 12 threads.
2. **Operand swap (candidate C)**: same expansion bottleneck on the
   weight side; swapping operands moves the broadcast to activations
   but the expansion still must produce weight u32s. No win.
3. **Wider act bands (32 lanes)**: amortizes the broadcast but not the
   expansion. Bounded by the same 1100-ops-per-256-values wall.

## Conclusion

The ~5x theoretical headroom in the CPU MoE path is real (RAM bandwidth
+ VNNI peak), but *iq2_xxs's code format* (4-bit-indexed 2 KB grid +
keven signs per 8 values) makes per-use dequantification ~30x more
expensive than the MAC it feeds. The formats that would feed VNNI
natively are the "k-quant + separate scales" family (Q2_K: 2-bit
payload + 4-bit super-block scales) whose expansion is a shift/mask,
not a LUT walk. Since re-quantizing the model is out of scope
(quality-first constraint), **the NG pipeline project stops here**:
the CPU path cannot beat the PCIe streaming path (152-212 t/s) for
iq2_xxs on this hardware.

## What was proven on the way (bankable)

- The M0 gate harness (goldens + selftest + results log) - reusable as-is
- block layout ground truth (66-byte blocks: 2B fp16 d + 8 groups of
  [4 grid-idx | 4B sign/scale]) - encoded in fused_bench.cpp
- The fused-kernel correctness methodology (scalar reference sharing
  the true layout; bias-correction math for (v+128)*a semantics)
- Two independent confirmations that per-use LUT expansion cannot feed
  VNNI on Zen5 (vector-LUT and scalar-chain variants)
