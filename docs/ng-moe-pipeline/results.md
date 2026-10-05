# NG MoE pipeline - results log

Append one block per gate run. Format:

    ## YYYY-MM-DD <milestone> <GATE: pass|fail>
    commit: <sha>
    config: <one-liner>
    | metric | value | gate | note |
    decision: <proceed/bisect/stop + why>

## 2026-10-04 baseline (M0 reference row)

commit: 2a6f795a1 (glm5-min)
config: mmid.cpp [4096x2048x288] iq2_xxs, 2048 tok x 10 used, 12 threads

| metric | value |
|---|---|
| ms/op | 207 |
| GB/s (weights) | 3.0 |
| GMAC/s | 917 |
| llama-bench pp8192 (CPU path) | 97-103 t/s |
| llama-bench pp8192 (streaming, default) | 147-161 t/s |

decision: baseline recorded

## 2026-10-04 M0 gate run
commit: 74eb95abf
config: engine=GGML_CPU_TILED_MM=1, 2048 tok x 10 used, 12 threads
| config | ms | GB/s | GMAC/s | check |
|---|---|---|---|---|
| up/iq2_xxs | 198.0 | 3.1 | 868 | PASS |
| down/iq2_xxs | 187.7 | 3.3 | 915 | PASS |
| up/iq3_xxs | 225.5 | 4.1 | 762 | PASS |
| down/iq3_xxs | 204.2 | 4.5 | 841 | PASS |
| up/iq2_s | 227.4 | 3.4 | 756 | PASS |
| up/iq4_xs | 240.8 | 5.3 | 713 | PASS |
decision: (fill in)

## 2026-10-04 M1 investigation notes (X1 rejected; VNNI-kernel status verified)

- **X1 (activation staging hoist) REJECTED by arithmetic**: extending the
  src1 tile from 256 to 4096 rows needs ~2.2 GiB per thread vs the 512 KiB
  TILED_WS_SLOT. The alternative (re-unpack per (expert, k-window)) still
  moves ~845 GiB of redundant q8 unpack per ubatch (44 experts x 8
  windows x ~2.4 GiB). Net loss; do not attempt.
- **VNNI kernel status**: the shipped libggml-cpu DOES run
  tiled_run_micro_vnni_8x16 for the hot instantiations (64x vpdpbusd +
  64x vpbroadcastd, fully unrolled, u32 loads hoisted to vmovq) - an
  initial wrong-function disassembly suggested otherwise; corrected.
  GGML_AVX512_VNNI=OFF in CMakeCache is cosmetic: -march=native defines
  __AVX512VNNI__ and the VNNI path is live. The load-port cost model in
  01-measured-baseline.md stands.
- Baseline re-verified after the CMake flag flip (no change): up/iq2_xxs
  221 ms, 2.8 GB/s, 776 GMAC/s. All goldens PASS.
- A2 (8x32 microtile): register audit says 8x32 does NOT fit (40+ zmm);
  8x20 (28 zmm) fits => only ~1.15x on the microkernel. The profitable
  A-variant is therefore NOT A2 but "A2'": keep 8x16 tiles, restructure
  the feed so weight bytes are loaded as vectors and broadcast once per
  4 rows (see M2 fused prototype; this merges A into B).

decision: M1's X1 dropped; A2 replaced by direct M2 work; M1 gate
reinterpreted as M2's gate (see 03-milestones).

## 2026-10-04 M2 fused prototype GATE: fail
commit: 9fa732eac
config: fused_bench.cpp, [2048 x 4096] iq2_xxs expert, 16 act rows, 1 core

| metric | value | gate | note |
|---|---|---|---|
| GMAC/s (1 core) | 17.5 | >= 167 | ~10x short |
| correctness | PASS | PASS | exact match vs scalar reference |

decision: STOP per plan. Mechanism: ~1100 scalar ops per 256 weight
values in the expansion (movzbl/movw-dominated profile); iq2_xxs's
LUT code format cannot feed VNNI. Post-failure cards (codebook
pre-expansion, operand swap, wider bands) all bounded below gate.
The CPU path cannot beat streaming for this quant; project closed.
See bench-moe-cpu/M2-VERDICT.md.

## 2026-10-05 256k-context serving config (user-run + verification)
commit: 9fa732eac+ (launcher env: CTX=262144)
config: ncmoe 42 (MTP off frees the VRAM for it), --lazy-mode off, -fit n/a

| metric | value | note |
|---|---|---|
| VRAM | 23.6 GiB | ncmoe 42 fits only with MTP off (42+MTP OOM'd, confirmed) |
| pp warm-up | 62.6 -> 135.7 -> 211.6 t/s | 3 passes to converge page cache |
| pp warm | ~212 t/s | same as the 32k-era warm number |
| tg | 12.6 t/s | MTP off; enable MTP when VRAM allows |
| RAM | 10 used + 46 cache | eager load kept ~25 GiB off the page cache vs lazy |

decision: 262144 ctx works at full warm speed with ncmoe 42 + MTP off +
lazy off. Cold start costs ~2 warm-up passes (~3 min) after every boot.
256k caveat: DSA indexer cost grows with n_past but is minor vs the
expert streaming; measured warm pp matches 32k-era numbers.
