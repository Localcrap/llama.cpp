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

## 2026-10-05 correction: 256k pp numbers are workload-dependent (user caught it)

commit: 8681c2e63+
The "~212 t/s warm" figure is the fully-converged large-prompt best case
only. The general per-request reality at 262144 ctx / ncmoe 43:

| request | measured |
|---|---|
| small prompt (49-300 tok) | 8-17 s each => 6-20 t/s |
| 7021-tok prompt, warming | 140 s -> 70 s (50-100 t/s) |
| 7021-tok prompt, warm | 34 s => 204 t/s |
| tg (no MTP) | 8-11 t/s |

Why: every request streams ~87 GiB of experts over x4 (~12-13 s floor,
nothing VRAM-resident at ncmoe 43), the 89 GiB expert set thrashes the
page cache (89 > ~79 usable), and the DSA indexer/kpool work grows
linearly with n_past (a 147-token chunk costs 13.9 s at n_past ~= 250k).

decision: 256k ctx on this box is viable for warm batch processing of
long documents, NOT for interactive small-prompt use (10-17 s per
message). Interactive + long-context needs the memory budget problem
solved (see 5800X dual-GPU analysis) or a smaller model.

## 2026-10-07 mainline master (MoE cache PR #29887) vs glm5-min A/B - same box, same harness
commit: bd4eeaa04 (mainline master, MoE cache merged) vs glm5-min HEAD
config: ab-ml.sh, 32k ctx, ub 2048, t12/16, fa, -fit off, 2 warm pp + 1 tg

| config | pp warm | tg |
|---|---|---|
| mainline streaming (ncmoe 41) | 210.6 t/s | 8.01 t/s |
| mainline -cmoe --moe-cache-mib 13000 | 230.4 t/s | **3.58 t/s** |
| mainline + MTP (mainline's own, #29928) | 122.2 t/s | 7.77 t/s |
| glm5-min streaming | 215.2 t/s | 8.27 t/s |
| glm5-min + MTP (essay gen) | 197.2 t/s | 7.69 t/s |

| PCIe rx MB/s (p50/p90/max) during pp+tg |
|---|
| cmoe: 4455/6580/13720 - streaming: 1606/6806/7032 |

decision: mainline == fork on pp and decode (within noise). The MoE
expert cache (PR #29887) is a decode REGRESSION on this box: 13 GiB
cache = 15% coverage of 89 GiB host experts; GLM's hit rate at that
coverage is far below the ~75% break-even on a x4 link (the thread's
own GLM data point: 47% hit on x16 => uploads became the bottleneck).
MTP is neutral for open-ended generation, +10-20% on predictable text.
 recommendation: either branch performs identically with our tuned
settings; mainline adds the cache option (useful when PCIe is x16 or
experts are smaller) at the cost of re-tracking our patches.

## 2026-10-10 PolyStrata strata-poly vs llama.cpp on the 3090 x4 box (UD-IQ2_XXS)

hot-expert VRAM cache + concurrent CPU cold experts + MTP/lookup guessing.
Full table + mechanics + fixes: docs/findings/2026-10-polystrata-glm.md

| ctx | strata-poly tg fresh/adapted | llama.cpp tg |
|---|---|---|
| 32K | 13.4-15.8 / 18-22.1 | 8.0-8.3 |
| 128K | 10.4 / n.m. | n.t. |
| 256K | 6.6 / n.t. | 8-11 |

pp: strata 108-181 vs llama.cpp 210 (both x4-streaming-bound).
adopt: strata-poly for <=128K interactive (2-2.7x decode), llama.cpp for 256K.
no REAP50 needed: lazy page-cache residency fits in 79 GiB usable (no thrash).
