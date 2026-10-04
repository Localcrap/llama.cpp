# 01 - Measured baseline

Every design decision in this plan traces to a number on this page. All are
reproducible via [`bench-moe-cpu/`](../../bench-moe-cpu/) unless marked
otherwise. Environment: glm5-min build-cuda (GGML_NATIVE, sm_86), 12 threads
unless noted, model shapes from GLM-5.3-Flash UD-IQ2_XXS.

## 1. Workload shape (per 2048-token ubatch, 39 CPU-expert layers)

| quantity | value | derivation |
|---|---|---|
| expert tensors (ops) | 117 | 39 layers x (up, gate, down) |
| weight bytes touched per op | 604 MB | [4096x2048x288] @ 2.0625 bpw |
| weight bytes per ubatch | ~78 GiB | 117 x ~670 MB avg (down is 2x rows) |
| routed rows per expert | ~71 avg | 2048 tok x 10 used / 288 experts (skewed) |
| MACs per ubatch | ~20.1 TMAC | 117 x ~172 GMAC |

## 2. Engine throughputs (isolated, `mmid.cpp`, 12 threads)

| engine | ms/op | GB/s (weights) | GMAC/s | % VNNI peak |
|---|---|---|---|---|
| tiled (current) | 207 | 3.0 | 917 | ~15% |
| vec_dot stock | 860 | 0.7 | 200 | ~3% |
| GPU staging (x4) | - | 6.5-7.0 | - | - |
| **theoretical floor** | ~35-45 | 45 (RAM) | 6000 (VNNI) | 100% |

Theoretical floor: max(78 GiB / 45 GB/s, 20.1 TMAC / 6 TMAC/s) ~= 3.4 s/ubatch
=> ~550 t/s pp ceiling with a perfect pipeline. With realistic 60-70% of
floor: 350-420 t/s.

## 3. Where the 207 ms goes (perf, pp32768, whole-pipeline shares)

| stage | share | absolute (ms/op equiv) | bound by |
|---|---|---|---|
| `tiled_run_micro_vnni_8x16<32>` | 45% | 93 | load ports (see 4) |
| `tiled_unpack_src1_q8_K` + repack | ~10% | 21 | 32x redundancy (per weight-group!) |
| `tiled_unpack_src0` (iq2/iq3/iq2s) + repack | ~15% | 31 | scalar LUT + 4x data amplification |
| `tiled_store_window_scatter` | 2.5% | 5 | |
| `quantize_row_q8_K` (activations) | 2.3% | 5 | |
| gomp barriers + tail | ~7% | 15 | static split, Zen5c stragglers |
| remainder (ids/sort/copy glue) | ~18% | 37 | |

## 4. The microkernel cost model (measured + counted)

Per 8x16 microtile call (SUBBLK=32, from source + disassembly):

- 512 `vpdpbusd` (VNNI MACs: 32K MACs/call)
- **512 scalar `u32` loads** (`src0.q[(i0+t)*qk_stride + qk_off + kg*4]`)
- 512 `_mm512_set1_epi32` broadcasts
- ~100 other (bsums, scales, epilogue)

Zen5 port arithmetic: 512 scalar loads at 2-3/c = 170-250 cy; 512 dpbusd at
2/c = 256 cy; they contend for the same load/ALU pressure => measured
~1000 cycles/call => 32K MACs / 1000 cy = 32 MACs/cy/core. Measured 917
GMAC/s / 12 threads = 76 GMAC/s/core / 4.0 GHz = 19 MACs/cy/core (with
memory stalls on top) - consistent. **The math runs at 25-50% speed because
of how it is fed.**

## 5. Dequant/unpack costs (isolated, single core, L2-resident)

| variant | output GB/s | per-32-codes cost |
|---|---|---|
| reference AVX2 (scalar LUT loads) | 4.7 | ~27 cy |
| v1 gather (vpgatherdq + multishift) | 3.9 | ~33 cy |
| v2 column tables (vpermi2b x16/64B) | 1.7 | ~75 cy |

Vector LUT approaches lose on Zen5: `vpermi2b` is multi-uop, gathers
pipeline poorly. The reference's scalar loads are fine *when L1-resident*;
what kills the stock vec_dot path in-situ is LUT thrash under a 604 MB
stream + latency chains (0.06 GB/s/core effective). Implication for NG:
**dequant LUT tables must stay L1-hot and expansion must interleave with
many independent streams; do not chase vpermi2b/gather tricks.**

## 6. Memory system

- RAM: weights stream cold; practical aggregate ~45 GB/s for 12-thread
  sequential-ish reads (STREAM-class). Each core can keep ~10-16 outstanding
  L2 misses; MLP is the limiting factor for scattered patterns.
- Expert access pattern per op: 288 experts x ~71 rows, row stride 8-16 KB
  (rows of the same expert are strided in the packed tensor): prefetchable
  stride patterns if processed row-major per expert.
- L2: 1 MB/core Zen5, 0.5 MB Zen5c. Activation tile (256 rows x 256 vals
  q8_K = 64 KB + bsums) is L2-resident by design; keep it that way.

## 7. Thread topology

- 4x Zen5 @ 5.1 + 8x Zen5c @ 3.3 (1.55x speed ratio). Static row-group
  splits leave Zen5 cores waiting at the gomp barrier (measured: pp flat
  from 12 -> 16 -> 24 threads).
- Decode (tg) prefers 12 threads; SMT-24 collapses it. pp unaffected.

## 8. Integration constraints

- Ops must remain expressible as ggml MUL_MAT_ID on CPU backend (or a
  drop-in `ggml_compute_forward_mul_mat_id*` replacement behind an env).
- Quant types to cover (by tensor count in this model): iq2_xxs (82),
  iq3_xxs (54), iq4_xs (6), iq2_s (2). All share the "grid LUT + keven
  signs + subblock scales" family; kernels are per-type templates.
- Correctness bar: numeric drift vs the current tiled path within fp32
  accumulation reordering tolerance (see 04-validation).
