# GLM-5.3-Flash serving on RTX 3090 + Ryzen AI 9 HX 370: performance findings

**Scope:** everything measured during the 2026-10-03/04 hardware-utilization
sessions on this box, serving `GLM-5.3-Flash UD-Q2_XXS` (4 shards, 95 GiB,
43 MoE layers x 288 experts, ~2.06 bpw experts) via the `glm5-min` fork.
All numbers are reproducible with the tools in [`bench-moe-cpu/`](../bench-moe-cpu/)
and the commands inline below.

## The box

| part | detail |
|---|---|
| GPU | RTX 3090, 24 GiB, **PCIe Gen4 x4** (external link) |
| CPU | Ryzen AI 9 HX 370: 4x Zen5 @ 5.1 GHz + 8x Zen5c @ 3.3 GHz, 12C/24T |
| RAM | 91 GiB (LPDDR5X, single NUMA) |
| iGPU | Radeon 890M (unused; HIP+CUDA co-build rejected as out of scope) |

Serving config (baseline for all numbers): `-ngl 99 --n-cpu-moe 39 -fa on
-ub 2048 -t 12 --threads-batch 16 -c 32768`, MTP on for decode
(`--spec-type draft-mtp -fit off`, n-cpu-moe 41).

## Headline numbers (pp8192 unless noted)

| config | pp | tg (warm) | notes |
|---|---|---|---|
| PCIe streaming (default) | 147-161 bench / **212-233 server** | 14-15.4 (MTP) | the serving path |
| CPU experts (`GGML_CUDA_MMI_ON_CPU=1`) | 97-103 | 12.8 | tiled CPU path |
| iGPU (ROCm build, pre-strip) | ~74 equiv | ~6 | abandoned |
| MMQ / CUBLAS / no-mmap / pinning / poll sweeps | no effect | - | all within noise |

## Finding 1: `--n-cpu-moe` never ran anything on the CPU

**Root cause.** The override puts expert *weights* in host memory, but
`ggml_backend_cuda_device_supports_op()` accepts `MUL_MAT_ID` regardless of
where `src0` lives. The scheduler therefore assigns the op to the GPU, and
treats the host-resident weight tensor as a *split input*: **the full expert
tensor is copied CPU->GPU through a recycled VRAM scratch buffer before every
op, on every ubatch.**

**Evidence chain** (all reproducible):
- gdb breakpoint + `cudaPointerGetAttributes` probe: all 43 layers dispatch
  `MUL_MAT_ID` to CUDA with scheduler-clone names (`CUDA0#blk.N...#0`);
  consecutive layers share identical staging pointers (the recycled buffer).
- `nvidia-smi dmon -s t` during pp: PCIe RX p50 = 5.0 GB/s, p90 = 7.1 GB/s
  (the x4 Gen4 ceiling). Volume matches 81 GiB of expert weights per
  2048-token ubatch exactly (~40 MB/token).
- perf (user-space, main thread only): 88-92% of samples in `libcuda.so`,
  zero samples on the 12 compute threads; GPU utilization pegged 92-100%.

**Consequence.** pp is PCIe-bandwidth-bound at **~160-180 t/s bench ceiling**
(server amortizes to 212-233 on long prompts). This single mechanism explains
every anomaly we had measured: core-count invariance (CPU idle), ubatch
flatness above 2048 (per-token PCIe bytes constant), ncmoe insensitivity
(slope ~2%/layer), MMQ/CUBLAS irrelevance (GPU not the limiter).

**Fix (shipped, env-gated).** `ggml/src/ggml-cuda/ggml-cuda.cu` refuses
`MUL_MAT_ID` on CUDA when `src0` is in non-CUDA memory and
`GGML_CUDA_MMI_ON_CPU=1` is set; the op then executes on the CPU backend.
Default remains off because on x4 the staging path still beats the current
CPU tiled path (see Finding 2). Commit `2a6f795a1` on `glm5-min`.

## Finding 2: the CPU MoE path is real but ~2.2x slower than streaming

Isolated `MUL_MAT_ID` bench (one expert tensor, [4096x2048x288], 2048 tokens
x 10 used, 12 threads; `bench-moe-cpu/mmid.cpp`):

| path | ms/op | aggregate on weight bytes | GMAC/s |
|---|---|---|---|
| ggml-cpu tiled (`GGML_CPU_TILED_MM=1`) | 207 | 3.0 GB/s | 917 |
| stock vec_dot (`GGML_CPU_TILED_MM=0`) | 860 | 0.7 GB/s | 200 |
| GPU staging (in-situ equivalent) | - | 6.5-7 GB/s | - |

117 such ops per 2048-token ubatch x 207 ms = 24 s ≈ the observed 20 s
CPU-path time per ubatch (97-103 t/s).

**The stock vec_dot path collapses on RAM-streamed weights** (0.06 GB/s/core)
even though the identical kernel sustains 2.28 GB/s/core in an L2-resident
microbench: the 2 KB grid LUT + 1 KB sign LUT thrash out of L1/L2 when a
604 MB weight stream flows through, and the scalar-LUT dependency chains
cannot hide L2 latency. The tiled path wins because its dequant-unpack runs
once into a compact tile layout that the VNNI microkernel then reuses.

## Finding 3: the tiled pipeline's anchor is the microkernel's A-operand feed

perf breakdown of the CPU path (pp32768, 12 threads):

| component | share | note |
|---|---|---|
| `tiled_run_micro_vnni_8x16<32>` | 45% | the MAC engine |
| `tiled_unpack_src1_q8_K` + `repack_src1` | ~10% | **re-done for every weight-row group (32x redundancy)** |
| `tiled_unpack_src0` (all quant types) + `repack_src0` | ~15% | LUT dequant + interleave |
| `tiled_store_window_scatter` | 2.5% | |
| `quantize_row_q8_K_ref` | 2.3% | activations |
| gomp barriers | ~7% | |

Inside the 8x16 microtile (SUBBLK=32): per call, 512 `vpdpbusd` but also
**512 scalar `u32` loads + 512 broadcasts** of weight codes (`src0.q[...]`,
4 bytes at a time). 917 GMAC/s = ~76 GMAC/s/core ≈ 30% of the chip's
~6 TMAC/s VNNI peak: the kernel is load-port bound, not math bound.
Arithmetic: 512 scalar loads at 2-3/cycle ≈ 200-250 cycles, 512 dpbusd at
2/cycle = 256 cycles; with port contention the measured ~1000 cycles/call
matches.

Secondary structural cost: the pipeline moves ~4x the weight bytes through
L1/L2 scratch (unpack -> repack -> microkernel -> scatter), and re-unpacks
the activation tile once per weight-row group.

## Finding 4: AVX-512/VBMI2 LUT kernels do not pay on Zen5 (for this shape)

A from-scratch unpack prototype (`bench-moe-cpu/v2.h`): column-major grid
tables + `_mm512_permutex2var_epi8` (vpermi2b) lookups + `vpmultishiftqb`
sign-field extraction + GFNI sign application. **Fully correct**
(0 mismatches / 16384 rows vs the reference, three real bugs found on the
way: vpermi2b concat semantics, interleaved sign words in the qs layout,
and a table typo). Performance:

| variant | output GB/s (L2-resident, 1 core) |
|---|---|
| reference (AVX2, scalar LUT loads) | 4.7 |
| v1 (vpgatherdq entries + multishift signs) | 3.9 |
| v2 (column tables, 16 vpermi2b/64B) | 1.7 |

`vpermi2b` is multi-uop and slow-ish on Zen5; gathers pipeline poorly. The
LUT problem is not the dominant cost anyway (Finding 3).

## The complete causal chain (dense pp)

1. Model (95 GiB) >> VRAM (24 GiB) => 39/43 expert layers must live in host RAM.
2. Whichever engine computes those experts must move ~78 GiB/ubatch from RAM.
3. GPU engine: moves it over x4 PCIe (6.5-7 GB/s) => **152-212 t/s** (current).
4. CPU engine: tiled pipeline moves it from RAM at 3.0 GB/s => 97-103 t/s.
5. Ideal CPU engine: ~45 GB/s RAM + ~6 TMAC/s VNNI => **~500-550 t/s ceiling**.

So the CPU path has a ~5x theoretical headroom, gated entirely on pipeline
quality. That is the subject of [the next-gen pipeline plan](../ng-moe-pipeline/00-overview.md).

## Other findings (negative results, kept so we don't re-test)

- **mmap vs no-mmap:** no-mmap is worse (tg 10.8 vs ~14): 95 GiB of anon
  memory overshoots the 91 GiB RAM and pages; mmap lets the GPU-resident
  19 GiB stay out of the page cache equation.
- **Thread sweeps:** pp invariant across 4/8/12/16/24 threads on the
  streaming path (CPU idle). tg wants 12 threads (SMT-24 collapses it to 7).
  `--poll`, `--cpu-strict`, cpu-mask pinning (Zen5-only vs Zen5c-only vs
  mixed): all within noise for pp; decode prefers core count over type.
- **ubatch:** 2048 optimal; 4096/8192 flat-to-worse (per-token PCIe bytes
  constant; attention grows).
- **mlock:** impossible (`ulimit -l` = 8 MB; model > RAM anyway).
- **ISA:** `GGML_NATIVE=ON` already enables full Zen5 AVX-512/VNNI/VBMI.
  No Zen5-specific IQ2 kernels exist upstream (only Q4_0/Q8_0 repack).
- **Requant math:** every *larger* format (Q2_K 2.56 bpw) breaks the RAM
  wall (111 GiB experts > 91 GiB); every *smaller* one (IQ1_S) needs an
  imatrix we don't have. Parked per the Q2_XXS sweet-spot call.

## Artifact index

- `bench-moe-cpu/mmid.cpp` - isolated MUL_MAT_ID bench (tiled vs vec_dot)
- `bench-moe-cpu/unpack_bench.cpp`, `v2.h` - AVX-512 unpack prototypes + reference
- `bench-moe-cpu/order.cpp` - layout-order verifier (theory A/B)
- glm5-min commits: `2a6f795a1` (env-gated CPU execution of host-resident MoE),
  `f0099aafe` (MTP graph), `ea5aba6f7`/`2d7afadb1`/`581e8326b` (strip)
- raw perf data: `/tmp/opencode/perf-pp2.data` (streaming pp),
  `/tmp/opencode/perf-cpupath.data` (CPU path) - session-local, not committed
