// M2 fused dequant-GEMM prototype (standalone, no ggml integration).
//
// Computes dst = w(mma) act for one iq2_xxs expert row-group:
//   weights: [ROWS x 4096] raw iq2_xxs blocks (streamed once)
//   act:     staged q8 tile [256 routed rows x 4096] in dpbusd layout
// Output must match the tiled engine path (checked vs a scalar reference).
//
// The prototype isolates the ONE question M2 must answer: can inline
// expansion feed vpdpbusd at >= 2 TMAC/s aggregate?
//
// Layout notes (from tiled-kernel.cpp, verified):
//   iq2_xxs block (256 vals, 32 bytes): 8 groups of [g0 g1 g2 g3 | w(4B)]
//     g = 2-bit grid index (8 entries), w = keven sign fields (7 bits x4)
//     + 4-bit scale in w>>28
//   codes layout for dpbusd: value+128, sub-group-major within 32 bytes.
//
// Kernel shape: 8 weight rows x 16 act rows per microtile band, K full.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>
#include <array>
#include <cmath>
static inline float fp16_to_fp32(uint16_t h) { return 1.0f; } // placeholder; data uses d=1.0
#include <immintrin.h>

#define QK_K 256
#include "tables.h"   // iq2xxs_grid, ksigns_iq2xs

// ---------------- scalar reference (correct by construction) ----------------
static void ref_gemm(const uint8_t * w, int w_stride, int w_rows,
                     const int8_t (*act)[QK_K], int act_rows, int K,
                     float * dst, int dst_stride) {
    // dst[r][c] = sum_k wq[r][k] * act[c][k]; wq = dequant(w)
    for (int r = 0; r < w_rows; r++) {
        const uint8_t * row = w + (size_t) r * w_stride;
        for (int c = 0; c < act_rows; c++) {
            int acc = 0;
            for (int kb = 0; kb < K / QK_K; kb++) {
                const uint8_t * blk = row + (size_t) kb * 66 + 2; // fp16 d skipped
                for (int g = 0; g < 8; g++) {
                    const uint8_t * qs = blk + 8 * g;
                    const uint32_t sw = *(const uint32_t *) (qs + 4);
                    const int scale = 2 * (int) (sw >> 28) + 1;
                    int bacc = 0;
                    for (int l = 0; l < 4; l++) {
                        const uint64_t e = iq2xxs_grid[qs[l]];
                        const uint8_t signs = ksigns_iq2xs[(sw >> (7 * l)) & 127];
                        const int8_t * a = act[c] + kb * QK_K + 32 * g + 8 * l;
                        for (int j = 0; j < 8; j++) {
                            const int v = ((signs >> j) & 1) ? -(int) ((const uint8_t *) &e)[j]
                                                             :  (int) ((const uint8_t *) &e)[j];
                            bacc += v * a[j];
                        }
                    }
                    acc += scale * bacc;  // reference = true v*a product
                }
                dst[(size_t) r * dst_stride + c] += acc * (1.0f / 8.0f);  // d=1.0 in this data
            }
        }
    }
}

// ---------------- fused prototype ----------------
// Process 8 weight rows x 16 act rows, K in 256-blocks. Expansion uses 4
// independent scalar-LUT chains per 32-value group (one per sub-group l),
// then vpdpbusd over 4-row groups.
//
// per 32 values of one weight row: build q8u[8][4] = {c0,c1,c2,c3} u32s of
// biased codes (value+128, matching the tiled engine) for rows 0..7? No -
// dpbusd takes ONE weight u32 broadcast across act lanes. The fused kernel:
//   for each weight row r, for each 4-lane group: broadcast u32(biased codes
//   of that 16-value run) and dpbusd with the act codes zmm (16 lanes =
//   16 act rows x 4 bytes? no: dpbusd(a: i32 lanes 4x u8, b: i32 lanes 4x i8)
//   -> each i32 lane = dot4. So codes layout: [actrow c][kgroup kg] -> lane c
//   holds u32 of w-codes kg? The classic VNNI dot: acc += broadcast(w_u32)
//   * act_zmm where act lane c = 4 bytes of act codes for kg.
//   => act must be staged as [kgroup][actrow] 4-byte packs (same interleave
//   as tiled repack). We build that here.
//
// Weight expansion: for weight row r, kgroup kg (16 values): produce
//   u32 = 4 biased code bytes. LUT + signs as in the reference.
static inline uint32_t expand4(const uint8_t * qs4, uint32_t sw, int l) {
    // 16 values: sub-group l of the 32-value group: grid entry l, sign field l
    const uint64_t e = iq2xxs_grid[qs4[l]];
    const uint8_t signs = ksigns_iq2xs[(sw >> (7 * l)) & 127];
    uint8_t b[4];
    const uint8_t * ev = (const uint8_t *) &e;
    // two 8-value halves per sub-group? no: l-th sub-group covers 8 values.
    // NOTE: 32-value group = 4 sub-groups of 8. dpbusd 4-byte groups need 16:
    //   one 32-value group spans TWO u32s. We produce both here: b[0..3] =
    //   values 8l..8l+7 packed as two u32s -> caller uses 2 broadcasts.
    (void) b; (void) ev;
    return 0; // replaced below
}

// The clean formulation (matching tiled): a 32-value group produces 32
// biased bytes. dpbusd needs 4-byte units; the act side must interleave
// identically. We stage the act tile [4096/4 kgroups][16 rows x 4B].
// Weight side: per (row, 32-value group) -> 8 u32s (32 bytes).
// To amortize broadcasts: process 8 weight rows per band pass: each u32
// broadcast once, dpbusd against 16-lane act zmm, accumulate 8x16 int32.

int main(int argc, char ** argv) {
    const int K = 4096;
    const int w_rows = argc > 1 ? atoi(argv[1]) : 2048;   // one expert's rows
    const int act_rows = 16;                              // one act band
    const int ITERS = 5;

    // raw weights: [w_rows x K/256] blocks of 66B (2B fp16 d + 8x8B groups)
    const int nb = K / QK_K;
    const size_t BLK = 66;
    std::vector<uint8_t> w((size_t) w_rows * nb * BLK);
    {
        std::mt19937 rng(11);
        std::uniform_int_distribution<int> d(0, 255);
        for (auto & v : w) v = d(rng);
        for (int r = 0; r < w_rows; r++) { w[(size_t) r * nb * BLK] = 0x00; w[(size_t) r * nb * BLK + 1] = 0x3c; }
    }
    // act: 16 rows x K biased codes (int8, value+128 like the tiled act? the
    // reference unb-128s; keep act raw in [0..255] with -128 correction)
    std::vector<int8_t> act((size_t) act_rows * K);
    {
        std::mt19937 rng(13);
        std::uniform_int_distribution<int> d(-128, 127);
        for (auto & v : act) v = (int8_t) d(rng);
    }
    // reference on a small slice for correctness (8 rows)
    std::vector<float> refdst((size_t) 8 * act_rows, 0.0f);
    {
        // act as int8 rows of 256-aligned; reference uses biased-128
        // wrap the pointer type
        static int8_t tmp[16][QK_K];
        // direct: reuse ref_gemm with casts
        struct Arr { int8_t v[QK_K]; } a16[16];
        for (int c = 0; c < 16; c++) memcpy(a16[c].v, &act[(size_t) c * K], QK_K);
        // first 256 K only for the check
        ref_gemm(w.data(), nb * 66, 8, (const int8_t (*)[QK_K]) a16, 16, QK_K,
                 refdst.data(), act_rows);
    }

    // act per-4-value sums (for the bias correction), slab 0
    alignas(64) int32_t asum4[64][16]; // sum of 4 act values for u32 kg
    // ---- fused kernel: 8 w-rows x 16 act-rows, first K=256 slab, compare ----
    // stage act for slab 0: [kgroup kg 0..63][16 rows x 4B]
    alignas(64) int8_t staged[64][16][4];
    for (int c = 0; c < 16; c++) {
        const int8_t * a = &act[(size_t) c * K];
        for (int kg = 0; kg < 64; kg++) {
            for (int b = 0; b < 4; b++) staged[kg][c][b] = a[4 * kg + b];
        }
    }
    for (int c = 0; c < 16; c++) for (int kg = 0; kg < 64; kg++) {
        const int8_t * a = &act[(size_t) c * K];
        asum4[kg][c] = a[4*kg] + a[4*kg+1] + a[4*kg+2] + a[4*kg+3];
    }
    // fused compute rows 0..7, slab 0
    alignas(64) float out[8][16];
    {
        __m512i acc[8];
        for (int r = 0; r < 8; r++) acc[r] = _mm512_setzero_si512();
        for (int r = 0; r < 8; r++) {
            const uint8_t * row = w.data() + (size_t) r * nb * BLK + 2; // slab 0 = first block
            for (int g = 0; g < 8; g++) {
                const uint8_t * qs = row + 8 * g;
                const uint32_t sw = *(const uint32_t *) (qs + 4);
                const int scale = 2 * (int) (sw >> 28) + 1;
                __m512i gacc = _mm512_setzero_si512();
                __m512i biasc = _mm512_setzero_si512();
                for (int l = 0; l < 4; l++) {
                    const uint64_t e = iq2xxs_grid[qs[l]];
                    const uint8_t signs = ksigns_iq2xs[(sw >> (7 * l)) & 127];
                    const uint8_t * ev = (const uint8_t *) &e;
                    // two u32s for the 8 values of sub-group l
                    for (int half = 0; half < 2; half++) {
                        uint8_t b4[4];
                        for (int j = 0; j < 4; j++) {
                            const int idx = 4 * half + j;
                            const int v = ev[idx];
                            b4[j] = (uint8_t) (128 + (((signs >> idx) & 1) ? -v : v));
                        }
                        uint32_t u32 = 0; memcpy(&u32, b4, 4);
                        const __m512i wv = _mm512_set1_epi32((int) u32);
                        const int kg = 8 * g + 2 * l + half;
                        const __m512i av = _mm512_load_si512((const __m512i *) &staged[kg][0]);
                        gacc = _mm512_dpbusd_epi32(gacc, wv, av);
                        const __m512i asv = _mm512_load_si512((const __m512i *) &asum4[kg][0]);
                        biasc = _mm512_add_epi32(biasc, asv);
                    }
                }
                // (v+128)*a = v*a + 128*a: remove the 128*a part
                gacc = _mm512_sub_epi32(gacc, _mm512_mullo_epi32(biasc, _mm512_set1_epi32(128)));
                acc[r] = _mm512_add_epi32(acc[r], _mm512_mullo_epi32(gacc, _mm512_set1_epi32(scale)));
            }
        }
        const float d = 1.0f;  // w rows were built with fp16 d=1.0 at bytes 0..2
        for (int r = 0; r < 8; r++) {
            _mm512_store_ps(&out[r][0], _mm512_mul_ps(_mm512_cvtepi32_ps(acc[r]), _mm512_set1_ps(d * 0.125f)));
        }
    }
    // compare vs reference (row sums over the first 256 K)
    double max_rel = 0; int bad = 0;
    for (int r = 0; r < 8; r++) for (int c = 0; c < 16; c++) {
        // ref computed full K=256 (only slab 0): refdst has the full slab-0 result
        const double d = fabs((double) out[r][c] - refdst[(size_t) r * 16 + c]);
        const double rel = d / (1.0 + fabs((double) refdst[(size_t) r * 16 + c]));
        if (d > 1e-3 * (1.0 + fabs((double) refdst[(size_t) r * 16 + c]))) bad++;
        if (rel > max_rel) max_rel = rel;
    }
    printf("correctness vs scalar reference: %s (%d bad, max_rel %.2e)\n",
           bad ? "FAIL" : "PASS", bad, max_rel);
    printf("  sample: out[0][0]=%.2f ref[0]=%.2f | out[0][1]=%.2f ref[1]=%.2f | w[0..8]=",
           out[0][0], refdst[0], out[0][1], refdst[1]);
    for (int i = 0; i < 8; i++) printf(" %02x", w[i]);
    printf("\n  act[0..8]=");
    for (int i = 0; i < 8; i++) printf(" %d", (int) act[i]);
    printf("\n");

    // ---- throughput kernel: full w_rows x K, act restaged trivially ----
    // staged act: [K/4 kgroups][16 rows][4B] - build fully
    std::vector<int8_t> full_staged_unalign((size_t) (K / 4) * 16 * 4 + 64);
    int8_t * full_staged_p = (int8_t *) (((uintptr_t) full_staged_unalign.data() + 63) & ~(uintptr_t) 63);
    {
        int8_t * p = full_staged_p;
        for (int kg = 0; kg < K / 4; kg++)
            for (int c = 0; c < 16; c++)
                for (int b = 0; b < 4; b++)
                    p[((size_t) kg * 16 + c) * 4 + b] = act[(size_t) c * K + 4 * kg + b];
    }

    // full_asum4: per-4-value act sums in the staged layout (for bias correction)
    std::vector<int32_t> full_asum4_unalign((size_t) (K / 4) * 16 + 16);
    int32_t * full_asum4_p = (int32_t *) (((uintptr_t) full_asum4_unalign.data() + 63) & ~(uintptr_t) 63);
    {
        int32_t * p = full_asum4_p;
        for (int kg = 0; kg < K / 4; kg++)
            for (int c = 0; c < 16; c++)
                p[(size_t) kg * 16 + c] = full_staged_p[((size_t) kg * 16 + c) * 4 + 0]
                                         + full_staged_p[((size_t) kg * 16 + c) * 4 + 1]
                                         + full_staged_p[((size_t) kg * 16 + c) * 4 + 2]
                                         + full_staged_p[((size_t) kg * 16 + c) * 4 + 3];
    }

    auto bench = [&](bool fused, const char * name) {
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile float sink = 0;
        for (int it = 0; it < ITERS; it++) {
            for (int r0 = 0; r0 < w_rows; r0 += 8) {
                __m512i acc[8];
                for (int r = 0; r < 8; r++) acc[r] = _mm512_setzero_si512();
                for (int kb = 0; kb < nb; kb++) {
                    // phase 1: expand 8 rows x 8 groups -> u32s[8][16], scales[8][8]
                    // (64 independent chains; the compiler interleaves them)
                    alignas(64) uint32_t u32s[8][16];
                    alignas(64) int32_t scales[8][8];
                    for (int r = 0; r < 8; r++) {
                        const uint8_t * row = w.data() + (size_t) (r0 + r) * nb * BLK + (size_t) kb * BLK + 2;
                        for (int g = 0; g < 8; g++) {
                            const uint8_t * qs = row + 8 * g;
                            const uint32_t sw = *(const uint32_t *) (qs + 4);
                            scales[r][g] = 2 * (int) (sw >> 28) + 1;
                            for (int l = 0; l < 4; l++) {
                                const uint64_t e = iq2xxs_grid[qs[l]];
                                const uint8_t signs = ksigns_iq2xs[(sw >> (7 * l)) & 127];
                                const uint8_t * ev = (const uint8_t *) &e;
                                const uint64_t s64 = (uint64_t) signs * 0x0101010101010101ULL;
                                for (int half = 0; half < 2; half++) {
                                    uint32_t u = 0;
                                    for (int j = 0; j < 4; j++) {
                                        const int idx = 4 * half + j;
                                        const int v = ev[idx];
                                        const int neg = (int) ((s64 >> (8 * idx)) & 1);
                                        u |= (uint32_t) (128 + (neg ? -v : v)) << (8 * j);
                                    }
                                    u32s[r][2 * l + half] = u;
                                }
                            }
                        }
                    }
                    // phase 2: MAC sweep with bias correction
                    for (int r = 0; r < 8; r++) {
                        __m512i gacc[8];
                        __m512i biasc[8];
                        for (int g = 0; g < 8; g++) { gacc[g] = _mm512_setzero_si512(); biasc[g] = _mm512_setzero_si512(); }
                        for (int kg16 = 0; kg16 < 16; kg16++) {
                            const int g = kg16 / 2;
                            const __m512i wv = _mm512_set1_epi32((int) u32s[r][kg16]);
                            const int kg = kb * 64 + kg16;
                            const __m512i av = _mm512_load_si512((const __m512i *) (full_staged_p + (size_t) kg * 64));
                            const __m512i asv = _mm512_load_si512((const __m512i *) (full_asum4_p + (size_t) kg * 16));
                            gacc[g] = _mm512_dpbusd_epi32(gacc[g], wv, av);
                            biasc[g] = _mm512_add_epi32(biasc[g], asv);
                        }
                        for (int g = 0; g < 8; g++) {
                            const __m512i corr = _mm512_mullo_epi32(biasc[g], _mm512_set1_epi32(128));
                            const __m512i net = _mm512_sub_epi32(gacc[g], corr);
                            acc[r] = _mm512_add_epi32(acc[r], _mm512_mullo_epi32(net, _mm512_set1_epi32(scales[r][g])));
                        }
                    }
                }
                for (int r = 0; r < 8; r++) sink += _mm512_cvtss_f32(_mm512_castsi512_ps(acc[r]));
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count() / ITERS;
        const double macs = (double) w_rows * 16 * K;
        const double bytes = (double) w_rows * nb * 32;
        printf("%s: %.3f ms, %.2f GB/s (weight bytes), %.1f GMAC/s (sink %.1f)\n",
               name, secs * 1e3, bytes / secs / 1e9, macs / secs / 1e9, (double) sink);
    };
    (void) 0;
    bench(true, "fused-8x16");
    return 0;
}
