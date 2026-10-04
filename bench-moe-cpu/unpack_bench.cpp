// AVX-512/VBMI2 IQ2_XXS unpack prototype: reference vs candidates, correctness + GB/s.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <immintrin.h>

#include "tables.h"
#include "v2.h"

// ---------------- reference (current tiled.cpp implementation) ----------------
static void unpack_ref_row(const uint8_t * qs_row, uint8_t * out256, int32_t * scales8) {
    for (int ib32 = 0; ib32 < 8; ib32++) {
        uint32_t aux32[2];
        const uint8_t * aux8 = (const uint8_t *) aux32;
        memcpy(aux32, qs_row + 8 * ib32, 2 * sizeof(uint32_t));
        scales8[ib32] = (int32_t) (2 * (aux32[1] >> 28) + 1);
        uint64_t g[4];
        uint8_t signs4[4];
        for (int l = 0; l < 4; l++) {
            g[l] = iq2xxs_grid[aux8[l]];
            signs4[l] = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
        }
        const __m256i v = _mm256_set_epi64x((int64_t) g[3], (int64_t) g[2], (int64_t) g[1], (int64_t) g[0]);
        const __m256i sv = _mm256_shuffle_epi8(
            _mm256_set1_epi32((int32_t) (signs4[0] | signs4[1] << 8 | signs4[2] << 16 | signs4[3] << 24)),
            _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3));
        const __m256i sel = _mm256_set1_epi64x((int64_t) 0x8040201008040201ULL);
        const __m256i mask = _mm256_gf2p8affine_epi64_epi8(sel, sv, 0);
        const __m256i sgn = _mm256_sub_epi8(_mm256_xor_si256(v, mask), mask);
        _mm256_storeu_si256((__m256i *) (out256 + 32 * ib32),
                            _mm256_add_epi8(sgn, _mm256_set1_epi8((int8_t) 128)));
    }
}

// ---------------- candidate V1: gather the 8-byte grid entries ----------------
// per 64 output bytes: one 8-lane 64-bit gather for magnitudes,
// vpmultishiftqb for the 7-bit sign fields, one vpermi2b (128B table) for sign bytes.
static inline __m512i lut128(const uint8_t * tab128, const __m512i idx) {
    // 128-byte table = two 64-byte halves; index bit 6 selects the half
    const __m512i lo = _mm512_loadu_si512((const void *) (tab128 + 0));
    const __m512i hi = _mm512_loadu_si512((const void *) (tab128 + 64));
    return _mm512_permutex2var_epi8(lo, idx, hi);
}

// shifts for vpmultishiftqb: byte i reads qword i/8 at bit offset shifts[i] (mod 56).
// we lay out the two sign words of groups j, j+1 in one qword: w_lo (bits 0..31), w_hi (bits 32..63).
// lane p (0..63) belongs to group j + (p>=32), sub-group l = (p/8)%4:
//   qword for lane p = p/8... but we need per-lane qword = p/16 (each word covers 16 lanes? no - 32 lanes/group? )
// NOTE: 64 output bytes cover TWO 32-byte groups: group A lanes 0..31, group B lanes 32..63.
//   qword content {wA, wB}: lane p uses word (p>=32), field (p/8)%4 => bit offset = 32*(p>=32) + 7*((p/8)%4) (max 53 < 56 ok)
alignas(64) static const uint8_t selv8[64] = {
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128 };
alignas(64) static const uint8_t ms_shifts[64] = {
     0, 0, 0, 0, 0, 0, 0, 0,   7, 7, 7, 7, 7, 7, 7, 7,  14,14,14,14,14,14,14,14,  21,21,21,21,21,21,21,21,
    32,32,32,32,32,32,32,32,  39,39,39,39,39,39,39,39,  46,46,46,46,46,46,46,46,  53,53,53,53,53,53,53,53 };

static void unpack_v1_row(const uint8_t * qs_row, uint8_t * out256, int32_t * scales8) {
    __m512i mag[4], sidx[4], sv[4], selv, kres[4];
    uint64_t w[4];
    selv = _mm512_load_si512((const void *) selv8);
    const __m512i shifts = _mm512_load_si512((const void *) ms_shifts);
    const __m512i ktab_lo = _mm512_loadu_si512((const void *) (ksigns_iq2xs + 0));
    const __m512i ktab_hi = _mm512_loadu_si512((const void *) (ksigns_iq2xs + 64));
    const __m512i mask7f = _mm512_set1_epi8(0x7f);

    // stage 1: gathers + index extraction (independent across halves)
    for (int h = 0; h < 4; h++) {
        const uint8_t * qs = qs_row + 16 * h;
        const __m256i gidx8 = _mm256_setr_epi32(qs[0], qs[1], qs[2], qs[3], qs[8], qs[9], qs[10], qs[11]);
        mag[h] = _mm512_i32gather_epi64(gidx8, (const long long int *) iq2xxs_grid, 8);
        w[h] = (uint64_t) (*(const uint32_t *) (qs + 4)) | ((uint64_t) (*(const uint32_t *) (qs + 12)) << 32);
        sidx[h] = _mm512_and_si512(_mm512_multishift_epi64_epi8(shifts, _mm512_set1_epi64((long long int) w[h])), mask7f);
        sv[h] = _mm512_permutex2var_epi8(ktab_lo, sidx[h], ktab_hi);
        kres[h] = _mm512_xor_si512(_mm512_and_si512(sv[h], selv), selv);
    }
    // stage 2: sign masks + apply
    const __m512i bias = _mm512_set1_epi8((int8_t) 128);
    for (int h = 0; h < 4; h++) {
        const __mmask64 smask = _mm512_testn_epi8_mask(kres[h], _mm512_set1_epi8((int8_t) 0xff));
        const __m512i neg = _mm512_mask_sub_epi8(mag[h], smask, _mm512_set1_epi8(0), mag[h]);
        _mm512_storeu_si512((void *) (out256 + 64 * h), _mm512_add_epi8(neg, bias));
    }
    for (int h = 0; h < 4; h++) {
        scales8[2*h+0] = (int32_t) (2 * ((uint32_t) w[h] >> 28) + 1);
        scales8[2*h+1] = (int32_t) (2 * ((uint32_t) (w[h] >> 32) >> 28) + 1);
    }
}

int main(int argc, char ** argv) {
    const int n_rows = argc > 1 ? atoi(argv[1]) : 4096;   // rows in the working set
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> b(0, 255);
    std::vector<std::vector<uint8_t>> qs(n_rows, std::vector<uint8_t>(64));
    std::vector<std::vector<uint8_t>> out_ref(n_rows, std::vector<uint8_t>(256));
    std::vector<std::vector<uint8_t>> out_v1(n_rows, std::vector<uint8_t>(256));
    std::vector<std::vector<int32_t>> sc_ref(n_rows, std::vector<int32_t>(8));
    std::vector<std::vector<int32_t>> sc_v1(n_rows, std::vector<int32_t>(8));
    for (auto & q : qs) for (auto & v : q) v = b(rng);

    for (int r = 0; r < n_rows; r++) unpack_ref_row(qs[r].data(), out_ref[r].data(), sc_ref[r].data());
    for (int r = 0; r < n_rows; r++) unpack_v1_row(qs[r].data(), out_v1[r].data(), sc_v1[r].data());
    init_Tcol();
    for (int r = 0; r < n_rows; r++) unpack_v2_row(qs[r].data(), out_v1[r].data(), sc_v1[r].data());
    int bad = 0, bads = 0;
    for (int r = 0; r < n_rows && bad < 8; r++) {
        if (memcmp(out_ref[r].data(), out_v1[r].data(), 256)) {
            if (!bad) { printf("first mismatch row %d:\n  ref:", r);
                for (int i = 0; i < 32; i++) printf(" %02x", out_ref[r][i]);
                printf("\n  v1 :", r);
                for (int i = 0; i < 32; i++) printf(" %02x", out_v1[r][i]);
                printf("\n"); }
            bad++;
        }
        for (int i = 0; i < 8; i++) if (sc_ref[r][i] != sc_v1[r][i]) bads++;
    }
    printf("correctness: %d byte mismatches, %d scale mismatches over %d rows\n", bad, bads, n_rows);
    // decompose the first mismatching lanes (v2 output lives in out_v1)
    int found = 0;
    for (int r = 0; r < n_rows && found < 8; r++) {
        const uint8_t * q = qs[r].data();
        for (int p = 0; p < 256 && found < 8; p++) {
            if (out_ref[r][p] != out_v1[r][p]) {
                const int h = p / 64, rem = p % 64;
                const int j = rem / 32, b = rem % 8;
                const int l = (rem % 32) / 8;
                const int g = q[16 * h + 8 * j + l];
                const uint32_t w = *(const uint32_t *) (q + 16 * h + 4 + 8 * j);
                const int si = (w >> (7 * l)) & 127;
                const int mag = Tcol[b][g];
                const int sbit = (ksigns_iq2xs[si] >> b) & 1;
                printf("row %d lane %3d: ref %02x got %02x | half %d grp %d sub %d bit %d | g=%d T=%d sidx=%d sbyte=%02x sbit=%d => want %02x\n",
                       r, p, out_ref[r][p], out_v1[r][p], h, j, l, b, g, mag, si, ksigns_iq2xs[si], sbit,
                       (uint8_t) (128 + (sbit ? -mag : mag)));
                if (++found == 8) goto next;
            }
        }
    }
    next:;

    // timing (streaming: 64-byte inputs random order to mimic expert rows)
    auto bench = [&](auto fn, const char * name) {
        auto t0 = std::chrono::high_resolution_clock::now();
        constexpr int ITERS = 20;
        uint64_t acc = 0;
        for (int it = 0; it < ITERS; it++) {
            for (int r = 0; r < n_rows; r++) {
                fn(qs[r].data(), out_v1[r].data(), sc_v1[r].data());
                acc += out_v1[r][0];
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        printf("%s: %.3f s (%llu) - output %.2f MB -> %.2f GB/s, input %.2f GB/s\n",
               name, secs, (unsigned long long) acc,
               (double) ITERS * n_rows * 256 / secs / 1e9,
               (double) ITERS * n_rows * 64 / secs / 1e9);
    };
    bench(unpack_ref_row, "ref ");
    bench(unpack_v1_row, "v1  ");
    bench(unpack_v2_row, "v2  ");
    return 0;
}
