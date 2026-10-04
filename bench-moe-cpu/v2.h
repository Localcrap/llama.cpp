// V2: column-table vpermi2b unpack (no gathers, sequential input loads)
#pragma once
#include <immintrin.h>
#include "tables.h"

static uint8_t Tcol[8][256];  // Tcol[c][g] = byte c of iq2xxs_grid[g]
static bool Tcol_init = false;
static void init_Tcol() {
    for (int g = 0; g < 256; g++) {
        const uint8_t * e = (const uint8_t *) &iq2xxs_grid[g];
        for (int c = 0; c < 8; c++) Tcol[c][g] = e[c];
    }
    Tcol_init = true;
}

alignas(64) static const uint8_t v2_div8[64] = {
    // qs half = [g0 g1 g2 g3 | w0(4B)] [g4 g5 g6 g7 | w1(4B)] - words interleave the index groups
    0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,
    8,8,8,8,8,8,8,8, 9,9,9,9,9,9,9,9, 10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11 };
alignas(64) static const uint8_t v2_selv[64] = {
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128 };
alignas(64) static const uint8_t v2_ms_shifts[64] = {
     0, 0, 0, 0, 0, 0, 0, 0,   7, 7, 7, 7, 7, 7, 7, 7,  14,14,14,14,14,14,14,14,  21,21,21,21,21,21,21,21,
    32,32,32,32,32,32,32,32,  39,39,39,39,39,39,39,39,  46,46,46,46,46,46,46,46,  53,53,53,53,53,53,53,53 };
// column masks: lanes p with p%8 == c (0x0101...01 pattern shifted by c)
static const __mmask64 v2_cmask_v[8] = {
    0x0101010101010101ull, 0x0202020202020202ull, 0x0404040404040404ull, 0x0808080808080808ull,
    0x1010101010101010ull, 0x2020202020202020ull, 0x4040404040404040ull, 0x8080808080808080ull };

static void unpack_v2_row(const uint8_t * qs_row, uint8_t * out256, int32_t * scales8) {
    const __m512i div8 = _mm512_load_si512((const void *) v2_div8);
    const __m512i selv = _mm512_load_si512((const void *) v2_selv);
    const __m512i shifts = _mm512_load_si512((const void *) v2_ms_shifts);
    const __m512i ktab_lo = _mm512_loadu_si512((const void *) (ksigns_iq2xs + 0));
    const __m512i ktab_hi = _mm512_loadu_si512((const void *) (ksigns_iq2xs + 64));
    const __m512i mask7f = _mm512_set1_epi8(0x7f);
    const __m512i bias = _mm512_set1_epi8((int8_t) 128);
    __m512i tlo[8], thi[8];
    for (int c = 0; c < 8; c++) {
        tlo[c] = _mm512_loadu_si512((const void *) (Tcol[c] + 0));
        thi[c] = _mm512_loadu_si512((const void *) (Tcol[c] + 64));
    }
    static bool once = [](){ init_Tcol(); return true; }();
    (void) once;
    __m512i t23[8], t23b[8];
    for (int c = 0; c < 8; c++) {
        t23[c]  = _mm512_loadu_si512((const void *) (Tcol[c] + 128));
        t23b[c] = _mm512_loadu_si512((const void *) (Tcol[c] + 192));
    }

    for (int h = 0; h < 4; h++) {
        const uint8_t * qs = qs_row + 16 * h;
        // grid index per lane: broadcast the 16-byte half and select qs[p/8]
        const __m512i half = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) qs));
        const __m512i idx_g = _mm512_permutexvar_epi8(div8, half);
        // 8 masked column lookups: Tcol[c] is 256 bytes = two 128-byte vpermi2b concats;
        // index bit 7 picks the concat, bits 6:0 the byte within it
        const __mmask64 hi_sel = _mm512_test_epi8_mask(idx_g, _mm512_set1_epi8((int8_t) 0x80));
        const __m512i idx7 = _mm512_and_si512(idx_g, _mm512_set1_epi8(0x7f));
        __m512i mag = _mm512_setzero_si512();
        for (int c = 0; c < 8; c++) {
            const __mmask64 col = v2_cmask_v[c];
            mag = _mm512_or_si512(mag,
                _mm512_or_si512(
                    _mm512_maskz_permutex2var_epi8(col & ~hi_sel, tlo[c], idx7, thi[c]),
                    _mm512_maskz_permutex2var_epi8(col &  hi_sel, t23[c], idx7, t23b[c])));
        }
        // sign fields
        const uint64_t w = (uint64_t) (*(const uint32_t *) (qs + 4)) | ((uint64_t) (*(const uint32_t *) (qs + 12)) << 32);
        const __m512i sidx = _mm512_and_si512(
            _mm512_multishift_epi64_epi8(shifts, _mm512_set1_epi64((long long int) w)), mask7f);
        const __m512i sv = _mm512_permutex2var_epi8(ktab_lo, sidx, ktab_hi);
        const __mmask64 smask = _mm512_testn_epi8_mask(
            _mm512_xor_si512(_mm512_and_si512(sv, selv), selv), _mm512_set1_epi8((int8_t) 0xff));
        const __m512i neg = _mm512_mask_sub_epi8(mag, smask, _mm512_set1_epi8(0), mag);
        _mm512_storeu_si512((void *) (out256 + 64 * h), _mm512_add_epi8(neg, bias));
        // scales
        scales8[2*h+0] = (int32_t) (2 * ((uint32_t) w >> 28) + 1);
        scales8[2*h+1] = (int32_t) (2 * ((uint32_t) (w >> 32) >> 28) + 1);
    }
}
