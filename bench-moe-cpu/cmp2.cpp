#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#define QK_K 256
#include "tables.h"
// compare byte-order of expand: fused b4[j] uses idx = 4*half+j over entry e (8 values)
// staged act a[4*kg+j]; kg = 8g+2l+half; so fused w u32 lane j pairs with a[32g+8l+4half+j]
// and w value idx = 4*half+j must equal the value at position 32g+8l+4half+j in the ref.
// ref: value j-th of entry l is at 32g+8l+j. => ref position 32g+8l+4half+j corresponds to
// ref value index (4half+j) of entry l. fused uses ev[4*half+j] - SAME. OK.
// then the only remaining difference: the fused CORRECTNESS kernel (not the sim) uses
// staged[kg] loaded as __m512i from &staged[kg][0] - 16 rows x 4B = 64B - lane c = row c.
// And biasc loads asum4[kg][0] the same way. So simulate EXACTLY the vector code scalar-wise:
int main() {
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> bd(0,255), ad(-128,127);
    uint8_t wrow[32]; for (auto & v : wrow) v = bd(rng); wrow[0]=0x00; wrow[1]=0x3c;
    int8_t a[256]; for (auto & v : a) v = (int8_t) ad(rng);
    // fused vector semantics per lane c:
    int lane_ref[16], lane_fused[16];
    for (int c = 0; c < 16; c++) {
        int acc = 0;
        for (int g = 0; g < 8; g++) {
            const uint8_t * qs = wrow + 4*g;
            const uint32_t sw = *(uint32_t*)(qs+4);
            const int scale = 2*(int)(sw>>28)+1;
            int gacc = 0, biasc = 0;
            for (int l = 0; l < 4; l++) {
                const uint64_t e = iq2xxs_grid[qs[l]];
                const uint8_t signs = ksigns_iq2xs[(sw >> (7*l)) & 127];
                for (int half = 0; half < 2; half++) {
                    uint8_t b4[4];
                    for (int j = 0; j < 4; j++) {
                        const int idx = 4*half+j;
                        const int v = ((signs>>idx)&1) ? -(int)((uint8_t*)&e)[idx] : (int)((uint8_t*)&e)[idx];
                        b4[j] = (uint8_t)(128+v);
                    }
                    uint32_t u32 = 0; memcpy(&u32, b4, 4);
                    const int kg = 8*g + 2*l + half;
                    for (int j = 0; j < 4; j++) {
                        const int av = a[4*kg+j];  // as staged
                        const int wq = (int)((uint8_t*)&u32)[j];
                        gacc += wq * av;
                        biasc += av;
                    }
                }
            }
            gacc -= 128*biasc;
            acc += scale*gacc;
        }
        lane_fused[c] = acc; lane_ref[c] = 0; // ref for lane c (same a): compute true
        for (int g = 0; g < 8; g++) {
            const uint8_t * qs = wrow + 4*g;
            const uint32_t sw = *(uint32_t*)(qs+4);
            const int scale = 2*(int)(sw>>28)+1;
            int bacc = 0;
            for (int l = 0; l < 4; l++) {
                const uint64_t e = iq2xxs_grid[qs[l]];
                const uint8_t signs = ksigns_iq2xs[(sw >> (7*l)) & 127];
                for (int j = 0; j < 8; j++) {
                    const int v = ((signs>>j)&1) ? -(int)((uint8_t*)&e)[j] : (int)((uint8_t*)&e)[j];
                    bacc += v * a[32*g+8*l+j];
                }
            }
            lane_ref[c] += scale*bacc;
        }
    }
    int bad = 0; for (int c = 0; c < 16; c++) if (lane_ref[c] != lane_fused[c]) bad++;
    printf("per-lane: %d/16 mismatch (ref[0]=%d fused[0]=%d)\n", bad, lane_ref[0], lane_fused[0]);
    return 0;
}
