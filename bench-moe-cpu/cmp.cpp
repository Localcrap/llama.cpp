#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <cmath>
#define QK_K 256
#include "tables.h"
int main() {
    // one weight row, one act row, K=256 (one block). Compare scalar vs the fused math.
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> bd(0,255), ad(-128,127);
    uint8_t blk[32]; for (auto & v : blk) v = bd(rng); blk[0]=0x00; blk[1]=0x3c;
    int8_t a[256]; for (auto & v : a) v = (int8_t) ad(rng);
    // scalar reference (v*a)
    int acc_ref = 0;
    for (int g = 0; g < 8; g++) {
        const uint8_t * qs = blk + 4*g;
        const uint32_t sw = *(uint32_t*)(qs+4);
        const int scale = 2*(int)(sw>>28)+1;
        int bacc = 0;
        for (int l = 0; l < 4; l++) {
            const uint64_t e = iq2xxs_grid[qs[l]];
            const uint8_t signs = ksigns_iq2xs[(sw >> (7*l)) & 127];
            for (int j = 0; j < 8; j++) {
                const int v = ((signs>>j)&1) ? -(int)((uint8_t*)&e)[j] : (int)((uint8_t*)&e)[j];
                bacc += v * a[32*g + 8*l + j];
            }
        }
        acc_ref += scale * bacc;
    }
    // fused: dpbusd lanes per 4-value group; simulate scalar
    int acc_fused = 0;
    for (int g = 0; g < 8; g++) {
        const uint8_t * qs = blk + 4*g;
        const uint32_t sw = *(uint32_t*)(qs+4);
        const int scale = 2*(int)(sw>>28)+1;
        int gacc = 0, biasc = 0;
        for (int l = 0; l < 4; l++) {
            const uint64_t e = iq2xxs_grid[qs[l]];
            const uint8_t signs = ksigns_iq2xs[(sw >> (7*l)) & 127];
            for (int half = 0; half < 2; half++) {
                for (int j = 0; j < 4; j++) {
                    const int idx = 4*half + j;
                    const int v = ((signs>>idx)&1) ? -(int)((uint8_t*)&e)[idx] : (int)((uint8_t*)&e)[idx];
                    const int wq = 128 + v;
                    // act lane value for this u32 position: a[32g+8l+?]
                    // fused kg mapping: u32 covers values 8*l + 4*half + j
                    const int av = a[32*g + 8*l + 4*half + j];
                    gacc += wq * av;
                    biasc += av;
                }
            }
        }
        gacc -= 128 * biasc;
        acc_fused += scale * gacc;
    }
    printf("ref=%d fused=%d %s\n", acc_ref, acc_fused, acc_ref==acc_fused ? "MATCH" : "MISMATCH");
    return 0;
}
