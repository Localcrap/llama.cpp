#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include "tables.h"
int main() {
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> d(0,255);
    uint8_t qs[64]; for (auto & v : qs) v = d(rng);
    // ref row 0 first 32 bytes (from bench output):
    const uint8_t ref[32] = {0x88,0x78,0xab,0x67,0x88,0x88,0x78,0x67, 0xab,0x78,0x78,0x55,0x78,0x78,0x88,0x78,
                             0x88,0x99,0x88,0x78,0x99,0x88,0x99,0x67, 0x88,0x99,0x78,0x78,0x99,0x88,0x88,0xab};
    // theory A (natural): chunk k = grid[qs[8*ib+k]], sign field k
    // theory B (reversed): chunk k = grid[qs[8*ib+3-k]], sign field 3-k
    for (int th = 0; th < 2; th++) {
        uint8_t out[32]; int bad = 0;
        for (int p = 0; p < 32; p++) {
            const int ib = p/32, c = (p%32)/8, b = p%8;
            const int l = th ? 3-c : c;
            const uint32_t w = *(uint32_t*)(qs + 8*ib + 4);
            const uint8_t mag = ((uint8_t*)&iq2xxs_grid[qs[8*ib+l]])[b];
            const uint8_t sb = ksigns_iq2xs[(w >> (7*l)) & 127];
            out[p] = 128 + (((sb >> b) & 1) ? -mag : mag);
        }
        for (int p = 0; p < 32; p++) if (out[p] != ref[p]) bad++;
        printf("theory %s: %d/32 mismatches\n", th ? "B(rev)" : "A(nat)", bad);
    }
    printf("qs: "); for (int i = 0; i < 16; i++) printf("%02x ", qs[i]); printf("\n");
    return 0;
}
