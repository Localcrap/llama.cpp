// IQ2_XXS vec_dot microbench: reference vs AVX-512 candidates
// Links against libggml-cpu for the reference kernel + tables.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <random>

// from ggml
extern "C" {
#include "ggml.h"
typedef struct { float d; uint16_t qs[16]; } block_iq2_xxs; // QK_K=256, 32B qs per block => 16 uint16
typedef struct { float d; int8_t qs[256]; } block_q8_K;
void ggml_vec_dot_iq2_xxs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
}

static void make_weights(block_iq2_xxs * x, block_q8_K * y, int nb, std::mt19937 & rng) {
    std::uniform_int_distribution<int> b256(0,255);
    std::uniform_int_distribution<int> b16(0,15);
    std::uniform_int_distribution<int> q8(-128,127);
    for (int i = 0; i < nb; ++i) {
        x[i].d = 0.01f * (1 + (i % 13));
        // pack like the quantizer: 4 groups of (4 grid bytes | sign/scale word)
        for (int g = 0; g < 4; ++g) {
            uint8_t grid_bytes[4];
            for (int k = 0; k < 4; ++k) grid_bytes[k] = b256(rng);
            uint32_t signword = b256(rng) | (b256(rng)<<8) | (b256(rng)<<16) | ((uint32_t)b16(rng)<<28);
            uint8_t * dst = (uint8_t *) x[i].qs + g*8;
            memcpy(dst, grid_bytes, 4);
            memcpy(dst+4, &signword, 4);
        }
        for (int k = 0; k < 256; ++k) y[i].qs[k] = q8(rng);
        y[i].d = 0.5f;
    }
}

int main(int argc, char ** argv) {
    const int nb = 512;                // blocks per row
    const int n  = nb * 256;
    const int n_rows = 64;              // independent rows to defeat cache reuse partially
    std::mt19937 rng(42);

    std::vector<std::vector<block_iq2_xxs>> X(n_rows, std::vector<block_iq2_xxs>(nb));
    std::vector<std::vector<block_q8_K>>   Y(n_rows, std::vector<block_q8_K>(nb));
    for (int r = 0; r < n_rows; ++r) make_weights(X[r].data(), Y[r].data(), nb, rng);

    // reference timing
    auto t0 = std::chrono::high_resolution_clock::now();
    float acc = 0;
    constexpr int ITERS = 200;
    for (int it = 0; it < ITERS; ++it) {
        for (int r = 0; r < n_rows; ++r) {
            float s = 0;
            ggml_vec_dot_iq2_xxs_q8_K(n, &s, 0, X[r].data(), 0, Y[r].data(), 0, 1);
            acc += s;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1-t0).count();
    double bytes = (double) ITERS * n_rows * nb * 32.0; // 32B of iq2 data per 256 values
    printf("reference: %.3f s, acc=%.3f, iq2-bytes/s = %.2f MB/s (values/s = %.1fM)\n",
           secs, acc, bytes/secs/1e6, (double) ITERS*n_rows*n/secs/1e6);
    return 0;
}
