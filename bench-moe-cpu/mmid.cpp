// Isolated MUL_MAT_ID bench: iq2_xxs experts [4096, 2048, 288] x N tokens x 10 used
// Measures the ggml-cpu tiled path in isolation (no scheduler, no GPU).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <cstdlib>

extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
}

int main(int argc, char ** argv) {
    const int64_t K = 4096, N = 2048, E = 288;
    const int64_t n_tok = argc > 1 ? atoi(argv[1]) : 2048;
    const int64_t n_used = 10;
    const int n_threads = argc > 2 ? atoi(argv[2]) : 12;

    struct ggml_init_params ip = { (size_t)4<<30, NULL, false }; // 0 = size the ctx automatically
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t ne0[3] = {K, N, E};
    const int64_t ne1[3] = {K, n_used, n_tok};
    const int64_t ne2[2] = {n_used, n_tok};
    struct ggml_tensor * src0 = ggml_new_tensor(ctx, GGML_TYPE_IQ2_XXS, 3, ne0);
    struct ggml_tensor * src1 = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne1);
    struct ggml_tensor * ids  = ggml_new_tensor(ctx, GGML_TYPE_I32, 2, ne2);
    struct ggml_tensor * dst  = ggml_mul_mat_id(ctx, src0, src1, ids);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> row(K);

    // hand-craft valid-shape iq2_xxs blocks (random codes; perf-equivalent)
    const size_t rs = ggml_row_size(GGML_TYPE_IQ2_XXS, K);
    std::vector<uint8_t> qrow(rs);
    {
        uint16_t * p = (uint16_t *) qrow.data();
        const int nwords = (int) (rs / 2);
        std::uniform_int_distribution<int> wd(0, 65535);
        p[0] = 0x3c00; // d = 1.0fp16
        for (int i = 1; i < nwords; ++i) p[i] = wd(rng);
    }
    uint8_t * base = (uint8_t *) src0->data;
    for (int64_t i = 0; i < N*E; ++i) memcpy(base + i*rs, qrow.data(), rs);

    for (auto & v : row) v = dist(rng);
    float * s1 = (float *) src1->data;
    for (int i = 0; i < K*n_tok; ++i) s1[i] = dist(rng);

    std::uniform_int_distribution<int> ed(0, (int)E-1);
    int32_t * id = (int32_t *) ids->data;
    for (int i = 0; i < n_used*n_tok; ++i) id[i] = ed(rng);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);

    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, NULL);
    std::vector<uint8_t> work(plan.work_size + 16*1024*1024);
    plan.work_data = work.data();
    plan.n_threads = n_threads;

    ggml_graph_compute(graph, &plan); // warmup
    auto t0 = std::chrono::high_resolution_clock::now();
    const int ITERS = 5;
    for (int i = 0; i < ITERS; ++i) ggml_graph_compute(graph, &plan);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1-t0).count() / ITERS;

    double macs = (double) n_tok * n_used * N * K;
    double expert_bytes = (double) N * E * rs;
    printf("MUL_MAT_ID: %lld tok x %lld used x [%lld x %lld x %lld], %d threads\n",
           (long long)n_tok, (long long)n_used, (long long)K, (long long)N, (long long)E, n_threads);
    printf("  %.1f ms/call, %.1f GMAC/s, experts-touched %.2f GB -> %.1f GB/s effective\n",
           secs*1e3, macs/secs/1e9, expert_bytes/1e9, expert_bytes/secs/1e9);
    printf("  dst[0..3] = %.3f %.3f %.3f %.3f\n",
           ((float*)dst->data)[0], ((float*)dst->data)[1], ((float*)dst->data)[2], ((float*)dst->data)[3]);
    return 0;
}
