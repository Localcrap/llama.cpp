// Isolated MUL_MAT_ID bench + golden checker (NG MoE pipeline, M0 harness).
//
// Modes:
//   time               benchmark the current engine (default)
//   ref-dump  FILE     write dst floats to FILE (run with GGML_CPU_TILED_MM=0 for the reference engine)
//   check     FILE     run the op, compare dst against FILE (tolerance 1e-4 rel), then time it
//   selftest  FILE     like check, but perturbs dst first - must FAIL (harness self-test)
//
// Usage: mmid [up|gate|down] [iq2_xxs|iq3_xxs|iq2_s|iq4_xs] [tokens] [threads] [mode] [file]
// Seeds are fixed; every run with the same args produces identical tensors.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <cstdlib>
#include <cmath>

extern "C" {
#include "ggml.h"
#include "ggml-cpu.h"
}

struct shape_t { const char * name; int64_t K, N; };

// (K, N) for GLM-5.3-Flash expert tensors; E = 288, n_used = 10
static const shape_t SHAPES[] = {
    { "up",   4096, 2048 },
    { "gate", 4096, 2048 },
    { "down", 2048, 4096 },
};

int main(int argc, char ** argv) {
    const char * shape_name = argc > 1 ? argv[1] : "up";
    const char * type_name  = argc > 2 ? argv[2] : "iq2_xxs";
    const int64_t n_tok     = argc > 3 ? atoll(argv[3]) : 2048;
    const int n_threads     = argc > 4 ? atoi(argv[4]) : 12;
    const char * mode       = argc > 5 ? argv[5] : "time";
    const char * golden     = argc > 6 ? argv[6] : nullptr;

    const shape_t * S = nullptr;
    for (auto & s : SHAPES) if (!strcmp(s.name, shape_name)) S = &s;
    if (!S) { fprintf(stderr, "bad shape %s\n", shape_name); return 2; }

    enum ggml_type t;
    if      (!strcmp(type_name, "iq2_xxs")) t = GGML_TYPE_IQ2_XXS;
    else if (!strcmp(type_name, "iq3_xxs")) t = GGML_TYPE_IQ3_XXS;
    else if (!strcmp(type_name, "iq2_s"))   t = GGML_TYPE_IQ2_S;
    else if (!strcmp(type_name, "iq4_xs"))  t = GGML_TYPE_IQ4_XS;
    else { fprintf(stderr, "bad type %s\n", type_name); return 2; }

    const int64_t K = S->K, N = S->N, E = 288, n_used = 10;

    struct ggml_init_params ip = { (size_t) 6 << 30, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);

    const int64_t ne0[3] = {K, N, E};
    const int64_t ne1[3] = {K, n_used, n_tok};
    const int64_t ne2[2] = {n_used, n_tok};
    struct ggml_tensor * src0 = ggml_new_tensor(ctx, t, 3, ne0);
    struct ggml_tensor * src1 = ggml_new_tensor(ctx, GGML_TYPE_F32, 3, ne1);
    struct ggml_tensor * ids  = ggml_new_tensor(ctx, GGML_TYPE_I32, 2, ne2);
    struct ggml_tensor * dst  = ggml_mul_mat_id(ctx, src0, src1, ids);

    // deterministic fill: d = 1.0 (fp16 at offset 0 of each block), rest random
    const size_t rs = ggml_row_size(t, K);
    {
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> wd(0, 255);
        std::uniform_real_distribution<float> fd(-1.0f, 1.0f);
        uint8_t * base = (uint8_t *) src0->data;
        std::vector<uint8_t> qrow(rs);
        for (int64_t i = 0; i < N * E; ++i) {
            for (size_t b = 0; b < rs; ++b) qrow[b] = wd(rng);
            qrow[0] = 0x00; qrow[1] = 0x3c;  // fp16 1.0
            memcpy(base + i * rs, qrow.data(), rs);
        }
        float * s1 = (float *) src1->data;
        for (int64_t i = 0; i < K * n_used * n_tok; ++i) s1[i] = fd(rng);
        std::uniform_int_distribution<int> ed(0, (int) E - 1);
        int32_t * id = (int32_t *) ids->data;
        for (int64_t i = 0; i < n_used * n_tok; ++i) id[i] = ed(rng);
    }

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);

    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, NULL);
    std::vector<uint8_t> work(plan.work_size + (16 << 20));
    plan.work_data = work.data();

    ggml_graph_compute(graph, &plan); // warmup

    if (!strcmp(mode, "ref-dump")) {
        if (!golden) { fprintf(stderr, "ref-dump needs a file\n"); return 2; }
        FILE * f = fopen(golden, "wb");
        if (!f) { perror("fopen"); return 2; }
        const size_t n_dst = ggml_nbytes(dst) / sizeof(float);
        if (fwrite(dst->data, sizeof(float), n_dst, f) != n_dst) { perror("fwrite"); return 2; }
        fclose(f);
        printf("ref-dump: %zu floats (%.1f MiB) -> %s [%s/%s]\n",
               n_dst, ggml_nbytes(dst) / 1048576.0, golden, shape_name, type_name);
        return 0;
    }

    // time the current engine
    auto t0 = std::chrono::high_resolution_clock::now();
    const int ITERS = 5;
    for (int i = 0; i < ITERS; ++i) ggml_graph_compute(graph, &plan);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count() / ITERS;

    const double macs = (double) n_tok * n_used * N * K;
    const double expert_bytes = (double) N * E * rs;
    printf("%s/%s %lld tok x %lld used x [%lld x %lld x %lld] %d thr: "
           "%.1f ms, %.1f GB/s(weights), %.0f GMAC/s",
           shape_name, type_name, (long long) n_tok, (long long) n_used,
           (long long) K, (long long) N, (long long) E, n_threads,
           secs * 1e3, expert_bytes / secs / 1e9, macs / secs / 1e9);

    if (!strcmp(mode, "check") || !strcmp(mode, "selftest")) {
        if (!golden) { printf(" (no golden given)\n"); return 2; }
        if (!strcmp(mode, "selftest")) {
            // inject an error to prove the checker catches it
            ((float *) dst->data)[77] += 0.01f * (1.0f + fabsf(((float *) dst->data)[77]));
        }
        FILE * f = fopen(golden, "rb");
        if (!f) { printf(" (golden open failed)\n"); return 2; }
        const size_t n_dst = ggml_nbytes(dst) / sizeof(float);
        std::vector<float> ref(n_dst);
        if (fread(ref.data(), sizeof(float), n_dst, f) != n_dst) { printf(" (golden short)\n"); fclose(f); return 2; }
        fclose(f);
        const float * got = (const float *) dst->data;
        // default tolerance: fp32 accumulation reordering only; use GGMM_TOL
        // for deliberately looser cross-engine checks (e.g. the classic AVX2
        // vec_dot kernel saturates int16 maddubs on adversarial data)
        const double tol_base = getenv("GGMM_TOL") ? atof(getenv("GGMM_TOL")) : 1e-4;
        double max_rel = 0.0; size_t worst = 0; size_t n_bad = 0;
        for (size_t i = 0; i < n_dst; ++i) {
            const double tol = tol_base * (1.0 + fabs((double) ref[i]));
            const double d = fabs((double) got[i] - (double) ref[i]);
            if (d > tol) n_bad++;
            const double rel = d / (1.0 + fabs((double) ref[i]));
            if (rel > max_rel) { max_rel = rel; worst = i; }
        }
        const bool pass = (n_bad == 0);
        printf(" | check: %s (%zu bad, max_rel %.2e @ %zu)\n",
               pass ? "PASS" : "FAIL", n_bad, max_rel, worst);
        return pass ? 0 : 1;
    }
    printf("\n");
    return 0;
}
