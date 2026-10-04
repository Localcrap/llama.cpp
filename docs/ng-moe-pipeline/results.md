# NG MoE pipeline - results log

Append one block per gate run. Format:

    ## YYYY-MM-DD <milestone> <GATE: pass|fail>
    commit: <sha>
    config: <one-liner>
    | metric | value | gate | note |
    decision: <proceed/bisect/stop + why>

## 2026-10-04 baseline (M0 reference row)

commit: 2a6f795a1 (glm5-min)
config: mmid.cpp [4096x2048x288] iq2_xxs, 2048 tok x 10 used, 12 threads

| metric | value |
|---|---|
| ms/op | 207 |
| GB/s (weights) | 3.0 |
| GMAC/s | 917 |
| llama-bench pp8192 (CPU path) | 97-103 t/s |
| llama-bench pp8192 (streaming, default) | 147-161 t/s |

decision: baseline recorded

## 2026-10-04 M0 gate run
commit: 74eb95abf
config: engine=GGML_CPU_TILED_MM=1, 2048 tok x 10 used, 12 threads
| config | ms | GB/s | GMAC/s | check |
|---|---|---|---|---|
| up/iq2_xxs | 198.0 | 3.1 | 868 | PASS |
| down/iq2_xxs | 187.7 | 3.3 | 915 | PASS |
| up/iq3_xxs | 225.5 | 4.1 | 762 | PASS |
| down/iq3_xxs | 204.2 | 4.5 | 841 | PASS |
| up/iq2_s | 227.4 | 3.4 | 756 | PASS |
| up/iq4_xs | 240.8 | 5.3 | 713 | PASS |
decision: (fill in)
