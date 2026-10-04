# NG MoE pipeline - results log

Append one block per gate run. Format:

```
## YYYY-MM-DD <milestone> <GATE: pass|fail>
commit: <sha>
config: <one-liner>
| metric | value | gate | note |
decision: <proceed/bisect/stop + why>
```

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
```
