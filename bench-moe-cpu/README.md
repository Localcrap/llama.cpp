# MoE CPU-path benchmarking tools (2026-10-04 session)

Isolated benchmarks for the ggml-cpu MUL_MAT_ID (MoE) path, built during the
"tiled pipeline" investigation. Link against build-cuda's libggml-cpu.

    g++ -O2 -march=native mmid.cpp -I<ggml>/ggml/include -L<build>/bin -lggml-cpu -lggml-base -o mmid
    LD_LIBRARY_PATH=<build>/bin ./mmid 2048 12          # one expert tensor op, isolated

    g++ -O3 -march=native unpack_bench.cpp -o unpack_bench && ./unpack_bench 16384
    # reference tiled unpack vs AVX-512 candidates (correctness + GB/s)

## Findings (GLM-5.3-Flash UD-Q2_XXS shapes: [4096 x 2048 x 288] x 2048 tok x 10 used)

- ggml-cpu "tiled" path: 207 ms/op = 3.0 GB/s aggregate (12 threads), 917 GMAC/s
  - anchored by the microkernel's 512 scalar u32 loads + broadcasts per 8x16 tile
    (load-port bound at ~13% of Zen5 VNNI peak)
- stock vec_dot path (GGML_CPU_TILED_MM=0): 860 ms/op = 0.7 GB/s (LUT thrash on
  RAM-streamed weights; the 2.3 GB/s/core kernel figure only holds L2-resident)
- GPU PCIe streaming of the same op: ~6.5-7 GB/s (x4 Gen4 ceiling) = fastest today
- v2 (AVX-512 column-table vpermi2b unpack): correct (0/16384) but 1.7 GB/s -
  vpermi2b column lookups do not pay off on Zen5 for this shape

## Conclusion

The CPU MoE path is ~2.2x off the PCIe streaming path. Making it competitive
needs a next-gen tiled pipeline (8x32 microtiles to amortize scalar weight
loads, ~1.5x projected - still short) or a full rewrite (~5x theoretical at
RAM bandwidth). None of it beats the streaming path on this box without
multi-week upstream-scale work.
