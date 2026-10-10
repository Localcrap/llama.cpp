# PolyStrata (`strata-poly`) GLM-5.3-Flash on the 3090 x4 box — 2026-10-10

Repo: https://github.com/VecSzn/PolyStrata (MIT, community engine built on Strata by
Niko1221, uses ggml/llama.cpp parts). Validated upstream on exactly one machine
(RTX 5090 32 GB + Xeon 8470Q, 120 GiB RAM); we are the first known 24 GB run.

## Mechanism (why it beats llama.cpp decode on this box)

Per token GLM routes 8 of 288 experts in each of 42 MoE layers (~2.5 GiB touched).
llama.cpp streams the cold experts over our x4 PCIe link (~7 GB/s) every token —
the measured decode wall (8.3 t/s). strata-poly instead:

- pins a hot-expert cache in VRAM (profile-counted, follows the answer with
  in-flight swaps; 1,545 experts / 10.9 GiB on our 24 GB card at 32K ctx),
- computes the cold experts **on the CPU, concurrently**, reading them from the
  OS page cache (mmap, no copy) — RAM bandwidth ~10x our PCIe link,
- guesses ahead with the model's MTP block + n-gram lookup (answer-preserving),
- prefill streams cold experts through a VRAM staging buffer — same PCIe physics
  as llama.cpp, so pp does not improve.

Our earlier `GGML_CUDA_MMI_ON_CPU=1` all-CPU path failed on ggml's tiled kernels
being feed-bound (~3 GB/s). strata-poly's CPU path sustains the cold traffic
properly — the concept was right, the ggml kernel was the bottleneck.

## Measured on this box (UD-IQ2_XXS, full 288-expert model, 32K ctx)

| run | config | pp | tg |
|---|---|---|---|
| r1 | cold page cache, shipped profile | 167 | 14.86 |
| r2 | same instance, warm+adapted | 181 | **22.12** |
| r3 | fresh restart, our profile | 121 | 15.82 |
| r4 | fresh, `--prompt-batch 2048` | 178 | 14.19 (worse: batch eats VRAM cache) |
| r5 | fresh restart | 158 | 13.43 |
| r6 | same instance, adapted | 177 | 17.92 |
| r7 | fresh, 128K ctx (cache 8.0 GiB / 1,134 experts) | 108 | 10.44 |
| r8 | fresh, 256K ctx (cache 4.2 GiB / 587 experts) | 151 | **6.60** |

Reference, same harness: llama.cpp fork 8.27 t/s / mainline 8.01 t/s (32K),
pp 210-215 warm; llama.cpp at 256K: 8-11 t/s.

Verdict: **2-2.7x llama.cpp decode at <=32K** (15.8 fresh, 18-22 adapted),
still ahead at 128K (~10-14). **At 256K it loses (6.6 fresh)** — KV + sparse
attention overhead plus a 4.9% expert cache; llama.cpp stays the 256K engine.
pp is 121-181 vs llama.cpp 210 — the x4 wall applies to both; no win there.
Repeat prompts are near-instant either way (their checkpoint cache).

## RAM reality (differs from their table)

Their estimate said IQ2_XXS needs ~90 GB beside a 24 GB card; we have 91 total
(~79 usable). It works: the engine never preallocates the cold set — it lets the
page cache hold it lazily. End state: 86 GiB in buff/cache, swap-in during the
bench max 23 MB/s (no meaningful thrash). REAP50 download NOT needed.

## Integration notes

- Lives in `/mnt/ssd/projects/PolyStrata`; engine `build/strata-poly` (built
  with `-DCMAKE_CUDA_ARCHITECTURES=86`, CUDA 12.4), server via
  `python3 serve/server.py --engine strata --config glm.json --port 8095`.
- Restart helper: `bench-moe-cpu/poly-restart.sh`; bench: `poly-bench.sh`.
- `glm.json`: `--max-context 32768 --expert-profile glm.profile` (profile
  updates itself from our answers; flushes every <=10 s).
- OpenAI + Anthropic-compatible APIs + web app on its port. VRAM allows only
  ONE server at a time: strata-poly (23.1 GiB) or llama.cpp (23.6 GiB).
- 256K stays on llama.cpp (`run-glm-q2-ml.sh` / `run-glm-q2-spf.sh`).
- Keep `--conversation-cache-mib` OFF here: parked conversations would evict
  the expert page cache (RAM is 100% committed to it; free = 0).

## Fixes we had to make (all small)

1. `pip install -r requirements.txt` (jinja2, regex) — server wrapper deps.
2. No ninja on this box: drop `-G Ninja`, use the Makefile generator.
3. `-DCMAKE_CUDA_ARCHITECTURES=86` (their check rejects the CMake default 52).
4. Our repaired shard 1 is metadata-only (0 tensors, 9,430,122 B); their parser
   computes `data_start_ = align32(metadata_end)` and throws when
   `data_start_ > size` (9,430,144 > 9,430,122). Padded +22 zero bytes
   (backup: `*.gguf.bak`); llama.cpp unaffected, verified.

## Trade-offs / caveats

- Nondeterminism: an expert computed on CPU vs GPU differs in last digits;
  with swaps enabled a greedy tie can flip across runs. `--expert-swaps 0`
  pins the cache (same prompt -> same answer) at some decode cost.
- Upstream benchmarked 32K only; 64K/128K "start and answer"; 256K never
  tested upstream (ours works but is slower than llama.cpp).
- Maturity: 17 stars, one machine upstream. Source is readable; built and ran
  first try once the four items above were fixed.
- No images, requests queue one at a time (parallel decode unsupported).
