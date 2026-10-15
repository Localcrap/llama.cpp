# GLM-5.3-Flash daily driver: PolyStrata (`strata-poly`)

> SUPERSEDED as daily driver 2026-10-14 by Project Maya: docs/maya-server.md (2x decode,
> 2x prefill at 262144 ctx, parallel requests). Kept as an alternative engine; the
> guide below still applies if you start it (stop Maya first - one engine owns the GPU).

Adopted 2026-10-10 as the daily engine for context <= 128K; llama.cpp stays the
256K engine. Evaluation data: [findings/2026-10-polystrata-glm.md](findings/2026-10-polystrata-glm.md).

## Start / stop

```bash
/mnt/ssd/projects/glm53-reap-llamacpp/run-glm-poly.sh          # start (defaults: 262144 ctx, MTP off)
/mnt/ssd/projects/glm53-reap-llamacpp/run-glm-poly.sh stop     # stop
```

First start after reboot takes ~1-2 min (page cache refills while it answers);
restarts while warm take ~25 s. Endpoints (any model name / API key unless
`API_KEY` is set):

| what | url |
|---|---|
| Web app + Monitor | `http://<host>:8095` |
| OpenAI-compatible | `http://<host>:8095/v1` |
| Anthropic-compatible | `ANTHROPIC_BASE_URL=http://<host>:8095` |

Over tailscale use `100.102.201.104:8095`. Default bind is `0.0.0.0` (LAN +
tailscale); set `HOST=100.102.201.104` for tailnet-only, `API_KEY=...` to
require a bearer token.

## Knobs

```bash
CTX=32768 ./run-glm-poly.sh     # fast regime
MTP=on   ./run-glm-poly.sh      # re-enable the draft block
LOOKUP=0 ./run-glm-poly.sh      # disable n-gram guessing too
```

## MTP: on vs off (the one that matters)

The draft ("MTP") block is a full extra MoE layer, not a small head. On this
box it is a net LOSS:

| 262144 ctx | experts in VRAM | fresh decode |
|---|---|---|
| MTP on | 587 / 4.2 GiB (of 12,384) | 6.6 t/s |
| **MTP off (default)** | **702 / 5.0 GiB (of 12,096)** | **13.8 t/s** |

Why: UD-IQ2_XXS's i-quant experts make a guessed token cost nearly a full
token on the CPU (upstream measured ~31 t/s with and without the block), so
the draft layer's 0.8 GiB + per-step forward is pure overhead here. Upstream's
K-quant sizes (3.0BIT, REAP50) do gain 10-25% from it — irrelevant for us.
`MTP=on` only makes sense to reproduce upstream numbers or if a future quant
changes the math. `LOOKUP=3` (n-gram repeat guessing) stays on in both modes:
it costs no extra forward pass and helps on repetitive text.

Context vs cache (MTP off adds ~+20% experts at every ctx):

| ctx | experts in VRAM |
|---|---|
| 32K | 1,545 / 10.9 GiB (measured with MTP; no-draft ~1,700 est.) |
| 128K | 1,134 / 8.0 GiB (with MTP) |
| 262K | 702 / 5.0 GiB (no-draft) |

Decode expectations: 32K ~18-22 t/s adapted, 128K ~10-14, 262K ~14 fresh and
climbing as the profile adapts. First answer after a restart is always the
slowest — the VRAM cache only holds the profile's *predicted* hot experts and
re-picks during each answer.

## Logs

```bash
tail -f /mnt/ssd/projects/PolyStrata/logs/server.log   # live tok/s, one line per second
tail -f /mnt/ssd/projects/PolyStrata/logs/engine.log   # load stats, prompt-cache hits
```

`engine.log` per load: `N of M routed experts in VRAM (X GiB)`; per answer:
prompt-cache hit %, parked conversations, VRAM free.

## The expert profile (make it yours)

`/mnt/ssd/projects/PolyStrata/glm.profile` counts which experts YOUR work uses; the
engine loads the hot ones into VRAM at start and rewrites the file as you use
it (flush every <=10 s and at stop). The longer you use it, the better starts
get. Deleting it resets to the shipped profile.

## Switching engines (VRAM is exclusive)

llama.cpp (`run-glm-q2-ml.sh`, port 8084) and strata-poly (port 8095) cannot
run at the same time (23.6 vs 23.1 GiB of 24). Stop one before starting the
other. Rule of thumb:

- **strata-poly**: interactive chat/coding/agents at <= 128K, repeat-prompt
  iteration (checkpoint cache: re-sends are near-instant), tailscale serving.
- **llama.cpp**: 262144-ctx work (strata-poly drops below it there), llama.cpp
  API features (lambdas like /completion, loRA, etc.), anything benchmarking.

## Updating (git pull + rebuild)

```bash
/mnt/ssd/projects/glm53-reap-llamacpp/run-glm-poly.sh stop   # engine must be stopped to relink
cd /mnt/ssd/projects/PolyStrata && git pull --ff-only
cmake --build build --target strata-poly -j12
/mnt/ssd/projects/glm53-reap-llamacpp/run-glm-poly.sh        # same config comes back up
```

Notes from the 2026-10-11 update:

- The engine binary cannot be overwritten while running (ETXTBSY) — stop first.
- Untracked local files (`glm.json`, `glm-run.json`, `glm.profile`,
  `glm-tokenizer/`, `logs/`) survive pulls untouched.
- Flags occasionally change upstream: if the server fails to start right after
  an update with a usage dump in `logs/server.log`, an option was renamed or
  dropped — check `./build/strata-poly` (prints the current option list) and
  fix `run-glm-poly.sh`'s `ARGS`.
- If `build/` does not exist (fresh clone), configure first — no ninja on this
  box, and the arch flag is mandatory (their check rejects CMake's default 52):

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86
  ```

- Upstream also ships `./update.sh`, but it targets their `setup.sh`-managed
  installs; our manual build keeps our own flags and folder layout.

## If something breaks

- **Server won't start / "GGUF: data section starts past EOF"**: fixed by the
  +22 B pad on shard 1 (backup `GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf.bak`
  next to the model). Never re-download shard 1 over it without re-padding.
- **ModuleNotFoundError on server start**: `pip install -r requirements.txt`
  inside the PolyStrata folder (jinja2, regex).
- **RAM**: the cold experts live in the page cache (~80 GiB wanted). Close
  RAM-heavy apps before starting; browser + VMs will push experts to swap and
  decode tanks. `free -g` should show buff/cache ~86 GiB when settled.
- **Nondeterminism**: with the adaptive cache on, the same prompt can differ
  in last digits across runs (expert computed on CPU vs GPU). Not fixable and
  harmless for chat; `--expert-swaps 0` (add to glm-run.json args) pins it at
  a decode cost.

## Files

| path | what |
|---|---|
| `/mnt/ssd/projects/PolyStrata` | the engine + server (git clone) |
| `run-glm-poly.sh` (this repo) | daily launcher |
| `bench-moe-cpu/poly-restart.sh`, `poly-bench.sh` | bench helpers (use `glm.json`, 32K) |
| `PolyStrata/glm-run.json` | generated by the launcher each start |
| `PolyStrata/glm.profile` | your learned expert profile (precious, back it up) |
