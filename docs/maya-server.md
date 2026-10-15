# GLM-5.3-Flash daily driver: Project Maya (Maya-S24)

Adopted 2026-10-14 as THE daily engine on this box (single RTX 3090 24 GB,
x4 PCIe, 91 GiB RAM). Supersedes strata-poly (kept: `docs/polystrata-server.md`)
and llama.cpp for serving. Evaluation: `docs/ng-moe-pipeline/results.md`.

## Measured here (262144 ctx, tuned)

| engine | prefill | decode |
|---|---|---|
| **Maya-S24** | **424 t/s** | **22.3 fresh / 29.5 warm** |
| strata-poly | ~150-180 | ~14 / ~20 |
| llama.cpp | ~210 | 8-11 |

Calibration (`--calibrate`, 10-15 min) picked `STRATA_GLM_PCIE_SHARE=0.10`
at 24.7 tok/s; do NOT skip it after hardware changes. Warm = second
generation on the same question; first answers after a start are slowest
(caches fill with what YOU use).

## Start / stop / context

```bash
cd /mnt/ssd/projects/project-maya
./maya.sh                                  # foreground start (Ctrl+C stops)
nohup ./maya.sh > /tmp/opencode/maya-start.log 2>&1 &   # headless start
pkill -f "serve/server.py"; pkill -f "build/strata"     # stop
```

Config: `maya-maya-s24.json` — `--max-context` (set to 262144), tuned env in
`env`. Change context: `bench-moe-cpu/maya-ctx.sh 262144` (stop + edit + start
+ verify `n_ctx` via the API). First start at a new context takes a few
minutes (it pins ~all free RAM but ~6 GB for the expert tier — the desktop
WILL feel it; `STRATA_GLM_RAM_HEADROOM_GB` raises the 6).

Endpoints: `http://<host>:8096/` dashboard, `/v1` OpenAI, `/v1/messages`
Anthropic. Bound 0.0.0.0, no API key (consistent with strata-poly; add
`--api-key` if that ever bothers you). Tailscale: `100.102.201.104:8096`.

## What Maya is

Engine + server + dashboard dedicated to GLM-5.3-Flash (MIT), built from
Strata's code, rewritten: three-tier experts (VRAM hot set -> RAM -> NVMe,
migrated as you chat), self-tuning, kv-streaming attention (cache in RAM,
window in VRAM — why 262144 ctx costs little VRAM), MTP block active,
optional DFlash2 drafter (skipped: non-commercial license), `--parallel N`
(up to 8 concurrent requests; not enabled). Own FP8-calibrated quants:
Maya-S24 claims 97.7% of FP8 zero-shot accuracy at ~95 GB — quality per GB
at least matches unsloth UD-IQ2_XXS. Reads pictures (vision encoder, 2.7 GB
on demand). 838 stars, active project (v1.1.10).

## Logs / diagnosis

- Engine log: `project-maya/maya-maya-s24.log`
- Server output: wherever nohup redirected (see command above)
- `./maya.sh --report` after a slow answer: hardware + setup + speed lines
- `./maya.sh --calibrate` re-tunes; `./maya.sh --bench` needs the server STOPPED

## Coexistence

- Owns the whole GPU: stop Maya before strata-poly (`run-glm-poly.sh`) or
  llama.cpp (`run-glm-q2-ml.sh`), and vice versa.
- RAM: while Maya runs, ~6 GB headroom is all other programs get.
- Disk: model at `/mnt/ssd/projects/Maya-data/models/Maya-S24` (~95 GB).
- Update: `git pull && ./maya.sh` (or dashboard About > Update); the model is
  not re-downloaded. Re-run `--calibrate` after GPU/CPU/RAM changes.
