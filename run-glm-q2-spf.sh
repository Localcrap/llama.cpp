#!/usr/bin/env bash

# GLM-5.3-Flash FULL model (unsloth UD-Q2_XXS, 288 experts, 95 GiB) on the
# glm5-min branch, CUDA build for RTX 3090 (sm_86).
#
# Tuned on this box (bench, 2026-10):
#   - ubatch 2048 doubles prompt processing vs 1024 (161 vs 89 t/s pp8192)
#   - decode wants 12 threads (SMT-24 collapses tg to 7 t/s); prefill wants 16
#   - --n-cpu-moe 39: 4 MoE layers' experts (~8 GiB) + all non-expert weights
#     on the 24 GiB 3090. 38 does not fit once ub2048's compute buffer and a
#     32k ctx land (validated: 39 -> 22.1/24 GiB).
#   - warm-state decode is ~13-15 t/s; cold it is ~2 t/s (95 GiB model in
#     91 GiB RAM: expert pages stream from NVMe until resident). Warm up with
#     a single ~3-4k-token prompt right after launch - it routes through all
#     288 experts and makes them page-cache-resident (~20 s at 200+ t/s pp).
#
#   - MTP: the glm5-next NextN graph (implemented in this fork) enables in-model
#     MTP self-drafting: --spec-type draft-mtp. Acceptance 0.83-0.94, warm tg
#     15.4 t/s vs ~13-14 without. Costs +1 n-cpu-moe (blk.45 experts ~2.4 GiB)
#     and -fit off (the VRAM-fit probe crashes on MTP contexts - known issue).
#     Set MTP=off to serve without it.
#
# NOTE: shard 1 of this GGUF was repaired for mainline (arch 'glm5next' ->
# 'glm5-next' + mainline-style kv keys); the original is saved as *.orig-arch.
# Run this model with THIS branch - the patrickbdevaney fork mangles it.

MODEL="${MODEL:-/mnt/ssd/projects/models/glm-5.3-flash/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf}"
SERVER_BIN="${SERVER_BIN:-/mnt/ssd/projects/llama.cpp/build-cuda/bin/llama-server}"

PORT="${PORT:-8084}"
HOST="${HOST:-0.0.0.0}"          # 0.0.0.0 to reach it over tailscale/LAN
CTX="${CTX:-32768}"
MTP="${MTP:-on}"                 # in-model MTP self-drafting for decode
N_CPU_MOE="${N_CPU_MOE:-41}"     # MoE layers with experts on CPU (43 total; 41 when MTP=on)
T_DEC="${T_DEC:-12}"             # decode threads (physical-ish cores)
T_PP="${T_PP:-16}"               # prefill threads
UB="${UB:-2048}"                 # ubatch - biggest pp lever

MTPARGS=()
if [ "$MTP" = "on" ]; then
  MTPARGS=(--spec-type draft-mtp -fit off)
fi

exec "$SERVER_BIN" \
    --model "$MODEL" \
    -ngl 99 \
    --n-cpu-moe "$N_CPU_MOE" \
    --ctx-size "$CTX" \
    --flash-attn on \
    --ubatch-size "$UB" \
    --batch-size 2048 \
    --jinja \
    --host "$HOST" \
    --port "$PORT" \
    --threads "$T_DEC" \
    --threads-batch "$T_PP" \
    "${MTPARGS[@]}" \
    "$@"
