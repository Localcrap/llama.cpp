#!/usr/bin/env bash

# GLM-5.3-Flash on MAINLINE llama.cpp (master bd4eeaa04+, with MoE expert cache).
# Binary: /mnt/ssd/projects/llama-ml/build-cuda/bin/llama-server
#
# Tuned settings carried over from the glm5-min measurements (2026-10):
#   -ub 2048 (pp 2x vs 1024), -t 12 decode / --threads-batch 16 prefill,
#   -fa on, streaming via --n-cpu-moe 41 (default) or -cmoe + MoE cache.
#
# MoE cache note (PR #29887, measured on this box 2026-10-07):
#   MOE_CACHE=on adds -cmoe --moe-cache-mib 13000. On our x4 PCIe link the
#   cache LOST decode (3.6 t/s vs 8-8.3 streaming): 13 GiB covers only ~15%
#   of the 89 GiB experts, hit rate too low, misses upload over the same
#   slow link. Off by default; try it if the expert set shrinks.
#
# MTP: mainline has glm5-next MTP (#29928). Neutral for open-ended text,
# +10-20% on predictable output (measured). MTP=on adds --spec-type draft-mtp.

MODEL="${MODEL:-/mnt/ssd/projects/models/glm-5.3-flash/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf}"
BIN="${BIN:-/mnt/ssd/projects/llama-ml/build-cuda/bin/llama-server}"

PORT="${PORT:-8084}"
HOST="${HOST:-0.0.0.0}"
CTX="${CTX:-32768}"              # 262144 works too (see notes below)
MTP="${MTP:-off}"
N_CPU_MOE="${N_CPU_MOE:-41}"     # 43 if CTX >= 262144 (VRAM wall)
T_DEC="${T_DEC:-12}"
T_PP="${T_PP:-16}"
UB="${UB:-2048}"

MTPARGS=()
if [ "$MTP" = "on" ]; then
  MTPARGS=(--spec-type draft-mtp)
fi

exec "$BIN" \
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
