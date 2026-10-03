#!/usr/bin/env bash

# GLM-5.3-Flash FULL model (unsloth UD-Q2_XXS, 288 experts) on MAINLINE llama.cpp,
# CUDA build (RTX 3090).
#
# Expert split: --n-cpu-moe keeps the FIRST N MoE layers' experts on the CPU
# (mmap-backed) and the LAST layers on the GPU. Non-expert weights (~5.1 GiB) are
# always on the GPU. N_CPU_MOE=36 -> last 7 MoE layers (~14 GiB) on the 3090.
#
# NOTE: shard 1 of this GGUF was repaired for mainline (arch 'glm5next' -> 'glm5-next'
# + mainline-style kv keys); the original is saved as *.orig-arch next to it.
#
# The unsloth quant follows mainline's implementation - run it with THIS binary,
# NOT with the patrickbdevaney fork (different math, output degenerates there).

MODEL="${MODEL:-/mnt/ssd/projects/models/glm-5.3-flash/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf}"
SERVER_BIN="${SERVER_BIN:-/mnt/ssd/projects/llama.cpp/build-cuda/bin/llama-server}"

PORT="${PORT:-8084}"
HOST="${HOST:-0.0.0.0}"         # 0.0.0.0 to reach it over tailscale/LAN
CTX="${CTX:-32768}"
N_CPU_MOE="${N_CPU_MOE:-36}"    # MoE layers with experts on CPU (43 total; 37 = 6 on GPU)
THREADS="${THREADS:-12}"

exec "$SERVER_BIN" \
    --model "$MODEL" \
    -ngl 99 \
    --n-cpu-moe "$N_CPU_MOE" \
    --ctx-size "$CTX" \
    --flash-attn on \
    --jinja \
    --host "$HOST" \
    --port "$PORT" \
    --threads "$THREADS" \
    --threads-batch "$THREADS" \
    --batch-size 2048 \
    --ubatch-size 1024 \
    "$@"
