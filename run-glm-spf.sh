#!/usr/bin/env bash

# GLM-5.3-Flash REAP50 server (patrickbdevaney fork) with speculative prefill.
# The draft (-md, Qwen3.5-2B, cross-family) only compresses prompts: importance
# estimation via attention, keep top P fraction, sparse target prefill.
# Decode speculation is force-disabled while --spec-prefill is on (prefill_only).
#
# Measured (13.7k-token wiki page, 3090 + CPU experts):
#   dense prefill  ~160 s   ->  sparse p=0.2  ~34 s (4.7x TTFT)
# Short prompts gain little: every decode pays a fixed ~7 s expert-touch cost.
# p=0.2 = target sees 20% of the prompt - check quality on your workload.

MODEL="${MODEL:-/mnt/ssd/projects/models/glm-5.3-flash/REAP50/GLM-5.3-Flash-REAP50-IQ3_M.gguf}"
DRAFT="${DRAFT:-/mnt/ssd/projects/models/Qwen3.5-2B-Q8_0.gguf}"
SERVER_BIN="${SERVER_BIN:-/mnt/ssd/projects/glm53-reap-llamacpp/build/bin/llama-server}"

PORT="${PORT:-8082}"
HOST="${HOST:-0.0.0.0}"         # 0.0.0.0 to reach it over tailscale/LAN
CTX="${CTX:-32768}"
SPEC_CTX="${SPEC_CTX:-16384}"   # draft context; must fit prompt + lookahead
P="${P:-0.2}"                   # fraction of prompt tokens kept
THREADS="${THREADS:-12}"

exec "$SERVER_BIN" \
    --model "$MODEL" \
    -md "$DRAFT" \
    --spec-prefill \
    --spec-prefill-ngl 99 \
    --spec-prefill-ctx "$SPEC_CTX" \
    --spec-prefill-p "$P" \
    -ngl 99 \
    -ot '\.ffn_.*_exps\.=CPU' \
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
