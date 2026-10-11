#!/usr/bin/env bash
#
# Daily-driver launcher: GLM-5.3-Flash on PolyStrata (strata-poly engine).
# https://github.com/VecSzn/PolyStrata - measured 2026-10-10, see
# docs/polystrata-server.md and docs/findings/2026-10-polystrata-glm.md
#
# Usage:
#   run-glm-poly.sh            start (restarts if already running)
#   run-glm-poly.sh stop       stop the server
#
# Knobs (env):
#   CTX=262144        context tokens (32768 = fastest regime, ~2x decode)
#   MTP=off           off = --no-draft. RECOMMENDED on this box: the draft
#                     block is a whole extra MoE layer; turning it off frees
#                     ~0.8 GiB VRAM (+115 experts at 256K) and roughly doubled
#                     fresh-start decode at 256K (13.8 vs 6.6 t/s).
#                     MTP=on restores it (helps only on very predictable text).
#   LOOKUP=3          n-gram repeat-guess length; 0 disables (free feature)
#   HOST=0.0.0.0      bind all interfaces; set to your tailscale IP
#                     (100.102.201.104) for tailnet-only exposure
#   PORT=8095
#   API_KEY=secret    require a bearer token (recommended when LAN-exposed)
#   MODEL=...         any strata-poly-compatible GGUF
#
# Logs:  logs/server.log  live tok/s lines while generating
#        logs/engine.log  load stats, prompt-cache hits, expert counts
#
# NOTE: owns the whole GPU. Stop it (or stop llama.cpp on 8084) before
# starting the other engine.

POLY=/mnt/ssd/projects/PolyStrata
MODEL="${MODEL:-/mnt/ssd/projects/models/glm-5.3-flash/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf}"
CTX="${CTX:-262144}"
MTP="${MTP:-off}"
LOOKUP="${LOOKUP:-3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8095}"

if [ "${1:-start}" = "stop" ]; then
  pkill -f "serve/server" 2>/dev/null
  pkill -f "build/strata" 2>/dev/null
  echo "strata-poly stopped"
  exit 0
fi

cd "$POLY" || exit 1
mkdir -p logs

ARGS=(--model "$MODEL"
      --max-context "$CTX"
      --expert-profile glm.profile
      --lookup "$LOOKUP")
[ "$MTP" = "on" ] || ARGS+=(--no-draft)

ARGS_JSON=$(printf '"%s", ' "${ARGS[@]}")
ARGS_JSON=${ARGS_JSON%, }

cat > glm-run.json <<EOF
{
  "exe": "build/strata-poly",
  "cwd": "$POLY",
  "args": [${ARGS_JSON}],
  "tokenizer": "glm-tokenizer/tokenizer",
  "model_name": "glm-5.3-flash",
  "log": "logs/engine.log"
}
EOF

pkill -f "serve/server" 2>/dev/null
pkill -f "build/strata" 2>/dev/null
sleep 4

[ -n "${API_KEY:-}" ] && export STRATA_API_KEY="$API_KEY"
nohup python3 serve/server.py --engine strata --config glm-run.json \
    --port "$PORT" --host "$HOST" > logs/server.log 2>&1 &

UP=0
for i in $(seq 1 60); do
  curl -s --max-time 2 "localhost:$PORT/v1/models" 2>/dev/null | grep -q glm && { UP=1; break; }
  sleep 5
done
[ "$UP" = 1 ] || { echo "FAILED to start - see $POLY/logs/server.log"; tail -5 "$POLY/logs/engine.log" 2>/dev/null; exit 1; }

grep "routed experts" logs/engine.log | tail -1
TS=$(tailscale ip -4 2>/dev/null | head -1)
echo "up:  http://localhost:$PORT  (web app + OpenAI /v1 + Anthropic /v1/messages)"
[ -n "$TS" ] && echo "     http://$TS:$PORT           (tailscale)"
[ -n "${API_KEY:-}" ] && echo "     auth: bearer $API_KEY"
echo "ctx: $CTX  mtp: $MTP  lookup: $LOOKUP  host: $HOST"
