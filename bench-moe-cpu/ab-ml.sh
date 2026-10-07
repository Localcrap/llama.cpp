#!/usr/bin/env bash
# Mainline-vs-fork A/B on the 3090 x4 box: pp / tg / PCIe traffic per token.
# Usage: ab-ml.sh <engine: ml|fork> <config-name> [extra server args...]
set -u
ENG=$1; CFG=$2; shift 2
ML=/mnt/ssd/projects/llama-ml/build-cuda/bin/llama-server
FK=/mnt/ssd/projects/llama.cpp/build-cuda/bin/llama-server
M=/mnt/ssd/projects/models/glm-5.3-flash/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf
PORT=8087
LOG=/tmp/opencode/ab-${ENG}-${CFG}.log

pkill -f "UD-IQ2_XX[S]" 2>/dev/null; sleep 3
if [ "$ENG" = "ml" ]; then BIN=$ML; else BIN=$FK; fi

ARGS=(--model "$M" -ngl 99 -c 32768 -fa on -ub 2048 -b 2048 --jinja
      --host 0.0.0.0 --port $PORT --threads 12 --threads-batch 16 -fit off)
case "$CFG" in
  stream*) ARGS+=(--n-cpu-moe 41) ;;
  cmoe*)   ARGS+=(-cmoe --moe-cache-mib 13000) ;;
  cmoe18)  ARGS+=(-cmoe --moe-cache-mib 18000) ;;
esac
ARGS+=("$@")

nohup "$BIN" "${ARGS[@]}" > "$LOG" 2>&1 &
SPID=$!
for i in $(seq 1 40); do
  curl -s --max-time 2 "localhost:$PORT/health" 2>/dev/null | grep -q '"status":"ok"' && { echo "UP"; break; }
  kill -0 $SPID 2>/dev/null || { echo DIED; grep -aE "error|unable|failed" "$LOG" | head -3; exit 1; }
  sleep 5
done

nohup nvidia-smi dmon -s t -d 1 -o T > /tmp/opencode/dmon-${ENG}-${CFG}.log 2>&1 &
DP=$!
python3 - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
def post(path, body, timeout=1200):
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:'+str(port)+path,
        json.dumps(body).encode(), headers={'Content-Type':'application/json'}), timeout=timeout))
    return r, time.time() - t0
para = ("The quick brown fox jumps over the lazy dog while the robot vacuum cleans the kitchen floor "
        "and the neighbor's cat watches from the fence. ")
for i in (1, 2):
    r, w = post('/completion', {"prompt": para*260, "n_predict": 1, "cache_prompt": False})
    print("pp pass %d: %.1f t/s" % (i, r['timings']['prompt_per_second']))
r, w = post('/v1/chat/completions', {"messages":[{"role":"user","content":"Write a detailed 300-word essay about the history of computing."}],"n_predict":256,"temperature":0})
u = r['usage']
print("tg: %.2f t/s (%d tok in %.1fs)" % (u['completion_tokens']/w, u['completion_tokens'], w))
PY
kill $DP 2>/dev/null
echo "--- pcie rx MB/s: p50/p90/max"
awk 'NR>2 && $1 !~ /#/ {print $(NF-1)}' /tmp/opencode/dmon-${ENG}-${CFG}.log | sort -n | awk '{a[NR]=$1} END {print a[int(NR*0.5)], a[int(NR*0.9)], a[NR]}'
pkill -f "UD-IQ2_XX[S]" 2>/dev/null
echo done
