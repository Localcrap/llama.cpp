#!/usr/bin/env bash
# PolyStrata strata-poly GLM bench on the 3090 x4 box.
# Same shape as ab-ml.sh: 2x pp passes (~3k tok) + 256-tok generation.
# Usage: poly-bench.sh [label]
set -u
LABEL=${1:-run}
PORT=8095
LOG=/tmp/opencode/poly-bench-${LABEL}.log

curl -s --max-time 3 "localhost:$PORT/v1/models" | grep -q glm || { echo "server not up"; exit 1; }

nohup nvidia-smi dmon -s t -d 1 -o T > /tmp/opencode/dmon-poly-${LABEL}.log 2>&1 &
DP=$!
( vmstat 2 > /tmp/opencode/vmstat-poly-${LABEL}.log & VP=$!; wait $DP ) 2>/dev/null

python3 - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
def post(body, timeout=3600):
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        'http://127.0.0.1:'+str(port)+'/v1/chat/completions',
        json.dumps(body).encode(), headers={'Content-Type':'application/json'}), timeout=timeout))
    return r, time.time() - t0

para = ("The quick brown fox jumps over the lazy dog while the robot vacuum cleans the kitchen floor "
        "and the neighbor's cat watches from the fence. ")
for i in (1, 2):
    r, w = post({"messages":[{"role":"user","content":para*150}], "max_tokens":1, "temperature":0})
    u = r.get('usage', {})
    print("pp pass %d: %d tok in %.1fs = %.1f t/s" % (i, u.get('prompt_tokens',0), w, u.get('prompt_tokens',0)/w))

r, w = post({"messages":[{"role":"user","content":"Write a detailed 300-word essay about the history of computing."}],
             "max_tokens":256, "temperature":0})
u = r.get('usage', {})
print("tg: %.2f t/s (%d tok in %.1fs, pp %d tok)" % (u.get('completion_tokens',0)/w, u.get('completion_tokens',0), w, u.get('prompt_tokens',0)))
PY
kill $DP 2>/dev/null; pkill -f "vmstat 2" 2>/dev/null
echo "--- pcie rx MB/s p50/p90/max:"
awk 'NR>2 && $1 !~ /#/ {print $(NF-1)}' /tmp/opencode/dmon-poly-${LABEL}.log | sort -n | awk '{a[NR]=$1} END {print a[int(NR*0.5)], a[int(NR*0.9)], a[NR]}'
echo "--- swap si/so max (thrash):"
awk 'END {print "si", si, "so", so} {if($7+0>si)si=$7; if($8+0>so)so=$8}' /tmp/opencode/vmstat-poly-${LABEL}.log 2>/dev/null
echo "--- VRAM/RAM now:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader; free -g | head -2
