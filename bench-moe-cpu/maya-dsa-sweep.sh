#!/usr/bin/env bash
# A/B the MAYA_GLM_DSA_SCORE2 take-two labels on the live box.
# Baseline (unset) + 6 shuffle-layout labels; each: restart -> 2 gens ->
# take the last "glm stat: decode ... ms/tok" from the engine log.
cd /mnt/ssd/projects/project-maya || exit 1
RES=/tmp/opencode/dsa-sweep-results.txt
: > "$RES"

set_env() {
  python3 - "$1" <<'PY'
import json, sys
p = 'maya-maya-s24.json'
c = json.load(open(p))
v = sys.argv[1]
if v == "unset":
    c.get('env', {}).pop('MAYA_GLM_DSA_SCORE2', None)
else:
    c.setdefault('env', {})['MAYA_GLM_DSA_SCORE2'] = v
json.dump(c, open(p, 'w'), indent=2)
PY
}

restart_and_measure() {  # $1 = label
  /mnt/ssd/projects/glm53-reap-llamacpp/bench-moe-cpu/maya-stop.sh > /dev/null 2>&1
  set_env "$1"
  nohup ./maya.sh > "/tmp/opencode/maya-start-dsa.log" 2>&1 &
  for i in $(seq 1 80); do
    curl -s --max-time 3 localhost:8096/v1/models 2>/dev/null | grep -q glm && break
    sleep 6
  done
  python3 - <<'PY' > /dev/null 2>&1
import json, time, urllib.request
def post(n):
    urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8096/v1/chat/completions',
        json.dumps({"messages":[{"role":"user","content":"Write a detailed 300-word essay about the history of computing."}],
                    "max_tokens":n,"temperature":0}).encode(),
        headers={'Content-Type':'application/json'}), timeout=1200).read()
post(64)   # warm
post(192)  # measured
PY
  STAT=$(grep "glm stat: decode" maya-maya-s24.log | tail -1 | grep -oE "decode [0-9.]+ ms/tok \([0-9.]+ tok/s\)")
  echo "$1 : $STAT" | tee -a "$RES"
}

restart_and_measure unset
restart_and_measure rx1b4
restart_and_measure rx2b4
restart_and_measure rx2b8
restart_and_measure rx1b8
restart_and_measure sx1b4
restart_and_measure sx2b8
echo "=== sweep done ===" >> "$RES"
cat "$RES"
