#!/usr/bin/env bash
# Restart the PolyStrata server and wait until /v1/models answers.
# Lives in its own file so pkill patterns never match a caller's cmdline.
cd /mnt/ssd/projects/PolyStrata || exit 1
pkill -f "serve/server" 2>/dev/null
pkill -f "build/strata" 2>/dev/null
sleep 4
nohup python3 serve/server.py --engine strata --config glm.json --port 8095 > /tmp/opencode/poly-server.log 2>&1 &
for i in $(seq 1 60); do
  curl -s --max-time 2 localhost:8095/v1/models 2>/dev/null | grep -q glm && { echo "server up (~$((i*5))s)"; break; }
  sleep 5
done
grep "routed experts" glm-engine.log | tail -1
