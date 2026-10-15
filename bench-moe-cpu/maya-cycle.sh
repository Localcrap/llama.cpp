#!/usr/bin/env bash
# Stop maya, calibrate, stop, bench, restart. Safe to re-run.
cd /mnt/ssd/projects/project-maya || exit 1

stop_maya() {
  pgrep -af "maya.py --setup|port 8096|build/strata" >> /tmp/opencode/maya-cycle.log
  pkill -f "maya.py --setup" 2>/dev/null
  pkill -f "serve/server.py" 2>/dev/null
  pkill -f "build/strata" 2>/dev/null
  sleep 6
}

stop_maya
echo "=== calibrate ===" >> /tmp/opencode/maya-cycle.log
./maya.sh --calibrate >> /tmp/opencode/maya-cycle.log 2>&1
stop_maya
echo "=== bench ===" >> /tmp/opencode/maya-cycle.log
./maya.sh --bench >> /tmp/opencode/maya-cycle.log 2>&1
tail -25 /tmp/opencode/maya-cycle.log

echo "=== final start ==="
nohup ./maya.sh >> /tmp/opencode/maya-cycle.log 2>&1 &
for i in $(seq 1 60); do
  curl -s --max-time 3 localhost:8096/v1/models 2>/dev/null | grep -q glm && { echo "MAYA BACK UP"; break; }
  sleep 5
done
