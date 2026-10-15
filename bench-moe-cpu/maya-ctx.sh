#!/usr/bin/env bash
# Stop maya (whole tree), set context in the config, start again, verify.
cd /mnt/ssd/projects/project-maya || exit 1
CTX="${1:-262144}"

pkill -f "serve/server.py" 2>/dev/null
pkill -f "build/strata" 2>/dev/null
pkill -f "maya.py" 2>/dev/null
sleep 6

python3 - "$CTX" <<'PY'
import json, sys
p = 'maya-maya-s24.json'
c = json.load(open(p))
a = c['args']
i = a.index('--max-context')
a[i+1] = sys.argv[1]
json.dump(c, open(p, 'w'), indent=2)
print('config ctx ->', a[i+1])
PY

nohup ./maya.sh > /tmp/opencode/maya-start.log 2>&1 &
for i in $(seq 1 100); do
  curl -s --max-time 3 localhost:8096/v1/models 2>/dev/null | grep -q glm && { echo "maya up"; break; }
  sleep 6
done
curl -s --max-time 5 localhost:8096/v1/models | grep -oE '"n_ctx": [0-9]+'
