#!/usr/bin/env bash
# Stop the whole Maya process tree (server + engine + python wrapper).
pkill -f "serve/server.py" 2>/dev/null
pkill -f "build/strata" 2>/dev/null
pkill -f "maya.py" 2>/dev/null
sleep 5
pgrep -af "maya.py|serve/server.py|build/strata" | grep -v grep || echo "maya fully stopped"
