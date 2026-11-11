#!/bin/bash
# READ-ONLY: is the new build healthy?
echo "== pm2"; sudo -u ubuntu pm2 jlist 2>/dev/null | python3 -c 'import json,sys
for a in json.load(sys.stdin): e=a["pm2_env"]; print(a["name"], e.get("status"), "restarts", e.get("restart_time"), "uptime_s", (__import__("time").time()*1000-e.get("pm_uptime",0))//1000)'
echo "== code"; git -C /home/ubuntu/acquire-main -c safe.directory='*' log --oneline -1
echo "== leaderboard api"; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000/api/leaderboard; curl -s http://127.0.0.1:3000/api/leaderboard | head -c 300; echo
echo "== /leaderboard page"; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:3000/leaderboard
echo "== recent out log"; tail -n 25 /home/ubuntu/.pm2/logs/acquire-main-out.log
echo "== recent error log"; tail -n 15 /home/ubuntu/.pm2/logs/acquire-main-error.log
