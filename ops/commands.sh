#!/bin/bash
# READ-ONLY diagnostics: what is actually serving ports 3000/3001?
echo "== host"; date; uptime; free -m
echo "== listening ports"; ss -ltnp | grep -E ':(3000|3001|8080)\b' || echo none
echo "== node / pm2 processes"; ps -eo pid,user,lstart,etimes,cmd | grep -E 'node|PM2|pm2' | grep -v grep
for p in $(ss -ltnp | grep -oE ':(3000|3001)\b.*pid=[0-9]+' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
  echo "== pid $p cwd: $(readlink /proc/$p/cwd)  started: $(ps -o lstart= -p $p)"
done
echo "== ubuntu pm2"; sudo -u ubuntu pm2 jlist 2>/dev/null | python3 -c 'import json,sys
for a in json.load(sys.stdin): e=a["pm2_env"]; print(a["name"], "pid", a.get("pid"), e.get("status"), "restarts", e.get("restart_time"), "cwd", e.get("pm_cwd"), "script", e.get("pm_exec_path"))'
echo "== root pm2"; HOME=/root pm2 jlist 2>/dev/null | python3 -c 'import json,sys
for a in json.load(sys.stdin): e=a["pm2_env"]; print(a["name"], "pid", a.get("pid"), e.get("status"), "restarts", e.get("restart_time"), "cwd", e.get("pm_cwd"), "script", e.get("pm_exec_path"))' || echo "no root pm2"
echo "== pm2 systemd units"; systemctl list-units --all | grep -i pm2
echo "== acquire-main error log (ubuntu)"; tail -n 15 /home/ubuntu/.pm2/logs/acquire-main-error.log 2>/dev/null
echo "== acquire-main error log (root)"; tail -n 15 /root/.pm2/logs/acquire-main-error.log 2>/dev/null
echo "== acquire-online error log"; tail -n 10 /home/ubuntu/.pm2/logs/acquire-online-error.log 2>/dev/null
echo "== code on disk"; git -C /home/ubuntu/acquire-main -c safe.directory='*' log --oneline -1; git -C /home/ubuntu/acquire-online -c safe.directory='*' log --oneline -1
