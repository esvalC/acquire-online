#!/bin/bash
# READ-ONLY: check pages after the SEO change
for u in / /tutorial /leaderboard /game/1234 /robots.txt /api/version; do
  echo "$u $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3000$u) $(curl -s http://127.0.0.1:3000$u | grep -o '<title>[^<]*\|noindex\|Sitemap.*\|version.*' | head -1)"
done
sudo -u ubuntu pm2 jlist 2>/dev/null | python3 -c 'import json,sys
for a in json.load(sys.stdin): print(a["name"], a["pm2_env"].get("status"), "restarts", a["pm2_env"].get("restart_time"))'
tail -n 5 /home/ubuntu/.pm2/logs/acquire-main-error.log
