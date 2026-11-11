#!/bin/bash
# READ-ONLY: inspect the latest game with a player named "Player"
cd /home/ubuntu/acquire-main
sudo -u ubuntu node -e '
const Database=require("better-sqlite3"); const zlib=require("zlib");
const db=new Database("/home/ubuntu/acquire-data/acquire.db",{readonly:true});
const rows=db.prepare("SELECT id, played_at, is_solo, mode, end_type, turn_count, duration_seconds, standings FROM game_sessions WHERE standings LIKE ? ORDER BY id DESC LIMIT 4").all("%\"name\":\"Player\"%");
for (const r of rows) console.log(JSON.stringify(r));
const s=rows[0]; if(!s) process.exit();
const g=db.prepare("SELECT replay_gz FROM game_sessions WHERE id=?").get(s.id);
const snaps=JSON.parse(zlib.gunzipSync(g.replay_gz).toString());
console.log("snapshots", snaps.length);
for (const sn of snaps.slice(-14)) {
  const big=Object.entries(sn.chains).filter(([,c])=>c.active).map(([k,c])=>k+":"+c.size+(c.size>=11?"*":"")).join(" ");
  console.log("turn",sn.turnNumber,"phase",sn.phase,"cur",sn.players[sn.currentPlayerIdx]&&sn.players[sn.currentPlayerIdx].name,"canEnd",sn.canDeclareEnd,"endDeclared",!!sn.endDeclared,"concede",sn.concedeActive,"|",big,"| bag",sn.tileBagCount);
}
const last=snaps[snaps.length-1]; console.log("END", JSON.stringify(last.endReason)); console.log((last.log||[]).slice(-25).join("\n"));
'
