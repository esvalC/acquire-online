#!/bin/bash
# Nightly database backup → S3 (run by cron as ubuntu).
#   backups/daily/acquire-<weekday>.db.gz   overwritten weekly → last 7 days
#   backups/monthly/acquire-<YYYY-MM>.db.gz kept (one per month, a few MB each)
# The server can't delete from S3, so rotation is done by overwriting.
# Restore: aws s3 cp s3://BUCKET/backups/daily/acquire-mon.db.gz - | gunzip > acquire.db
set -euo pipefail
BUCKET="${S3_BUCKET:-acquire-training-data}"
DB="${DB_PATH:-/home/ubuntu/acquire-data/acquire.db}"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
umask 077

# Consistent snapshot even while the site is writing (includes the WAL)
sqlite3 "$DB" ".backup '$TMP/acquire.db'"
CHECK=$(sqlite3 "$TMP/acquire.db" "PRAGMA integrity_check;")
[ "$CHECK" = "ok" ] || { echo "backup FAILED integrity check: $CHECK" >&2; exit 1; }
USERS=$(sqlite3 "$TMP/acquire.db" "SELECT COUNT(*) FROM users;")
gzip -9 "$TMP/acquire.db"
gunzip -t "$TMP/acquire.db.gz"

DAY=$(date -u +%a | tr 'A-Z' 'a-z'); MONTH=$(date -u +%Y-%m)
aws s3 cp "$TMP/acquire.db.gz" "s3://$BUCKET/backups/daily/acquire-$DAY.db.gz" --sse AES256 --only-show-errors
aws s3 cp "$TMP/acquire.db.gz" "s3://$BUCKET/backups/monthly/acquire-$MONTH.db.gz" --sse AES256 --only-show-errors
echo "$(date -u +%FT%TZ) backup ok: $(du -h "$TMP/acquire.db.gz" | cut -f1), $USERS users → $DAY + $MONTH"
