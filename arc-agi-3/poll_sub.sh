#!/bin/bash
cd "$(dirname "$0")"
# poll OCEAN v13 submission status until it leaves PENDING
while true; do
  out=$(.venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep ${1:-56408885})
  if echo "$out" | grep -q "COMPLETE\|ERROR"; then
    echo "$(date -u '+%F %T') $out" >> sub_result.log
    break
  fi
  sleep 600
done
