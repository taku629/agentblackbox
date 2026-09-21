#!/bin/bash
# poll OCEAN v13 submission status until it leaves PENDING
while true; do
  out=$(.venv312/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3 2>/dev/null | grep 56408885)
  if echo "$out" | grep -q "COMPLETE\|ERROR"; then
    echo "$(date -u '+%F %T') $out" >> sub_result.log
    break
  fi
  sleep 600
done
