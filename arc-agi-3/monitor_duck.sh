#!/bin/bash
# Poll duck-T4 + 27B kernel status every 10 min; log transitions.
prev14=""; prev27=""
while true; do
  s14=$(kaggle kernels status takumuhata/arc3-duck-qwen3-14b-awq-t4 2>&1 | tail -1 | grep -oE "RUNNING|COMPLETE|ERROR|CANCELING|QUEUED" | head -1)
  s27=$(kaggle kernels status takumuhata/arc3-duck-qwen3-8-27b 2>&1 | tail -1 | grep -oE "RUNNING|COMPLETE|ERROR|CANCELING|QUEUED" | head -1)
  if [ "$s14" != "$prev14" ] || [ "$s27" != "$prev27" ]; then
    echo "$(date -u +%H:%M) duckT4=$s14 qwen27=$s27" >> monitor_duck.log
    prev14=$s14; prev27=$s27
  fi
  # exit when duck-T4 resolves
  [ "$s14" = "COMPLETE" ] || [ "$s14" = "ERROR" ] && break
  sleep 600
done
echo "$(date -u +%H:%M) monitor done duckT4=$s14" >> monitor_duck.log
