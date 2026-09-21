#!/bin/bash
prev14=""; prev32=""
while true; do
  s14=$(kaggle kernels status takumuhata/arc3-duck-qwen3-14b-awq-t4 2>&1 | tail -1 | grep -oE "RUNNING|COMPLETE|ERROR|CANCEL" | head -1)
  s32=$(kaggle kernels status takumuhata/arc3-duck-qwen3-32b-awq-t4 2>&1 | tail -1 | grep -oE "RUNNING|COMPLETE|ERROR|CANCEL" | head -1)
  if [ "$s14" != "$prev14" ] || [ "$s32" != "$prev32" ]; then
    echo "$(date -u +%H:%M) duckT4-14b=$s14 duckT4-32b=$s32" >> monitor_duck.log
    prev14=$s14; prev32=$s32
  fi
  { [ "$s14" = "COMPLETE" ] || [ "$s14" = "ERROR" ]; } && { [ "$s32" = "COMPLETE" ] || [ "$s32" = "ERROR" ]; } && break
  sleep 600
done
echo "$(date -u +%H:%M) monitor2 done 14b=$s14 32b=$s32" >> monitor_duck.log
