#!/bin/bash
cd /home/takumu/kaggle/arc-agi-3
mkdir -p dev/solutions dev/solvelogs
PY=.venv312/bin/python
for g in ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30; do
  echo "$g"
done | xargs -P 4 -I{} sh -c "timeout 1500 $PY dev/solve2.py {} 150 > dev/solvelogs/{}.log 2>&1; echo done {}"
echo ALL_SOLVE_DONE
