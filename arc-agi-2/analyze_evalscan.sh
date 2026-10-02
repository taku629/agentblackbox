#!/usr/bin/env bash
# Post-evalscan analysis: download output, summarize timing/coverage,
# run the offline selection-strategy benchmark.
#   bash arc-agi-2/analyze_evalscan.sh
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=/home/ubuntu/.venv-kaggle/bin/kaggle
OUT=/tmp/evalscan

echo "== kernel status =="
$KAGGLE kernels status takumuhata/arc-agi2-evalscan || true

mkdir -p "$OUT"
cd "$OUT"
echo "== downloading kernel output -> $OUT =="
$KAGGLE kernels output takumuhata/arc-agi2-evalscan

echo
echo "== timing.log summary =="
if [ -f timing.log ]; then
    python3 - <<'PY'
import re
from collections import defaultdict
totals, ttt, batches = {}, {}, defaultdict(list)
for line in open('/tmp/evalscan/timing.log'):
    m = re.search(r'puzzle (\w+): (\w+(?: \w+)?) at (\d+)s \(dt (\d+)s|puzzle (\w+): TTT done at (\d+)s|puzzle (\w+): TOTAL (\d+)s', line)
    if 'TTT done at' in line:
        k = line.split('puzzle ')[1].split(':')[0]
        ttt[k] = int(line.rsplit(' ', 1)[1].rstrip('s'))
    elif 'batch at' in line:
        k = line.split('puzzle ')[1].split(':')[0]
        dt = int(re.search(r'dt (\d+)s', line).group(1))
        batches[k].append(dt)
    elif 'TOTAL' in line:
        k = line.split('puzzle ')[1].split(':')[0]
        totals[k] = int(line.rsplit(' ', 1)[1].rstrip('s'))
import statistics as st
print(f"tasks completed: {len(totals)}")
if totals:
    v = sorted(totals.values())
    print(f"per-task TOTAL: median {st.median(v):.0f}s, p90 {v[int(len(v)*0.9)]}s, max {v[-1]}s")
    print(f"sum task time: {sum(v)/3600:.1f}h across 4 workers -> ~{sum(v)/4/3600:.1f}h/worker")
if ttt:
    v = sorted(ttt.values())
    print(f"TTT: median {st.median(v):.0f}s, p90 {v[int(len(v)*0.9)]}s, max {v[-1]}s")
bd = [d for v in batches.values() for d in v]
if bd:
    print(f"DFS batches: {len(bd)}, median {st.median(bd):.0f}s each")
PY
else
    echo "(no timing.log — run did not reach decode or output incomplete)"
fi

echo
echo "== selection strategies vs eval solutions =="
python3 /home/ubuntu/repos/arc-prize-2026-agent-work/arc-agi-2/dev/select_bench.py "$OUT/inference_outputs"
