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
echo "== ALL_CORRECT coverage (generation vs selection split) =="
LOG=$(ls *.log 2>/dev/null | head -1)
if [ -n "$LOG" ]; then
    python3 - <<'PY'
import re, collections, glob
log = glob.glob('/tmp/evalscan/*.log')[0]
ac = collections.defaultdict(list)
for ln in open(log, errors='replace'):
    m = re.search(r'ALL_CORRECT:\s*([\d.]+)\s*-\s*([\d.]+)\s*(\d+x\d+)\s*\[([a-f0-9]+)_(\d+)', ln)
    if m:
        ac[(m.group(4), m.group(5))].append(float(m.group(1)))
gen_strong = gen_weak = gen_none = 0
for k, vs in sorted(ac.items()):
    mx = max(vs)
    tag = 'strong' if mx > 0.1 else 'weak'
    print(f"{k[0]}_{k[1]}: correct-in {len(vs)} subkeys, max {mx:.4f} [{tag}]")
    if mx > 0.1: gen_strong += 1
    else: gen_weak += 1
import json
base = '/home/ubuntu/repos/arc-prize-2026-agent-work/arc-agi-2/'
sol = json.load(open(base+'arc-agi_evaluation_solutions.json'))
ch = json.load(open(base+'arc-agi_evaluation_challenges.json'))
seen = set(ac)
gen_none = 0
by_class = collections.defaultdict(lambda: [0,0,0])  # class -> [gen, none, tot]
for k, so in sol.items():
    for i, gold in enumerate(so):
        ti = ch[k]['test'][0]['input']
        area_in, area_out = len(ti)*len(ti[0]), len(gold)*len(gold[0])
        cl = 'same' if area_out == area_in else ('extract' if area_out < area_in else 'larger')
        hit = (k, str(i)) in seen
        gen_none += (not hit)
        by_class[cl][0] += hit
        by_class[cl][1] += (not hit)
        by_class[cl][2] += 1
print(f"\ntest-outputs with correct grid generated: {len(ac)} "
      f"(strong-beam {gen_strong}, weak-beam {gen_weak})")
print(f"test-outputs where correct NEVER generated (capability miss): {gen_none}")
print("by task class (generated / missed / total):")
for cl, (g, n, t) in sorted(by_class.items()):
    print(f"  {cl:8s} {g:3d} / {n:3d} / {t:3d}   gen-rate {g/max(t,1)*100:.0f}%")
PY
else
    echo "(no kernel log downloaded)"
fi

echo
echo "== selection strategies vs eval solutions =="
python3 /home/ubuntu/repos/arc-prize-2026-agent-work/arc-agi-2/dev/select_bench.py "$OUT/inference_outputs"
