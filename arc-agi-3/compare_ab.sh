#!/usr/bin/env bash
# A/B compare: per-game scores of anim-v2 (control) vs flashnext-v3 (v4 HARD CAP).
#   bash arc-agi-3/compare_ab.sh
set -euo pipefail
KAGGLE=/home/ubuntu/.venv-kaggle/bin/kaggle
for k in arc3-duck-anim-v2 arc3-duck-anim-flashnext-v3; do
  echo "== $k =="
  $KAGGLE kernels status takumuhata/$k || true
  mkdir -p /tmp/ab_$k && cd /tmp/ab_$k
  $KAGGLE kernels output takumuhata/$k >/dev/null 2>&1 || true
done
python3 - <<'PY'
import json, glob, os, collections
def scores(d):
    for cand in [f'{d}/score.json', f'{d}/scores.json'] + glob.glob(f'{d}/**/score*.json', recursive=True):
        if os.path.exists(cand):
            try:
                s = json.load(open(cand))
                if isinstance(s, dict):
                    return s
            except Exception: pass
    return {}
def summary(d):
    for cand in [f'{d}/summary.txt']:
        if os.path.exists(cand): return open(cand).read()
    return ''
a = scores('/tmp/ab_arc3-duck-anim-v2'); b = scores('/tmp/ab_arc3-duck-anim-flashnext-v3')
print('\ncontrol (anim-v2) summary:\n', summary('/tmp/ab_arc3-duck-anim-v2')[:800])
print('\nexperiment (flashnext-v3) summary:\n', summary('/tmp/ab_arc3-duck-anim-flashnext-v3')[:800])
if a and b:
    keys = sorted(set(a)|set(b))
    print(f'\n{"game":10s} {"v2-ctrl":>8s} {"v3-exp":>8s}  delta')
    ta=tb=0; n=0
    for k in keys:
        va, vb = a.get(k), b.get(k)
        va = va.get('score') if isinstance(va,dict) else va
        vb = vb.get('score') if isinstance(vb,dict) else vb
        if va is None and vb is None: continue
        d_ = (vb or 0)-(va or 0)
        print(f'{k:10s} {va if va is not None else "-":>8} {vb if vb is not None else "-":>8}  {d_:+.2f}')
        ta+=va or 0; tb+=vb or 0; n+=1
    if n: print(f'\nmean: v2 {ta/n:.2f} vs v3 {tb/n:.2f}  (delta {tb/n-ta/n:+.2f})')
else:
    print('\n(no score.json yet — kernels still running or output not yet served)')
PY
