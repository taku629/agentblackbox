#!/bin/bash
# One-shot for the public M2 D' fork (submit_ag3_m2_dprime): push once the weekly GPU
# quota allows it, then submit the completed version once. Idempotent; state in
# arc-agi-3/fork_state/. Meant for cron (every 20 min); remove the cron line when done.
set -u
export PATH="$HOME/.local/bin:$PATH"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
S="$REPO/arc-agi-3/fork_state"; mkdir -p "$S"
SLUG=takumuhata/arc3-m2-dprime
COMP=arc-prize-2026-arc-agi-3
log() { echo "$(date -u +%FT%TZ) $*" >> "$S/log.txt"; }
[ -f "$S/submitted" ] && exit 0
if [ ! -f "$S/pushed" ]; then
    out=$(timeout 300 kaggle kernels push -p "$REPO/submit_ag3_m2_dprime" 2>&1)
    log "push: ${out: -200}"
    ver=$(echo "$out" | sed -n 's/.*Kernel version \([0-9]*\) successfully pushed.*/\1/p')
    [ -n "$ver" ] && echo "$ver" > "$S/pushed"
    exit 0
fi
st=$(timeout 60 kaggle kernels status "$SLUG" 2>&1 | tail -1)
case "$st" in
    *COMPLETE*)
        out=$(timeout 300 kaggle competitions submit "$COMP" -k "$SLUG" -v "$(cat "$S/pushed")" \
              -f submission.parquet -m "public M2 solution fork (dfranzen) + D' slot priority (shiiin9), unchanged" 2>&1)
        log "submit: ${out: -300}"
        echo "$out" | grep -qi "success" && touch "$S/submitted" ;;
    *ERROR*|*CANCEL*) log "kernel ended: $st"; touch "$S/submitted" ;;   # stop; needs a person
    *) : ;;
esac
