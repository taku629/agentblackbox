"""Push pending kernel-version bumps when a batch-GPU slot is free.

``pending_pushes.json`` maps kernel slug -> local submit dir. Each tick pops
entries whose push succeeds ("Kernel version N successfully pushed") and
records the new version in ``kernel_versions.json`` so ``submit_best.py``
submits the fresh code. A push rejected for GPU saturation
("Maximum batch GPU session count") or any other error stays queued for the
next tick. Safe to run repeatedly: empty queue exits immediately.

Usage: .venv/bin/python push_pending.py
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PENDING_FILE = HERE / "pending_pushes.json"
VERMAP_FILE = HERE / "kernel_versions.json"

KAGGLE = [sys.executable, "-m", "kaggle"]


def main():
    if not PENDING_FILE.is_file():
        print("no pending_pushes.json; nothing to do")
        return
    pending = json.loads(PENDING_FILE.read_text())
    if not pending:
        print("pending push queue is empty")
        return

    # GPU pushes are confined to Saturdays (UTC): the weekly GPU budget is
    # shared with other competitions and Sundays are reserved for RSNA.
    # Set TAAF_PUSH_ANY_DAY=1 to bypass the gate.
    if (
        os.environ.get("TAAF_PUSH_ANY_DAY") != "1"
        and datetime.now(timezone.utc).weekday() != 5
    ):
        print("outside Saturday(UTC) push window; queue left for Saturday")
        return

    vermap = json.loads(VERMAP_FILE.read_text()) if VERMAP_FILE.is_file() else {}
    remaining = dict(pending)
    for slug, rel_dir in pending.items():
        kdir = REPO / rel_dir
        print(f"pushing {slug} from {kdir}")
        r = subprocess.run(KAGGLE + ["kernels", "push", "-p", str(kdir)],
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr).strip()
        print(out[-400:])
        m = re.search(r"Kernel version (\d+) successfully pushed", out)
        if m:
            vermap[slug] = int(m.group(1))
            del remaining[slug]
            print(f"{slug} -> version {m.group(1)} recorded")
        else:
            print(f"{slug} push not accepted; will retry next tick")

    PENDING_FILE.write_text(json.dumps(remaining, indent=2) + "\n")
    VERMAP_FILE.write_text(json.dumps(vermap, indent=2) + "\n")


if __name__ == "__main__":
    main()
