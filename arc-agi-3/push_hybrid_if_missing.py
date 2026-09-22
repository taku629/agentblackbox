"""Push prepared kernel variants that do not exist on Kaggle yet.

Idempotent: once a kernel exists (queued, running, or complete) it is
skipped. A push while both batch-GPU slots are occupied is rejected by
Kaggle ("Maximum batch GPU session count") -- logged and treated as a
normal retry-later outcome, so this is safe to run on a schedule.

Usage: .venv/bin/python push_hybrid_if_missing.py
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# (slug, kernel dir under repo root). Pushed in order — each push claims one
# GPU slot, so earlier entries win when only one slot is free.
KERNELS = [
    ("takumuhata/arc3-duck-anim-flashnext", REPO / "submit_ag3_hybrid"),
    # NOTE: the patched-27B agent runs as v2 on the existing
    # takumuhata/arc3-duck-qwen3-8-27b kernel (dataset repoint in
    # submit_ag3_duck) -- no separate -patched kernel, to avoid spending
    # a batch-GPU slot on a duplicate experiment.
    ("takumuhata/arc3-duck-anim-v2", REPO / "submit_ag3_anim_v2"),
]

KAGGLE = [sys.executable, "-m", "kaggle"]


def main():
    for slug, kdir in KERNELS:
        if not kdir.is_dir():
            print(f"{slug}: {kdir} not found, skipping")
            continue
        st = subprocess.run(KAGGLE + ["kernels", "status", slug],
                            capture_output=True, text=True, timeout=120)
        if st.returncode == 0:
            print(f"{slug} already exists: {st.stdout.strip()[-120:]}")
            continue
        print(f"{slug} missing -> pushing {kdir}")
        r = subprocess.run(KAGGLE + ["kernels", "push", "-p", str(kdir)],
                           capture_output=True, text=True, timeout=300)
        print((r.stdout + r.stderr).strip()[-600:])


def _push_pending():
    """Process the pending kernel-version push queue (see push_pending.py)."""
    pending = HERE / "pending_pushes.json"
    if not pending.is_file() or not json.loads(pending.read_text() or "{}"):
        return
    subprocess.run([sys.executable, str(HERE / "push_pending.py")],
                   capture_output=False, timeout=600)


if __name__ == "__main__":
    main()
    _push_pending()
