"""Push the anim-hybrid kernel iff it does not exist on Kaggle yet.

Idempotent: once `takumuhata/arc3-duck-anim-flashnext` exists (queued,
running, or complete) this is a no-op. A push while both batch-GPU slots
are occupied is rejected by Kaggle ("Maximum batch GPU session count") --
that is logged and treated as a normal retry-later outcome.

Usage: .venv/bin/python push_hybrid_if_missing.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HYBRID_DIR = REPO / "submit_ag3_hybrid"
SLUG = "takumuhata/arc3-duck-anim-flashnext"

KAGGLE = [sys.executable, "-m", "kaggle"]


def main():
    st = subprocess.run(KAGGLE + ["kernels", "status", SLUG],
                        capture_output=True, text=True, timeout=120)
    if st.returncode == 0:
        print(f"{SLUG} already exists: {st.stdout.strip()[-120:]}")
        return
    print(f"{SLUG} missing -> pushing {HYBRID_DIR}")
    r = subprocess.run(KAGGLE + ["kernels", "push", "-p", str(HYBRID_DIR)],
                       capture_output=True, text=True, timeout=300)
    print((r.stdout + r.stderr).strip()[-600:])


if __name__ == "__main__":
    main()
