"""Score-aware watcher (kept as a stable entry point).

Delegates to submit_best.py --watch, which is idempotent and never
double-spends the daily slot (it checks the competition's own
numAllowedNow counter first). Safe to run alongside one-shot
submit_best.py invocations.

Equivalent direct call:
    .venv/bin/python submit_best.py --watch
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.execv(sys.executable, [sys.executable, str(HERE / "submit_best.py"), "--watch"])
