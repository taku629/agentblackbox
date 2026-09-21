"""DEPRECATED -- do not run.

This used to submit the OCEAN starter kernel unconditionally at ~00:10 UTC.
That burns the daily slot on a ~0.14-score kernel while the duck/LLM
candidates verify far higher (27B measured 4.79). It now delegates to the
score-aware watcher so a stale synced copy still does the right thing.

Equivalent direct call:
    .venv/bin/python submit_best.py --watch
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.execv(sys.executable, [sys.executable, str(HERE / "submit_best.py"), "--watch"])
