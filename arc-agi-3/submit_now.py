"""Submit the current OCEAN kernel to ARC-AGI-3 immediately.

Use in cloud sessions (which suspend when idle — a sleep-until-00:10-UTC
watcher is unreliable there). Run once any time after the daily submission
allowance resets (~00:00 UTC); if the slot is already consumed it exits.

Usage: .venv/bin/python submit_now.py [kernel_version]
"""
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

kv = int(sys.argv[1]) if len(sys.argv) > 1 else 15
api = KaggleApi()
api.authenticate()
r = api.competition_submit_code(
    file_name="submission.parquet",
    message=f"OCEAN v{kv}: online induction + click dwelling + press-cap + contact-kill attribution",
    competition="arc-prize-2026-arc-agi-3",
    kernel="takumuhata/arc-prize-2026-arc-agi-3-starter",
    kernel_version=kv)
print("SUBMITTED:", r)
