"""Poll kernel + submission status every 5 min, log transitions to
monitor_kernels.log. Runs up to 24h."""
import subprocess, time, logging
from datetime import datetime

from pathlib import Path
logging.basicConfig(filename=str(Path(__file__).resolve().parent/"monitor_kernels.log"),
                    level=logging.INFO, format="%(asctime)s %(message)s",
                    force=True)
KERNELS = [
    "takumuhata/arc3-duck-qwen3-14b-awq-t4",
    "takumuhata/arc3-duck-qwen3-8-27b",
    "takumuhata/forge-pathfinder-bfs-agent",
]
COMPS = ["arc-prize-2026-arc-agi-3", "arc-prize-2026-arc-agi-2"]

last = {}
for _ in range(288):
    for k in KERNELS:
        try:
            out = subprocess.run(["kaggle", "kernels", "status", k],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            st = out.split('"')[-2] if '"' in out else out
        except Exception as e:
            st = f"ERR {e}"
        key = ("k", k)
        if last.get(key) != st:
            logging.info(f"{k}: {last.get(key)} -> {st}")
            last[key] = st
    for c in COMPS:
        try:
            out = subprocess.run(["kaggle", "competitions", "submissions", c],
                                 capture_output=True, text=True, timeout=60).stdout
            lines = [l for l in out.splitlines() if "submission" in l and ("COMPLETE" in l or "PENDING" in l or "ERROR" in l)]
            st = " | ".join(l.split()[-3:] for l in lines[:3])
        except Exception as e:
            st = f"ERR {e}"
        key = ("c", c)
        if last.get(key) != st:
            logging.info(f"{c} submissions: {st}")
            last[key] = st
    time.sleep(300)
logging.info("monitor done")
