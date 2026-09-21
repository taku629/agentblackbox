"""Submit OCEAN kernel v7 to ARC-AGI-3 after the daily submission slot resets
(00:00 UTC). Sleeps until ~00:10 UTC, then retries every 15 min for 3h.
Logs to watch_submit.log."""
import sys, time, logging
from datetime import datetime, timezone, timedelta
sys.path.insert(0, "/home/takumu/kaggle/arc-agi-3")
from kaggle.api.kaggle_api_extended import KaggleApi

logging.basicConfig(filename="/home/takumu/kaggle/arc-agi-3/watch_submit.log",
                    level=logging.INFO, format="%(asctime)s %(message)s",
                    force=True)
api = KaggleApi(); api.authenticate()

now = datetime.now(timezone.utc)
target = (now + timedelta(days=1)).replace(hour=0, minute=10,
                                         second=0, microsecond=0)
wait = max(0, (target - now).total_seconds())
logging.info(f"sleeping {wait/3600:.1f}h until {target}")
time.sleep(wait)

for attempt in range(12):
    try:
        r = api.competition_submit_code(
            file_name="submission.parquet",
            message="OCEAN v15: induction + effective-object click dwelling + level-transition fix + safe-color hazard gate + trap exclusion + collect-color targeting + consumed-object unblock + perimeter touch-targets + traversable goal refs + continuity-gated tracking + perimeter poke + eff-press-cap + contact-kill hazard attribution",
            competition="arc-prize-2026-arc-agi-3",
            kernel="takumuhata/arc-prize-2026-arc-agi-3-starter",
            kernel_version=15)
        logging.info(f"OCEAN submit OK: {r}")
        break
    except Exception as e:
        logging.info(f"submit failed (attempt {attempt+1}): {str(e)[:200]}")
        time.sleep(900)
logging.info("watcher done")
