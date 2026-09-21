#!/usr/bin/env python
"""Push the hybrid (anim-solver + flash-next serving) kernel as soon as a GPU
session slot frees (limit is 2 batch GPU sessions; QUEUED counts).

Poll every 10 min. A slot frees only when a tracked kernel reaches a terminal
state. Keeps trying the push until it succeeds, then exits.
"""
import subprocess
import sys
import time
import datetime

PY = "/home/takumu/kaggle/arc-agi-3/.venv312/bin/python"
HYBRID_DIR = "/home/takumu/kaggle/submit_ag3_hybrid"
TRACKED = [
    "takumuhata/arc3-duck-qwen3-8-flash-next-nvfp4-mtp",
    "takumuhata/arc3-duck-flash-next-nvfp4-mtp-b",
    "takumuhata/arc3-duck-anim-flashnext",
]
TERMINAL = ("COMPLETE", "ERROR", "CANCELLED", "ACKNOWLEDGED")


def log(msg):
    print(f"{datetime.datetime.utcnow().isoformat()} {msg}", flush=True)


def status(slug):
    try:
        out = subprocess.run(
            ["kaggle", "kernels", "status", slug],
            capture_output=True, text=True, timeout=120,
        ).stdout
        for s in ("QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCELLED"):
            if s in out:
                return s
        return out.strip()[-80:]
    except Exception as e:
        return f"ERR {e}"


def push():
    r = subprocess.run(
        [PY, "-m", "kaggle", "kernels", "push", "-p", HYBRID_DIR],
        capture_output=True, text=True, timeout=300,
    )
    return r.stdout + r.stderr


log("monitor started")
while True:
    sts = {k: status(k) for k in TRACKED}
    log(" | ".join(f"{k.split('/')[-1]}={v}" for k, v in sts.items()))
    if sts.get(TRACKED[2]) in TERMINAL:
        log("hybrid reached terminal state; exiting")
        break
    active = sum(1 for k in TRACKED[:2] if sts.get(k) not in TERMINAL)
    if active < 2:
        log("GPU slot free — pushing hybrid")
        out = push()
        log(f"push result: {out.strip()[-400:]}")
        if "successfully pushed" in out:
            log("hybrid pushed; exiting")
            break
    time.sleep(600)
