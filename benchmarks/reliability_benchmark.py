"""Bounded synthetic benchmark; run as a module from the repository root."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agentblackbox.models import Session, ToolCall
from agentblackbox.storage import SQLiteStorage


def recording_case(path: Path, events: int) -> dict:
    store = SQLiteStorage(path)
    store.create_session(Session("recording", "benchmark", time.time_ns()))
    tracemalloc.start()
    started = time.perf_counter()
    with store.transaction():
        for index in range(events):
            store.insert_tool_call(
                ToolCall(f"event-{index}", "recording", index, "synthetic", {"n": index}, "ok", 0)
            )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    size = sum(candidate.stat().st_size for candidate in path.parent.glob(path.name + "*"))
    tracemalloc.start()
    started = time.perf_counter()
    replayed = store.get_events("recording")
    replay_seconds = time.perf_counter() - started
    _, replay_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(replayed) == events
    return {
        "events": events,
        "record_seconds": round(elapsed, 4),
        "events_per_second": round(events / elapsed, 1),
        "peak_memory_mib": round(peak / 1024 / 1024, 2),
        "replay_seconds": round(replay_seconds, 4),
        "replay_peak_memory_mib": round(replay_peak / 1024 / 1024, 2),
        "database_mib": round(size / 1024 / 1024, 2),
    }


def concurrent_case(path: Path, workers: int, events_per_worker: int) -> dict:
    started = time.perf_counter()

    def writer(worker: int) -> None:
        store = SQLiteStorage(path)
        session_id = f"writer-{worker}"
        store.create_session(Session(session_id, "benchmark", time.time_ns()))
        with store.transaction():
            for index in range(events_per_worker):
                store.insert_tool_call(
                    ToolCall(f"{worker}-{index}", session_id, index, "synthetic", {}, None, 0)
                )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(writer, range(workers)))
    elapsed = time.perf_counter() - started
    total = workers * events_per_worker
    store = SQLiteStorage(path)
    count = store._conn().execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
    assert count == total
    return {
        "writers": workers,
        "events": total,
        "seconds": round(elapsed, 4),
        "events_per_second": round(total / elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--concurrent-events", type=int, default=2_000)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="agentblackbox-bench-") as directory:
        root = Path(directory)
        result = {
            "recording": [recording_case(root / f"record-{n}.db", n) for n in args.events],
            "concurrent": concurrent_case(root / "concurrent.db", 8, args.concurrent_events),
        }
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
