# Synthetic Benchmark Results

Date: 2026-09-21

## Method

Run from the repository root with:

```console
python -m benchmarks.reliability_benchmark
```

The standard-library-only benchmark creates temporary WAL databases, records small synthetic tool
events in one atomic bulk transaction, loads the complete merged replay into memory, checks counts,
and deletes the databases afterward. `tracemalloc` measures Python allocations (not SQLite native
page cache or interpreter baseline). Results are one run in the Codex Cloud container and are not a
cross-machine performance guarantee.

## Results

| Events | Record time | Throughput | Write peak | Replay time | Replay peak | DB files |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.1031 s | 9,698/s | 0.02 MiB | 0.0752 s | 1.12 MiB | 0.46 MiB |
| 10,000 | 0.7746 s | 12,910/s | 0.02 MiB | 0.6824 s | 11.21 MiB | 2.31 MiB |
| 100,000 | 8.3453 s | 11,983/s | 0.02 MiB | 6.8519 s | 113.78 MiB | 44.53 MiB |

The concurrent case (8 writers × 2,000 events) recorded all **16,000 events in 0.8284 s**, or
**19,315 events/s**. Throughput remained approximately linear through 100,000 events. Streaming
writes kept traced Python memory flat; replay intentionally materializes all events and therefore
scales linearly in memory.

## Interpretation

Profiling identified transaction commit frequency—not serialization or UUID generation—as the
actionable bulk-ingestion bottleneck. The new transaction API amortizes WAL synchronization while
preserving immediate commits as the compatible default. A separate 10,000-event A/B run without
`tracemalloc` measured **0.7442 s (13,437/s)** with the compatible per-event commits and **0.3065 s
(32,631/s)** in one transaction: a **2.43× throughput improvement**. The 100,000-event replay is practical but
its list-based API is not constant-memory; applications needing millions of events should use a
future cursor/streaming replay API. Database growth includes temporary WAL/SHM files at measurement
time and varies after checkpointing.
