# Concurrency and Crash Test Report

Date: 2026-09-21

## Workloads

| Scenario | Workload | Assertion |
|---|---:|---|
| Shared database, thread-local connections | 8 writers × 150 events | 1,200/1,200 events and 8 successful sessions |
| Async decorated sessions | 12 simultaneous tasks | 12 distinct active session contexts |
| Timestamp collision | tool → LLM → tool at one fixed timestamp | exact insertion order |
| Duplicate delivery | same event twice, then same ID in another type | one globally identified event |
| Interrupted transaction | event followed by `KeyboardInterrupt` | zero events committed |
| Hard process exit | committed event + uncommitted event + `os._exit(9)` | first retained, second absent, integrity `ok` |
| Partial session | running session with one event | stable output over repeated replay |

The tests use synthetic identifiers and content, temporary databases, no services, and no
credentials. The writer regression uses independent `BlackBox`/connection instances, which models
normal simultaneous agent sessions rather than unrealistically sharing one SQLite cursor.

## Storage protocol

SQLite WAL permits readers alongside a writer, while SQLite still serializes write transactions.
Each event operation now performs these statements on one connection before commit:

1. `INSERT OR IGNORE` the global event identity and sequence.
2. If newly registered, insert its typed payload.
3. Commit immediately, unless inside `SQLiteStorage.transaction()`.

Consequently a crash cannot expose a registry entry without its payload (or vice versa). The
30-second busy timeout converts normal short writer contention into waiting rather than transient
`database is locked` failures. A bulk transaction greatly improves throughput but deliberately
increases the amount rolled back on termination; callers choose that atomicity/durability boundary.

## Ordering contract

New events order by timestamp and then committed registration sequence. IDs provide stable ordering
for legacy rows where original cross-table insertion order was never stored. This provides repeated
deterministic replay, not a claim that wall clocks across independent hosts are causally ordered.

## Remaining limits

There is no distributed sequence across multiple database files or remote hosts. SQLite allows one
active writer, so very long transactions can delay peers until the busy timeout. The recorder does
not mark abandoned `running` sessions as crashed automatically because doing so safely requires a
lease/heartbeat and a caller-defined expiry policy. Remote ingestion has no durable outbox,
acknowledgement replay, or server-side ordering guarantee.
