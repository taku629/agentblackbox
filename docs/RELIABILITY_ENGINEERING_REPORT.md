# Reliability Engineering Report

Date: 2026-09-21

## Scope and baseline

The review covered the recorder lifecycle and context variable, SQLite storage, deterministic
replay/export, remote forwarding, sharing, integrations, and the original test suite. The baseline
command `python -m pytest -q` passed **107 tests in 1.06 s** before changes.

## Confirmed defects and fixes

1. **Async decorators ended recording before the coroutine ran.** `BlackBox.record` wrapped every
   function synchronously, returning a coroutine after its context had closed. A reproducer observed
   `BlackBox.current() is None` inside an awaited decorated function. Coroutine functions now receive
   an async wrapper whose context spans the await. A 12-task regression also proves context isolation.
2. **Equal-timestamp events replayed nondeterministically.** Events lived in three tables and replay
   sorted only by nanosecond timestamp. A synthetic fixed clock changed cross-type insertion order.
   The storage now registers each event in an `event_order` table in the same transaction and replay
   uses `(timestamp, insertion sequence)`. Legacy events absent from that table remain readable with
   a stable `(timestamp, kind, id)` fallback.
3. **Duplicate delivery raised or could create ambiguous cross-type IDs.** The global event registry
   makes an event ID idempotent across all event types; the payload and registry write share one
   SQLite transaction.
4. **High-volume ingestion paid one durable commit per event.** The additive
   `SQLiteStorage.transaction()` API groups writes atomically without changing default durability.
   Nested scopes are supported and exceptions, including `KeyboardInterrupt`, roll back the outer
   unit. This is the optimization used for bounded bulk workloads.
5. **Credential-shaped values were stored verbatim.** Recording now recursively redacts sensitive
   keys, bearer credentials, `sk_...` and `abx_...` tokens in tool arguments/results, metadata, LLM
   text, exception messages, and tracebacks. The sanitizer is cycle-safe, depth-bounded, JSON-safe,
   and uses only the standard library.
6. **Lifecycle/cost mutation was unsynchronized.** A re-entrant state lock now protects start, stop,
   and cost accumulation; `start` is idempotent while active and `stop` after completion is a no-op.
7. **SQLite connections had weak contention configuration.** Every thread-local connection now
   enables foreign keys, WAL, `synchronous=NORMAL`, and a 30-second busy timeout. This matters
   because pragmas such as `foreign_keys` are per connection.

## Crash and adversarial behavior

An abruptly terminated subprocess retained its committed event, discarded an open multi-event
transaction, and returned `ok` from `PRAGMA integrity_check`. Circular nested tool input no longer
crashes serialization; it is represented as `[CYCLE]`. Partial/running sessions replay repeatedly
with stable output. Invalid JSON stored by external database corruption raises `JSONDecodeError`
instead of being silently accepted. Missing replay items retain the existing explicit `None`
exhaustion behavior, and failed tools retain their recorded error.

## Compatibility and limitations

All existing public call signatures remain valid. Redaction intentionally changes persisted secret
values to `[REDACTED]`; callers needing raw secrets should not use a trace recorder for that data.
Heuristic redaction cannot recognize every opaque secret when neither its key nor value has a known
credential shape. Existing pre-upgrade rows are not retroactively redacted or assigned their true
historical cross-table order. SQLite serializes writers, and a process killed after a default
single-event commit naturally retains that event. Remote transport remains best-effort and
synchronous: network errors are dropped rather than queued for eventual delivery, so the local WAL
database is the source of truth.

## Verification

The final suite contains regressions for concurrent writers, async context propagation, ordering,
duplicates, rollback, abrupt termination, privacy, partial sessions, repeated replay, and corruption.
See `CONCURRENCY_TEST_REPORT.md` and `BENCHMARK_RESULTS.md` for workload details.
