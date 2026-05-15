# 🔲 AgentBlackBox

**A flight recorder for AI agents.**  
Record every LLM call, every tool call, every error.
Replay the full run later — locally, no account.

[![PyPI](https://img.shields.io/pypi/v/agentblackbox)](https://pypi.org/project/agentblackbox/)
![Python](https://img.shields.io/pypi/pyversions/agentblackbox)
[![CI](https://github.com/taku629/agentblackbox/actions/workflows/ci.yml/badge.svg)](https://github.com/taku629/agentblackbox/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Why I built this

I built a handful of AI agents this year — small ones for side projects,
research, scheduling. They worked great in demos and quietly broke in
production. Every time something went wrong, my first question was the
same: *what did the agent actually do?* Stdout couldn't answer. The
hosted observability tools either wanted a credit card before I could
see anything, or they came bundled with a framework I wasn't already
using.

So I built the smallest possible thing that records every LLM call, every
tool call, and every error, and lets me replay the whole run later.
Local SQLite. Zero dependencies. One decorator.

That's AgentBlackBox.

---

## How it compares

| | AgentBlackBox | LangSmith | Langfuse | Phoenix (Arize) |
|---|---|---|---|---|
| Install in 5 seconds (`pip install`) | ✅ | ❌ (account first) | ⚠️ (Postgres needed) | ⚠️ (multiple deps) |
| Zero core dependencies | ✅ | ❌ | ❌ | ❌ |
| Local-first (no account, no cloud) | ✅ | ❌ | ⚠️ (self-host option) | ✅ |
| Replay-as-test (`MockBlackBox`) | ✅ | ❌ | ❌ | ❌ |
| Hosted Cloud option | 🔜 waitlist | ✅ | ✅ | ✅ |
| Built-in cost tracking (20+ models) | ✅ | ✅ | ✅ | ⚠️ |
| MIT license | ✅ | ❌ proprietary | ✅ | ✅ Elastic |

**TL;DR**: if you want a hosted dashboard with a polished UI and a credit
card form, use LangSmith or Langfuse — they're great. AgentBlackBox is
for the moment when you just want to know what your agent did *right
now*, on your laptop, before deciding whether to commit to a platform.

---

## Why local-first

Three reasons, in order of importance:

1. **Privacy by default.** Your prompts and tool results are usually the
   most sensitive thing in your app. Recording them to your own SQLite
   file means they never leave your machine unless you explicitly send
   them somewhere.
2. **No vendor lock-in.** The recording format is documented and exportable.
   You can move to LangSmith / Langfuse / your own warehouse anytime.
3. **It works offline.** Train rides, planes, customer SCIFs, air-gapped
   evals — observability shouldn't require an internet connection.

---

## Install

```bash
pip install agentblackbox                 # core only (zero dependencies)
pip install agentblackbox[dashboard]      # + web UI
```

Requires Python 3.10+.

---

## Quickstart

```python
from agentblackbox import BlackBox

# Drop-in decorator — existing code unchanged
@BlackBox.record(agent_name="researcher")
def run_agent(task: str):
    # your existing agent code here
    ...

run_agent("Summarize today's AI news")

# See what happened
sessions = BlackBox.list_sessions()
BlackBox.replay(sessions[0].session_id)
```

That's it. Every LLM call, tool use, cost, and error is now recorded locally.

## Cloud (Coming Soon)

A hosted version with team sharing, retention, alerting, and SSO is in
development. **It is not live yet.** If you want early access, sign up
for the waitlist on the [landing page](https://taku629.github.io/agentblackbox/)
and tell me what you'd want from it — I'm reading every response and the
first 50 signups get free beta access.

Until then, you can already self-host the cloud-mode dashboard yourself
on any server you control:

## Self-hosted remote mode

You can mirror recordings to your own AgentBlackBox dashboard while still keeping the
local SQLite log:

```python
from agentblackbox import BlackBox
from agentblackbox.remote import RemoteStorage

remote_store = RemoteStorage(
    api_key="abx_...",
    endpoint="https://your-agentblackbox.example.com",
)

with BlackBox.session("researcher", storage=remote_store) as bb:
    bb.record_tool_call("search", {"q": "ai evals"}, {"hits": 12}, 83.4)
```

To run the dashboard in authenticated cloud mode:

```bash
agentblackbox dashboard --cloud
```

---

## What gets recorded

| Event | Details |
|---|---|
| 🤖 **LLM call** | model, prompt, output, input/output tokens, cost, latency |
| 🔧 **Tool call** | name, arguments, return value, execution time |
| ❌ **Error** | type, message, full stack trace, timestamp |

All data is stored in a local SQLite file (`~/.agentblackbox/recordings.db`).  
Nothing is sent to any external server.

---

## Usage patterns

### Decorator
```python
@BlackBox.record(agent_name="coder")
def coding_agent(task):
    ...
```

### Context manager
```python
with BlackBox.session("planner") as bb:
    plan = agent.run(task)
    bb.record_tool_call("search", {"query": task}, result=plan)
```

### Manual recording
```python
with BlackBox.session("custom") as bb:
    bb.record_llm_call(
        model="gpt-4o",
        input_text="Summarize this",
        output_text="Here is a summary...",
        input_tokens=150,
        output_tokens=80,
        duration_ms=400.0,
    )
```

### OpenAI Agents SDK (auto-instrument)
```python
from agentblackbox.integrations import patch_openai_agents
patch_openai_agents()  # All agents recorded automatically
```

---

## Web Dashboard

```bash
agentblackbox dashboard
# → http://localhost:8765
```

- **Sessions** — all runs with status, cost, duration, auto-refreshes every 30s
- **Timeline** — step-by-step replay with expandable LLM inputs/outputs
- **Analytics** — daily cost trends, per-agent breakdown, model distribution

---

## CLI

```bash
agentblackbox sessions                    # list all sessions
agentblackbox replay <session_id>         # console replay
agentblackbox export <session_id>         # JSON export
agentblackbox dashboard --port 8765       # web UI
```

---

## Cost tracking

Supports 20+ models with automatic cost calculation:

| Provider | Models |
|---|---|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini, o3-mini |
| Anthropic | claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5, claude-3-5-sonnet, claude-3-opus, claude-3-haiku |

Unknown model names fall back to family-prefix pricing
(e.g. `claude-sonnet-*` → Sonnet rates) so new releases work without a code change.

---

## Roadmap

The library is intentionally small, but a few things are explicitly on
the table. Open an issue if any of these block you and I'll prioritize.

**v0.3 — privacy & safety**
- Redaction hooks: pluggable filter for sensitive substrings (API keys,
  PII, customer data) before recordings are written to disk.
- Configurable retention for the local SQLite store (e.g. "keep 30 days").
- Optional encryption-at-rest for the recordings DB.

**v0.4 — interoperability**
- OpenTelemetry OTLP exporter as an *optional* extra
  (`pip install agentblackbox[otel]`). Core stays zero-dep.
  Lets you start local and ship recordings to Honeycomb, Datadog,
  Grafana, or your own OTel stack without rewriting code.
- LlamaIndex auto-instrument
- CrewAI auto-instrument

**v0.5 — wider stack**
- AutoGen auto-instrument
- Node/TypeScript SDK (separate package, same wire format)
- Recording format spec — versioned, JSON-schema-validated.

**Cloud (separate product)**
- See [Cloud](#cloud-coming-soon) above. Beta is free; pricing TBD.

---

## License

MIT © 2026 Takumu Hata
