# 🔲 AgentBlackBox

**A flight recorder for AI agents.**  
Record every decision, tool call, and failure. Replay them later.

![PyPI](https://img.shields.io/pypi/v/agentblackbox)
![Python](https://img.shields.io/pypi/pyversions/agentblackbox)
![License](https://img.shields.io/badge/license-MIT-blue)
![Tests](https://img.shields.io/badge/tests-39%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-73%25-green)

---

## The problem

**72% of AI agent projects never reach production.**

Not because the agents are wrong — but because they're invisible.  
You can't debug what you can't see. You can't trust what you can't audit.

AgentBlackBox is the flight recorder your AI agents need.

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

## Remote / hosted mode

You can mirror recordings to a hosted AgentBlackBox dashboard while still keeping the
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
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o3-mini |
| Anthropic | claude-3-5-sonnet, claude-3-opus, claude-3-haiku, claude-3-5-haiku |

---

## License

MIT © 2026 Takumu Hata
