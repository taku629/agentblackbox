# 🔲 AgentBlackBox

**A flight recorder for AI agents.**
Record every decision, tool call, and failure. Replay them later.

[![PyPI](https://img.shields.io/pypi/v/agentblackbox)](https://pypi.org/project/agentblackbox/)
[![Python](https://img.shields.io/pypi/pyversions/agentblackbox)](https://pypi.org/project/agentblackbox/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/taku629/agentblackbox/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-97%20passed-brightgreen)](https://github.com/taku629/agentblackbox/actions)
[![GitHub](https://img.shields.io/badge/GitHub-taku629%2Fagentblackbox-181717?logo=github)](https://github.com/taku629/agentblackbox)

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

@BlackBox.record(agent_name="researcher")
def run_agent(task: str):
    bb = BlackBox.current()
    bb.record_llm_call(
        model="gpt-4o",
        input_text=task,
        output_text="Here are the findings...",
        input_tokens=150,
        output_tokens=320,
    )

run_agent("Summarize AI news")

# Replay what happened
sessions = BlackBox.list_sessions()
```

---

## What gets recorded

| Event | Details |
|---|---|
| 🤖 LLM call | model, prompt, output, tokens, cost, latency |
| 🔧 Tool call | name, arguments, return value, execution time |
| ❌ Error | type, message, full stack trace |

All data stored in local SQLite. **Nothing sent externally.**

---

## Usage patterns

```python
# Decorator
@BlackBox.record(agent_name="coder")
def coding_agent(task): ...

# Context manager
with BlackBox.session("planner") as bb:
    bb.record_tool_call("search", {"q": task}, result)

# One-line OpenAI Agents SDK instrumentation
from agentblackbox.integrations import patch_openai_agents
patch_openai_agents()
```

---

## Privacy & Security

```python
BlackBox.configure(
    masking=True,
    mask_patterns=["credit_card", "email", "api_key"],
)
```

Built-in patterns: credit card, email, phone (JP/US), IPv4, API key, AWS key, JWT, SSN.
Custom patterns: `custom_mask_patterns={"MY_ID": r"EMP-\d{6}"}`.
Zero overhead when disabled (default).

---

## Web Dashboard

```bash
agentblackbox dashboard   # → http://localhost:8765
```

- Sessions list with real-time filter and auto-refresh
- Animated timeline replay (▶ button)
- Cost analytics with Chart.js

---

## CLI

```bash
agentblackbox sessions [--agent NAME] [--status STATUS]
agentblackbox replay <session_id>
agentblackbox export <session_id> out.json
agentblackbox dashboard [--port 8765]
```

---

## 20+ models for cost tracking

OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek, xAI.

---

## License

MIT © 2026 Takumu Hata
