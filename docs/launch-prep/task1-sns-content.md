# Task 1: SNS / HN / Reddit コンテンツ一式

---

## X 日本語

### JP-1
```
AIエージェントが本番で何をやらかしたか、わかる？

LLM呼び出し・ツール実行・コスト・エラーを全部ローカルに記録するOSSを作った。
デコレータ1行でOK、外部送信ゼロ、PIIは自動マスキング。

pip install agentblackbox
@BlackBox.record(agent_name="my-agent")
def run(): ...

{GITHUB_URL}

#AIエージェント #LLM #Python #OSS #可観測性
```

### JP-2
```
「本番に出したいけど、何してるかわからなくて怖い」

AIエージェントあるある、解決した。

AgentBlackBox はエージェントのフライトレコーダー。
デコレータ1行で全イベントを記録 → ダッシュボードでタイムライン再生。
SQLiteローカル保存のみ、データ外部送信なし。

pip install agentblackbox[dashboard]

{GITHUB_URL}

#AIエージェント #LangChain #OpenAI #DevTools #Python
```

### JP-3
```
LangSmithやLangfuse、便利だけど「外部にデータ送りたくない」案件ない？

AgentBlackBox はローカルSQLiteだけで完結するエージェント可観測性ツール。
カード番号・メール・APIキーのPII自動マスキング付き。

@BlackBox.record(agent_name="secure-agent")

FastAPI + htmx のダッシュボードで再生も可能。

{GITHUB_URL}

#プライバシー #AIエージェント #OSS #セキュリティ #Python
```

### JP-4
```
AIエージェントのコスト、ちゃんと把握できてる？

AgentBlackBox を入れると毎回のLLM呼び出しコストが全部記録される。
gpt-4o, claude-3-5-sonnet, o3-mini など20+モデル対応。

pip install agentblackbox
agentblackbox dashboard  # → localhost:8765 でコスト可視化

{GITHUB_URL}

#LLMコスト #AIエージェント #Python #OSS #コスト管理
```

### JP-5
```
エラーが出たとき「どのLLM呼び出しのせいか」追えてる？

AgentBlackBox はAIエージェントのブラックボックス録画ツール。
デコレータ1行で全ステップを記録。エラー時はスタックトレース付きで残る。

from agentblackbox import BlackBox
@BlackBox.record(agent_name="debuggable")

GitHubスター押してもらえると励みになります→ {GITHUB_URL}

#デバッグ #AIエージェント #LLM #Python #OSS
```

---

## X English

### EN-1
```
Your AI agent is in production. Do you know what it's actually doing?

Built AgentBlackBox — a flight recorder for AI agents.
One decorator. Every LLM call, tool use, cost, and error recorded locally.
Zero external data transmission. PII auto-masked.

pip install agentblackbox
@BlackBox.record(agent_name="my-agent")

{GITHUB_URL}

#AIAgents #LLM #Python #OSS #Observability
```

### EN-2
```
"I'd ship this agent but I have no idea what it does in prod."

Solved that with AgentBlackBox.

→ @BlackBox.record decorator — one line
→ Timeline replay in a local web dashboard
→ Local SQLite only — no cloud, no vendor lock-in
→ PII masking built in

pip install agentblackbox[dashboard]
{GITHUB_URL}

#LLMOps #AIAgents #Python #DevTools #OpenSource
```

### EN-3
```
LangSmith and Langfuse are great — but sometimes you can't send data to a 3rd party.

AgentBlackBox is a local-only flight recorder for AI agents.
SQLite. No network calls. No vendor account needed.

pip install agentblackbox
agentblackbox dashboard  # instant web UI at localhost:8765

Star if this is useful → {GITHUB_URL}

#AIAgents #Privacy #LLMOps #Python #OSS
```

### EN-4
```
AI agent debugging tip: add this before you ship.

from agentblackbox import BlackBox

@BlackBox.record(agent_name="prod-agent")
def run_agent(task):
    ...  # your existing code, unchanged

Now every LLM call, tool use, cost, and error is recorded.
Replay later in the dashboard. No code changes needed beyond the decorator.

{GITHUB_URL}

#AIAgents #LangChain #Python #DevTools #Observability
```

### EN-5
```
Built an open-source "black box recorder" for AI agents.

Features:
✅ 1-line decorator
✅ LLM cost tracking (20+ models)
✅ PII auto-masking (cards, emails, API keys)
✅ Local SQLite — zero external calls
✅ Web dashboard with timeline replay

Trying to make AI agents safe to ship to production.

{GITHUB_URL}

#OSS #AIAgents #LLMOps #Python #BuildInPublic
```

---

## Show HN

### Show HN Pattern 1 — Problem-first

**Title:** Show HN: AgentBlackBox – a local flight recorder for AI agents (one decorator, zero external calls)

**Body:**

I've been building AI agents for a year, and the scariest moment is always the same: the first deploy to production. Not because the logic is wrong — but because I have no idea what the agent actually does once it's running.

LangSmith and Langfuse are useful, but they require sending your data to an external server, creating an account, and learning a new platform. For internal tools, regulated industries, or anything that touches sensitive data (PII, proprietary prompts), that's a non-starter.

So I built AgentBlackBox: a flight recorder that stores everything locally in SQLite.

**How it works:**

```python
from agentblackbox import BlackBox

@BlackBox.record(agent_name="my-agent")
def run_agent(task: str):
    # your existing code, completely unchanged
    ...
```

That's it. Every LLM call (model, tokens, cost, latency), every tool invocation, and every error is recorded. You can replay sessions in a web dashboard or via the CLI.

**What makes it different from LangSmith/Langfuse:**

- No account, no API key, no vendor — just `pip install agentblackbox`
- Data never leaves your machine (local SQLite only)
- PII is automatically masked (card numbers, emails, API keys) before storage
- Works with any framework: vanilla OpenAI, LangChain, LangGraph, OpenAI Agents SDK
- FastAPI + htmx dashboard starts with one command: `agentblackbox dashboard`

**Core features:**

- `@BlackBox.record` decorator — drop-in, no code changes needed
- Context manager and manual recording API for fine-grained control
- Cost tracking for 20+ models (OpenAI, Anthropic)
- Web dashboard with timeline replay and analytics
- Auto-instrumentation for OpenAI Agents SDK via `patch_openai_agents()`

```bash
pip install agentblackbox
```

{GITHUB_URL}

Feedback welcome, especially from people running agents in regulated environments or with strict data-residency requirements.

---

### Show HN Pattern 2 — Technical-focus

**Title:** Show HN: AgentBlackBox – SQLite-backed observability for AI agents, with PII masking

**Body:**

After shipping a few LLM-based agents to production I kept hitting the same wall: something goes wrong, costs spike, or the agent just gives bad output — and I have no trace of what happened.

The existing tools (LangSmith, Langfuse, Arize) are well-built but they're SaaS products. For a lot of real-world use cases — on-prem deployments, HIPAA environments, companies with strict data egress policies — sending your prompts and completions to a third party is simply not allowed.

AgentBlackBox takes a different approach: everything is stored locally in SQLite. There's no cloud component, no account, no telemetry.

**Technical design:**

- Storage: SQLite via SQLAlchemy, stored at `~/.agentblackbox/recordings.db`
- PII masking: regex-based scrubbing of credit cards, emails, phone numbers, API keys applied before write
- Dashboard: FastAPI backend serving htmx-enhanced HTML (no JS framework, fast cold start)
- Cost engine: static pricing table for 20+ models, extensible via config
- Framework hooks: context vars for async safety, `patch_openai_agents()` for SDK-level auto-instrumentation

**Usage:**

```python
# Option 1: decorator
@BlackBox.record(agent_name="researcher")
def run(task): ...

# Option 2: context manager
with BlackBox.session("planner") as bb:
    result = agent.run(task)
    bb.record_tool_call("search", {"q": task}, result=result)

# Option 3: OpenAI Agents SDK — zero code change
from agentblackbox.integrations import patch_openai_agents
patch_openai_agents()
```

```bash
pip install agentblackbox[dashboard]
agentblackbox dashboard  # → http://localhost:8765
```

**What I'd love feedback on:**

- The PII masking approach (regex vs. a small local model for better recall)
- Whether a "hybrid mode" (write to both SQLite and a cloud endpoint) would be useful
- Performance at high event volume (I've tested up to ~500 events/session, haven't benchmarked beyond that)

{GITHUB_URL}

---

### Show HN Pattern 3 — Story-driven

**Title:** Show HN: AgentBlackBox – I wanted LangSmith but without sending data to LangChain

**Body:**

Last year I was building an internal AI agent for a client in a regulated industry. The agent handled sensitive documents — nothing that could leave the network. I wanted observability. I looked at LangSmith, Langfuse, and similar tools. All of them require sending your prompts and responses to their servers.

I ended up logging to a JSON file and writing a script to parse it. It worked, but it was brittle and I had to redo it for every project.

AgentBlackBox is the tool I wanted then: a drop-in flight recorder that stores everything locally and gives you a decent UI to replay sessions.

**The core pitch:**

You add one decorator. Everything — LLM calls with tokens and cost, tool invocations, errors with stack traces — is recorded to a local SQLite file. A FastAPI + htmx dashboard lets you replay any session step by step. PII (card numbers, emails, API keys) is masked before it hits the disk.

```python
pip install agentblackbox

from agentblackbox import BlackBox

@BlackBox.record(agent_name="doc-processor")
def process(doc: str) -> str:
    # no changes to your existing logic
    return agent.run(doc)
```

**How it compares:**

| | AgentBlackBox | LangSmith | Langfuse |
|---|---|---|---|
| Storage | Local SQLite | LangChain Cloud | Langfuse Cloud / self-host |
| Account required | No | Yes | Yes |
| PII masking | Built-in | No | No |
| Setup | `pip install` | API key + SDK | API key + SDK |
| Pricing | Free / OSS | Freemium | Freemium |

I'm planning to add a cloud mode (opt-in, for teams who want shared dashboards) but the local-only mode will always be free and open source.

```bash
pip install agentblackbox
```

{GITHUB_URL}

Happy to answer questions about the implementation. The SQLite schema and PII masking logic are both pretty simple if anyone wants to look under the hood.

---

## Reddit LangChain

### r/LangChain Pattern 1

**Title:** Built a local-only observability tool for LangChain agents — no account, no cloud, one decorator

**Body:**

Hey r/LangChain,

Sharing something I built after getting frustrated with the "send everything to a cloud service" requirement of most LLM observability tools.

**AgentBlackBox** is a local flight recorder for AI agents. Everything goes to SQLite on your machine. Nothing leaves. PII (emails, card numbers, API keys) is auto-masked before storage.

**Setup:**

```bash
pip install agentblackbox[dashboard]
```

**With LangChain:**

```python
from agentblackbox import BlackBox
from langchain.agents import AgentExecutor

@BlackBox.record(agent_name="lc-agent")
def run_agent(task: str):
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor.invoke({"input": task})

run_agent("Search for the latest AI news and summarize")
```

**Then launch the dashboard:**

```bash
agentblackbox dashboard
# → http://localhost:8765
```

You get a timeline of every LLM call (with token counts and cost), every tool invocation, every error, and you can replay any session step by step.

**What I'd love feedback on:**

- Does the decorator approach work well with your LangChain setups, or do you need something deeper (callback-based, etc.)?
- Would a native LangChain callback handler be more useful than the decorator?
- Any edge cases in LangChain agent patterns that would break this?

**Who this is for:**

- People who can't use LangSmith due to data-residency requirements
- Anyone who wants zero-friction observability without setting up accounts
- Devs debugging agent behavior locally before shipping

{GITHUB_URL} — MIT licensed, feedback very welcome!

---

### r/LangChain Pattern 2

**Title:** I was using print() to debug my LangChain agent. Built something better (local, zero external calls)

**Body:**

Honest confession: I spent two weeks debugging a LangChain agent by printing intermediate results to the terminal and grepping through logs. It worked but it was terrible.

I wanted LangSmith-style observability but without the "send your prompts to us" part. So I built **AgentBlackBox**, an open source local flight recorder.

**Quick demo with LangChain:**

```python
from agentblackbox import BlackBox

# Step 1: add decorator
@BlackBox.record(agent_name="research-agent")
def run_research(query: str):
    chain = build_your_langchain_chain()
    return chain.invoke({"question": query})

# Step 2: run your agent as usual
run_research("What are the risks of deploying LLMs in healthcare?")

# Step 3: open dashboard
# agentblackbox dashboard → localhost:8765
# Click the session → timeline replay ▶
```

**What you see in the dashboard:**

- Each LLM call: model, full prompt, full response, tokens, cost, latency
- Each tool call: name, inputs, outputs, duration
- Errors: type, message, stack trace
- Session summary: total cost, total duration, event count

**The thing I'm most unsure about:** currently the PII masking is regex-based. It catches obvious stuff (emails, card numbers, API keys starting with `sk-`) but won't catch things like a person's name embedded in a prompt. Would love to hear how you all handle this in production.

GitHub (MIT): {GITHUB_URL}

---

## Reddit AI_Agents

### r/AI_Agents Pattern 1

**Title:** Show off: built an open-source "black box recorder" for AI agents — records everything locally, PII masked, web dashboard for replay

**Body:**

Hey all, sharing a side project that turned into something I use daily.

**The problem:** I'm running AI agents in prod and I kept asking "what did this agent actually do?" after something went wrong. Costs would be higher than expected, outputs would be weird, errors would appear — but I had no trace.

Existing tools (LangSmith, Langfuse, Arize) are great but require external accounts and sending your data to their servers. For my use cases (client work, sensitive data) that wasn't always an option.

**What I built:** AgentBlackBox — a flight recorder that stores agent activity in local SQLite. No cloud, no account, no telemetry.

**Core workflow:**

```python
from agentblackbox import BlackBox

# Add one decorator to your agent function
@BlackBox.record(agent_name="customer-support-bot")
def run_support_agent(user_message: str):
    # your existing agent code, no changes needed
    return agent.respond(user_message)
```

```bash
# Launch dashboard
agentblackbox dashboard
# Open http://localhost:8765
# Click any session → ▶ replay the full timeline
```

**What gets recorded:**

- LLM calls: model, prompt, response, tokens, cost ($), latency (ms)
- Tool calls: name, args, return value, duration
- Errors: type, message, full stack trace
- Session metadata: agent name, start time, total cost, status

**PII masking is automatic** — card numbers, emails, API keys, phone numbers are redacted before hitting the DB.

**Who should try this:**

- Anyone building agents for regulated industries (finance, health, legal)
- Devs who want local observability before committing to a paid SaaS
- Teams with data-residency requirements

Would love feedback on: what events/metadata do you wish you had during agent debugging? What's missing?

GitHub (MIT): {GITHUB_URL}

---

### r/AI_Agents Pattern 2

**Title:** How do you observe/debug AI agents in production? Built a local-first alternative to LangSmith

**Body:**

Genuinely curious how people in this sub handle agent observability in prod.

I've been building agents with LangGraph and OpenAI Agents SDK, and my debugging workflow was embarrassingly manual: logs, print statements, occasionally a LangSmith trace when I remembered to set it up.

I finally sat down and built something I actually want to use every day: **AgentBlackBox**, an open-source local flight recorder for agents.

**OpenAI Agents SDK — zero-change setup:**

```python
from agentblackbox.integrations import patch_openai_agents

patch_openai_agents()  # all agents auto-recorded from this point on

# your existing agent code completely unchanged
agent = Agent(name="assistant", instructions="You are a helpful assistant.")
result = Runner.run_sync(agent, "Write a haiku about debugging.")
```

**Or use the decorator manually:**

```python
from agentblackbox import BlackBox

@BlackBox.record(agent_name="langgraph-agent")
def run_graph(state):
    return graph.invoke(state)
```

**Then replay in the dashboard:**

```bash
agentblackbox dashboard  # → http://localhost:8765
```

Every session has a step-by-step timeline. Click any LLM call to see the full prompt and response. See exactly where costs came from and where errors happened.

**Things I'd genuinely love feedback on:**

1. What's the most painful part of your current agent debugging workflow?
2. Is local-only storage a dealbreaker for you, or a feature?
3. Would you want alerting (e.g., "notify me if cost per session exceeds $X")?
4. Multi-agent tracing across tool calls — how important is this for you?

If you're running agents in production or staging, I'd love for you to try it and tell me what's missing.

```bash
pip install agentblackbox[dashboard]
```

GitHub (MIT): {GITHUB_URL}
