# Task 4: 「会社」向け外部資料ドラフト

---

## One-pager JP（Notion貼り付け用）

---

# AgentBlackBox — プロダクト概要

> AIエージェントのフライトレコーダー。デコレータ1行で全記録。

---

## Problem

- AIエージェントは本番で何をしているかわからない
- LLM呼び出し・ツール実行・コスト・エラーの追跡手段がない
- 既存ツール（LangSmith, Langfuse）はクラウド必須 → 機密データを外部送信できない案件に使えない
- 「エージェントを本番に出したいが、怖くて出せない」エンジニアが急増

---

## Solution

- **AgentBlackBox**：AIエージェントのすべての動作をローカルに記録するOSSツール
- デコレータ1行で既存コードを一切変えずに計装可能
- ローカルSQLiteに全データ保存 → 外部送信ゼロ、ベンダーロックインなし
- PIIは書き込み前に自動マスキング（カード番号・メール・APIキーなど）
- FastAPI + htmx のダッシュボードでセッションをタイムライン再生

---

## Why Now

- LLMエージェント市場が急拡大（2024〜2026年が本番化の波）
- 企業のAIガバナンス要件が厳格化：ログ保持・PII管理が義務化されつつある
- プライバシー規制（GDPR, 個人情報保護法）によるデータ外部送信への圧力が増大
- 「エージェントを作れる人」は増えたが「安全に本番運用できる人」はまだ少ない

---

## Product

```python
pip install agentblackbox[dashboard]

@BlackBox.record(agent_name="my-agent")
def run_agent(task): ...

agentblackbox dashboard  # → localhost:8765
```

- LLM呼び出し（モデル・トークン・コスト・レイテンシ）を全記録
- ツール呼び出し、エラー（スタックトレース付き）を記録
- 20+モデルのコスト自動計算（OpenAI, Anthropic）
- タイムライン再生ダッシュボード（▶）
- OpenAI Agents SDK / LangChain / LangGraph 対応

---

## Who It's For

- AIエージェントを本番運用しているSaaS企業のエンジニア
- FinTech・HealthTech・LegalTechなど規制業種
- データ外部送信に制約のある大企業・官公庁系
- LLMエージェントを受託開発しているチーム

---

## Next

- v1.0 OSS公開（GitHub）
- クラウド版（チーム共有ダッシュボード + 有料プラン）開発開始
- エンタープライズ向けセルフホスト版の検討

---
GitHub: {GITHUB_URL} | PyPI: `pip install agentblackbox`

---

## One-pager EN（Notion貼り付け用）

---

# AgentBlackBox — Product Overview

> The flight recorder for AI agents. One decorator. Full local observability.

---

## Problem

- AI agents are black boxes in production — nobody knows what they're actually doing
- No standard way to track LLM calls, tool usage, costs, and errors across agent runs
- Existing tools (LangSmith, Langfuse) require sending data to external servers — not viable for regulated industries, sensitive data, or strict data-residency requirements
- "I'd ship this agent but I'm scared of what happens when it's live" is now a universal engineer complaint

---

## Solution

- **AgentBlackBox**: an open-source flight recorder that logs all agent activity to local SQLite
- One-line decorator — existing code unchanged
- Zero external data transmission; no account, no vendor lock-in
- PII auto-masked before storage (card numbers, emails, API keys, phone numbers)
- Web dashboard (FastAPI + htmx) for step-by-step session replay

---

## Why Now

- The LLM agent market is in its "first production wave" (2024–2026)
- Enterprise AI governance requirements are tightening — audit trails and PII controls are becoming mandatory
- Privacy regulations (GDPR, state laws) are increasing pressure against exporting prompt data
- The gap between "can build an agent" and "can safely run an agent in prod" is a real and growing problem

---

## Product

```python
pip install agentblackbox[dashboard]

@BlackBox.record(agent_name="my-agent")
def run_agent(task): ...

agentblackbox dashboard  # → http://localhost:8765
```

- Records every LLM call: model, tokens, cost, latency, full prompt/response
- Records every tool invocation and error (with stack trace)
- Auto-cost calculation for 20+ models (OpenAI, Anthropic)
- Timeline replay dashboard with analytics
- Works with OpenAI Agents SDK, LangChain, LangGraph, or any Python agent

---

## Who It's For

- Engineers running AI agents in production at SaaS companies
- Regulated industries: FinTech, HealthTech, LegalTech, GovTech
- Enterprises with data-residency or egress restrictions
- Consulting teams building LLM agents for clients

---

## Next

- v1.0 OSS release (GitHub + PyPI)
- Cloud version: shared team dashboards, multi-project management, paid plans
- Enterprise: self-hosted option with SSO and compliance exports

---
GitHub: {GITHUB_URL} | `pip install agentblackbox`

---

## 投資家向けピッチ（英語）

---

# AgentBlackBox — Investor Brief

> The flight recorder for AI agents.
> OSS core. Cloud business. Local-first.

---

## Problem

Every airplane has a black box. AI agents don't.

In 2026, thousands of engineering teams are deploying AI agents to production. These agents make decisions, call APIs, spend money, and touch sensitive user data — but when something goes wrong, there is no record of what happened.

The problem is not capability. The problem is **observability**:

- No standard way to replay what an agent did, step by step
- No automatic cost tracking across LLM providers
- No PII protection before logs are written
- Existing tools (LangSmith, Langfuse, Arize) require sending all prompt/response data to external servers — a non-starter for regulated industries, data-residency requirements, and sensitive-data use cases

**The result:** AI agents are being held back from production by a trust gap, not a capability gap.

---

## Solution

**AgentBlackBox** is an open-source Python library that acts as a flight recorder for AI agents.

One decorator. Every LLM call, tool use, cost, and error recorded. Local SQLite. No cloud required.

```python
@BlackBox.record(agent_name="customer-support")
def run_agent(task: str): ...
```

That's the entire integration. Nothing else changes. A web dashboard replays any session timeline with full context.

**Key differentiator:** local-first by design. Data never leaves the user's machine unless they explicitly opt in to cloud mode. PII is masked before any write.

---

## Why Now

- **Agent adoption is accelerating.** OpenAI, Anthropic, Google, and LangChain all launched agent frameworks in 2024–2025. The first production wave is happening now.
- **Governance is becoming a requirement.** Enterprise AI policies, EU AI Act, and sector-specific regulations are making audit trails and data controls mandatory.
- **The "trust gap" is a real blocker.** Developer surveys consistently show that observability and reliability — not model capability — are the top blockers to production deployment.
- **No one owns local-first observability.** LangSmith and Langfuse are cloud-SaaS. There is no credible local-first alternative.

---

## Market

- **Primary:** Engineering teams building LLM agents — estimated {TAM_ESTIMATE} globally in 2026 (growing)
- **Immediate beachhead:** Python developers using OpenAI, Anthropic, LangChain, LangGraph — ~{LANGCHAIN_DOWNLOADS} monthly downloads on LangChain alone
- **Monetization path:** OSS funnel → cloud product for teams → enterprise self-hosted with compliance features
- **Comparable SaaS:** Datadog ($30B+ market cap) for traditional infra observability; we are building the equivalent for AI agents

---

## Product

**OSS (free, MIT licensed)**
- `@BlackBox.record` decorator
- Local SQLite storage
- PII auto-masking
- Web dashboard (FastAPI + htmx)
- CLI tools (`sessions`, `replay`, `export`)
- 20+ model cost tracking

**Cloud (planned, paid)**
- Shared team dashboards
- Multi-project management
- API key management
- Usage analytics and alerts
- Billing: Stripe subscription

**Enterprise (planned)**
- Self-hosted deployment
- SSO / SAML
- Compliance exports (SOC 2, GDPR audit logs)
- SLA and support

---

## Traction

- GitHub Stars: **{GITHUB_STARS}**
- PyPI Downloads (total): **{PYPI_DOWNLOADS}**
- PyPI Downloads (last 30 days): **{PYPI_MONTHLY_DOWNLOADS}**
- Discord / community members: **{COMMUNITY_SIZE}**
- Early beta users: **{BETA_USERS}**
- Revenue: $0 (pre-monetization OSS phase)

---

## Team

**{FOUNDER_NAME}** — Founder & Solo Engineer

- {FOUNDER_BACKGROUND}
- Built AgentBlackBox from scratch; {GITHUB_STARS} stars in {TIME_SINCE_LAUNCH}
- Previously: {PREVIOUS_EXPERIENCE}

*Looking to hire first engineer and first GTM hire at seed stage.*

---

## Vision

In five years, every production AI agent runs with a black box.

AgentBlackBox is the Datadog for AI agents — but starting local, earning trust, and growing into the infrastructure layer that enterprises require before they can safely deploy agents at scale.

The OSS library is the distribution wedge. The cloud product is the business. The enterprise tier is the moat.

---

## The Ask

Raising **{RAISE_AMOUNT}** seed round.

Use of funds:
- Hire first engineer (backend / infra)
- Build and ship cloud v1 (6 months)
- Go-to-market: developer relations, content, OSS community growth

We believe the window to own "local-first AI agent observability" is open now.

---
GitHub: {GITHUB_URL}
Contact: {FOUNDER_EMAIL}
