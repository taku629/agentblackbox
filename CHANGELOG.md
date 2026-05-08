# Changelog

## [0.2.0] - 2026-05-08

### Added
- `patch_anthropic()` — auto-instrumentation for Anthropic Python SDK (sync + async)
- `BlackBoxCallbackHandler` — LangChain callback handler for LLM + tool recording
- `RemoteStorage` — push recordings to a remote AgentBlackBox cloud server
- Cloud ingest API (`POST /api/v1/ingest/*`) in the dashboard with API key auth
- `agentblackbox dashboard --cloud` flag: generates API key and starts cloud-mode server
- Landing page (`site/index.html`) deployable to GitHub Pages
- GitHub Actions workflow for automatic GitHub Pages deployment

## [0.1.0] - 2026-04-26

### Added
- `BlackBox` class with decorator and context manager support
- `record_llm_call()`, `record_tool_call()`, `record_error()` methods
- Cost auto-calculation for 20+ models (OpenAI, Anthropic)
- SQLite storage with WAL mode and full-text indexing
- `replay()` console timeline viewer
- `export_json()` session exporter
- `list_sessions()` with agent/status/limit filters
- Web dashboard (FastAPI + htmx + Chart.js)
  - Sessions list with real-time filter
  - Timeline detail view with accordion
  - Cost analytics with 3 Chart.js graphs
- CLI: `agentblackbox sessions|replay|export|dashboard`
- OpenAI Agents SDK integration stub (`patch_openai_agents()`)
