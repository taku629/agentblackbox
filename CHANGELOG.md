# Changelog

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
