# Changelog

## [0.2.0] - 2026-05-08

### Added
- **Anthropic SDK integration** — `patch_anthropic()` auto-records all `messages.create()` calls (sync + async)
- **LangChain integration** — `BlackBoxCallbackHandler` drop-in callback for any LangChain LLM/tool chain
- **`MockBlackBox`** — load a recorded session and replay its responses deterministically for testing
  - `BlackBox.load_recording(session_id)` factory method
  - `pop_llm_response()` / `pop_tool_result()` to consume recorded data in order
  - `remaining_llm_responses()` / `remaining_tool_results()` counters
- **`iter_events()`** — generator that yields all session events in timestamp order
- **`RemoteStorage`** — push recordings to a remote AgentBlackBox cloud server
- Cloud ingest API (`POST /api/v1/ingest/*`) in the dashboard with API key auth
- `agentblackbox dashboard --cloud` flag: generates API key and starts cloud-mode server
- Landing page (`site/index.html`) deployable to GitHub Pages
- GitHub Actions workflow for automatic GitHub Pages deployment
- New optional install extras: `anthropic`, `langchain`, `all`

### Improved
- Test suite: 92+ tests, **96% coverage** (was 39 tests, 73%)
- Coverage threshold raised to 90%

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
