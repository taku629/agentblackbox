# Changelog

## [0.2.2] - 2026-05-16

### Added
- **Share links without a server.** `agentblackbox share <session_id>`
  gzips the session and base64url-encodes it into a URL fragment, so a
  recording can be opened in the browser viewer with no upload step
  and no account. Override the host with `--base-url` or
  `AGENTBLACKBOX_SHARE_BASE_URL`.
- New static viewer at `site/view.html` — single file, renders the
  session timeline (LLM calls, tool calls, errors, costs) directly from
  the URL hash.
- Vercel deployment config (`vercel.json`) so the LP and viewer ship
  together as one static site.

### Notes
- Sessions over ~24KB encoded are auto-truncated and flagged in the
  viewer banner; the full-fidelity story belongs in the Cloud product.

## [0.2.1] - 2026-05-10

### Changed
- **README rewrite**: removed unsourced "72%" claim; added a comparison table
  against LangSmith / Langfuse / Phoenix; added "Why local-first" rationale.
- **Cloud positioning is now honest**: marked as Coming Soon (Beta is free)
  rather than implying a live $19/mo product. Self-hosted remote mode
  remains documented separately.
- Updated Anthropic model coverage in README to current 4.x family
  (Opus 4.7, Sonnet 4.6, Haiku 4.5).
- Test/coverage badges now reflect actual numbers (95 tests, 96% coverage).

### Added
- Real CI workflow (`.github/workflows/ci.yml`) running pytest on Python
  3.10/3.11/3.12 plus a clean-install smoke test. README badge now points
  to the live workflow instead of a static badge.
- `CONTRIBUTING.md` and issue templates (bug report, feature request).

### API Cleanup
- `BlackBox.replay(session_id)` and `BlackBox.export_json(session_id)`
  are now classmethods. The README quickstart `BlackBox.replay(sid)`
  now actually works without conjuring a dummy instance.
- Removed the `BlackBox.__new__(BlackBox); set fields manually` hack
  from CLI and `examples/basic_usage.py`.
- Existing `bb.replay(sid)` and `bb.export_json(sid)` calls still work
  (Python lets you call classmethods through instances). Test suite
  remains green at 95 / 95.

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
