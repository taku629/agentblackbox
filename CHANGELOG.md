# Changelog

## [0.2.0] - 2026-04-27

### Added
- PII auto-masking (9 built-in patterns, custom patterns support)
- `BlackBox.configure()` class method for global configuration
- Visual Replay dashboard with animated timeline
- Replay button (▶) for step-by-step session replay
- Cost breakdown bar chart per LLM call
- `delete_session()` storage method
- `export_json(path)` writes to file (breaking change from 0.1.0)

### Changed
- Model field names updated to spec: `call_id`, `latency_ms`, `args`, `error_id`
- `Session` now embeds `llm_calls`, `tool_calls`, `errors` lists
- Storage uses `save_session()` / `get_session()` / `list_sessions()` public API
- Timestamps changed from nanoseconds (int) to seconds (float)
- `gpt-4o` price corrected to $5.00/$15.00 per 1M tokens

### Added models
- claude-opus-4-5, gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash
- llama-3.1-70b, mistral-large, deepseek-chat, grok-2

## [0.1.0] - 2026-04-26

### Added
- BlackBox decorator and context manager
- LLM call, tool call, error recording
- Cost calculation for 20+ models
- SQLite local storage
- Web dashboard (FastAPI + htmx + Chart.js)
- CLI: sessions, replay, export, dashboard
