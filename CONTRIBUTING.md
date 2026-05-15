# Contributing to AgentBlackBox

Thanks for considering a contribution. AgentBlackBox is a small library
with a clear scope, so PRs that add focused, tested improvements get
merged quickly.

## Quick start (5 minutes)

```bash
git clone https://github.com/taku629/agentblackbox
cd agentblackbox
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest                 # 95 tests should pass in ~6 seconds
```

## What's in scope

- New SDK integrations (LlamaIndex, CrewAI, AutoGen, Pydantic AI, etc.)
- More accurate cost data for new model releases
- Performance improvements for the SQLite storage layer
- Dashboard UX improvements
- Replay-as-test (`MockBlackBox`) extensions
- Better error messages, type hints, docstrings

## What's out of scope

- Cloud-hosted billing, signup, or auth flows (that lives in the closed-source Cloud product)
- Anything that adds a runtime dependency to the *core* library
  (the dashboard, integrations, etc. can have their own optional deps)
- Rewrites in another language

If you're unsure whether your idea fits, open an issue first.

## Pull request guidelines

1. **Tests first.** Every behavior change needs a test. We aim for >90% coverage.
2. **No new core dependencies.** Optional deps are fine inside `pyproject.toml`'s
   `[project.optional-dependencies]`.
3. **One concern per PR.** Refactors and feature changes in the same PR
   make review hard.
4. **Update the changelog.** Add a line to `CHANGELOG.md` under "Unreleased".

## Reporting bugs

Use the bug report issue template. The most helpful bug reports include:

- Python version (`python --version`)
- AgentBlackBox version (`pip show agentblackbox`)
- The smallest snippet that reproduces the issue
- The full traceback

## Development tips

- The recorder is in `agentblackbox/recorder.py`. The storage layer is in
  `agentblackbox/storage.py`. Most changes touch one of these.
- The dashboard (`agentblackbox/dashboard/app.py`) is FastAPI + htmx; you
  can hot-reload with `agentblackbox dashboard --port 8765` and refresh.
- Cost data lives in `agentblackbox/cost.py` and uses a prefix-fallback
  table — adding a new model usually means adding two lines.

## Code of conduct

Be kind, be specific, and assume good faith. Disagreements about code
are fine; disagreements about people are not.

## License

By contributing, you agree your contributions are licensed under MIT,
the same license as the project.
