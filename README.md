# 🔲 AgentBlackBox

**A reproducible flight recorder and evaluation toolkit for AI agents.**

![PyPI](https://img.shields.io/pypi/v/agentblackbox)
![Python](https://img.shields.io/pypi/pyversions/agentblackbox)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 1. Project title and tagline
**AgentBlackBox**  
A reproducible flight recorder and evaluation toolkit for AI agents.

## 2. What problem it solves
AI agents execute complex reasoning loops, make tool calls, and alter state. When they fail or hallucinate, diagnosing *why* is often impossible if the execution environment wasn't fully instrumented. 
AgentBlackBox provides reproducible tracking and analysis for AI agent executions. It helps you answer:
- Which tasks were executed and what were the outcomes?
- What tools were called and what did they return?
- How much time and tokens were spent?
- How do different models and prompts compare?

## 3. Key features
- **Local-first Evaluation**: Store run traces locally in JSON/JSONL format.
- **Trace Context Manager**: Easy-to-use `with trace(...)` API for any Python codebase.
- **CLI Analytics**: List, show, compare, and export runs directly from your terminal.
- **Secret Redaction**: Automatically detect and mask API keys and passwords recursively.
- **SQLite Support**: Retains the legacy `BlackBox` SQLite recorder and web dashboard.

## 4. Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run a minimal example
python examples/basic_trace.py

# View the run
agentblackbox runs list
agentblackbox runs show <run_id> --events

# Run the benchmark demo
python examples/benchmark_demo.py
```

## 5. Minimal Python example

```python
from agentblackbox import trace

with trace(
    task="Fix failing unit tests",
    model="example-model",
    tags=["coding", "benchmark"],
    metadata={"repository": "night-refactor-bot"},
) as run:
    run.log_event(
        event_type="tool_call",
        name="pytest",
        input={"command": "pytest -q"},
    )

    run.log_event(
        event_type="tool_result",
        name="pytest",
        output={"exit_code": 1, "stdout": "..."},
        status="error",
    )

    run.set_outcome(
        status="failure",
        summary="Two tests failed because of an invalid import.",
    )
```

## 6. CLI examples

```bash
# List recent runs
agentblackbox runs list --limit 10

# Show details and timeline of a specific run
agentblackbox runs show <run_id> --events

# Compare two runs side-by-side
agentblackbox runs compare <run_id_a> <run_id_b>

# Export a run as a single JSON payload
agentblackbox runs export <run_id> --format json --output my_run.json
```

## 7. Storage layout

AgentBlackBox stores evaluation runs in your current working directory:

```text
.agentblackbox/
└── runs/
    └── <run_id>/
        ├── run.json       # Run metadata and outcome
        ├── events.jsonl   # Timeline of events
        └── error.json     # Exception details (if any)
```

## 8. Security and redaction policy
AgentBlackBox includes built-in recursive redaction for your agent logs.
It automatically replaces values matching common patterns (OpenAI, Anthropic, GitHub tokens) and sensitive keys (`api_key`, `token`, `password`, `secret`, `authorization`) with `[REDACTED]`.
This ensures you can safely commit or share your evaluation traces.

## 9. Running the reproducible demo
We provide a reproducible, synthetic benchmark demo that does not require any external APIs.
```bash
python examples/benchmark_demo.py
```
This script generates runs for multiple conditions (`baseline` vs `self_review`), compares them, and outputs a summary to `experiments/results/report.md`. Note: The results are a synthetic demonstration.

## 10. Development setup
1. Clone the repository
2. `pip install -e ".[dev]"`
3. Run tests with `pytest tests/`

## 11. Roadmap
- CI/CD integration for automated benchmark evaluations.
- Support for distributed tracing backends.
- Visual timeline comparisons in the web dashboard.

## 12. License / contribution guidance
MIT License. Contributions are welcome! Please open an issue or submit a pull request.
