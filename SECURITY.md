# Security Policy

## Reporting a vulnerability

If you find a security issue in AgentBlackBox, **do not open a public
issue**. Email me directly at:

  **karisutefin73@gmail.com**

Use the subject line: `[security] AgentBlackBox: <short summary>`

Include:
- The version affected (`pip show agentblackbox`)
- A reproduction case (smallest possible)
- The impact you see (data exposure, crash, etc.)
- Whether you've shared this with anyone else

I'll acknowledge receipt within 72 hours and aim to ship a fix or a
mitigation within 14 days for non-trivial issues.

## Scope

In scope:
- The `agentblackbox` library itself (recorder, storage, integrations,
  dashboard, remote/cloud ingest API)
- Default file permissions on `~/.agentblackbox/recordings.db`
- API key handling in `RemoteStorage`

Out of scope:
- Third-party SDKs (Anthropic, OpenAI, LangChain) — report those upstream
- Self-hosted dashboard deployments where the operator has misconfigured
  network access (we don't run your infrastructure)
- Vulnerabilities in dependencies of optional extras — please file
  with the upstream project first

## Data handling note

AgentBlackBox records LLM prompts, tool arguments, and tool return
values verbatim into a local SQLite file. **Treat this file as
sensitive** — it may contain API keys, customer data, PII, or other
secrets that flowed through your prompts.

The default location (`~/.agentblackbox/recordings.db`) is created with
user-only permissions on Unix. On Windows, ensure the file is in a
non-shared user directory.

A redaction hook for sensitive substrings is on the v0.3 roadmap. If
you need it sooner, open an issue describing your case.
