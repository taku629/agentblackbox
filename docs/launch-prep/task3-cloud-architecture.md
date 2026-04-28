# AgentBlackBox Cloud — Architecture Design Draft

---

# Architecture

## Overview

AgentBlackBox Cloud is the hosted, multi-tenant version of AgentBlackBox OSS.
The OSS client gains an opt-in cloud mode: instead of (or in addition to) writing to local SQLite, events are streamed to the cloud API over HTTPS.

## Components

| Component | Technology | Role |
|---|---|---|
| **OSS Client** | Python library | Records events; routes to SQLite, cloud, or both |
| **Cloud API** | FastAPI (Python) | Receives events, manages auth, serves dashboard data |
| **Database** | PostgreSQL (Supabase) | Primary data store for all sessions and events |
| **Auth** | Supabase Auth | Magic link + GitHub OAuth; issues JWTs |
| **Billing** | Stripe Billing | Subscription management, usage metering |
| **Frontend** | FastAPI + htmx | Web dashboard (same stack as OSS version) |
| **Background Workers** | FastAPI BackgroundTasks / ARQ | Async event ingestion, Stripe webhook handling |

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  User's Application                                          │
│                                                             │
│  @BlackBox.record(agent_name="my-agent")                    │
│  def run_agent(task): ...                                   │
└────────────────┬────────────────────────────────────────────┘
                 │ event emitted (in-process)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  AgentBlackBox OSS Client (Python)                          │
│                                                             │
│  EventRouter                                                │
│   ├─► LocalWriter  → ~/.agentblackbox/recordings.db (SQLite)│
│   └─► CloudWriter  → HTTPS POST /api/v1/events/batch        │
│         ├─ PII masking (before any write)                   │
│         ├─ In-memory queue (ring buffer, max 1000 events)   │
│         └─ Retry with exponential backoff                   │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTPS (Bearer: abb_...)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Cloud API  (FastAPI, hosted on Fly.io / Railway / AWS)     │
│                                                             │
│  POST /api/v1/events/batch                                  │
│   ├─ Validate API key → resolve org_id + project_id        │
│   ├─ Check billing quota (events/month)                     │
│   ├─ PII second-pass masking (server-side safety net)       │
│   └─ Bulk INSERT into PostgreSQL (events table)             │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  PostgreSQL (Supabase)                                       │
│   users / organizations / projects / api_keys /             │
│   sessions / events / billing_subscriptions                 │
└─────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Web Dashboard  (FastAPI + htmx, same codebase as OSS)      │
│                                                             │
│  Auth: Supabase JWT → user → org → projects                 │
│  Routes:                                                    │
│   /dashboard         → session list                         │
│   /sessions/{id}     → timeline replay                      │
│   /analytics         → cost trends, model distribution      │
│   /settings/billing  → Stripe Customer Portal               │
└─────────────────────────────────────────────────────────────┘

Stripe Webhooks:
  stripe → POST /webhooks/stripe
              └─ update billing_subscriptions.status
```

---

# Database

## Design Principles

- **Multi-tenant by `org_id`**: every table that holds user data includes `org_id` as a non-nullable foreign key.
- **Row-Level Security**: Supabase RLS policies enforce `org_id` isolation at the DB layer.
- **Soft deletes**: `deleted_at TIMESTAMPTZ` on user-facing entities instead of hard DELETE.
- **UUIDs**: all primary keys use `gen_random_uuid()` (PostgreSQL 13+).

---

## Schema

### `users`
```sql
CREATE TABLE users (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email        TEXT NOT NULL UNIQUE,
    display_name TEXT,
    avatar_url   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at   TIMESTAMPTZ
);

CREATE INDEX idx_users_email ON users(email);
```

---

### `organizations`
```sql
CREATE TABLE organizations (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL,
    slug       TEXT NOT NULL UNIQUE,     -- used in URLs: app.agentblackbox.io/acme
    plan       TEXT NOT NULL DEFAULT 'free',   -- free | pro | enterprise
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);

CREATE INDEX idx_organizations_slug ON organizations(slug);
```

---

### `organization_members`
```sql
CREATE TABLE organization_members (
    org_id     UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role       TEXT NOT NULL DEFAULT 'member',  -- owner | admin | member | viewer
    joined_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (org_id, user_id)
);

CREATE INDEX idx_org_members_user_id ON organization_members(user_id);
```

---

### `projects`
```sql
CREATE TABLE projects (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id      UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    slug        TEXT NOT NULL,
    description TEXT,
    created_by  UUID REFERENCES users(id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at  TIMESTAMPTZ,
    UNIQUE (org_id, slug)
);

CREATE INDEX idx_projects_org_id ON projects(org_id);
```

---

### `api_keys`
```sql
CREATE TABLE api_keys (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id   UUID REFERENCES projects(id) ON DELETE CASCADE,  -- NULL = org-level key
    name         TEXT NOT NULL,
    key_hash     TEXT NOT NULL UNIQUE,   -- SHA-256 of the actual key; never store plaintext
    key_prefix   TEXT NOT NULL,          -- first 8 chars, shown in UI: abb_1a2b3c4d...
    last_used_at TIMESTAMPTZ,
    expires_at   TIMESTAMPTZ,
    created_by   UUID REFERENCES users(id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at   TIMESTAMPTZ
);

CREATE INDEX idx_api_keys_key_hash  ON api_keys(key_hash);
CREATE INDEX idx_api_keys_org_id    ON api_keys(org_id);
CREATE INDEX idx_api_keys_project_id ON api_keys(project_id);
```

---

### `sessions`
```sql
CREATE TABLE sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id   UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    agent_name   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running',  -- running | completed | error
    started_at   TIMESTAMPTZ NOT NULL,
    ended_at     TIMESTAMPTZ,
    duration_ms  INTEGER,
    total_cost   NUMERIC(12, 6),         -- USD
    event_count  INTEGER NOT NULL DEFAULT 0,
    error_count  INTEGER NOT NULL DEFAULT 0,
    metadata     JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_org_id     ON sessions(org_id);
CREATE INDEX idx_sessions_project_id ON sessions(project_id);
CREATE INDEX idx_sessions_started_at ON sessions(started_at DESC);
CREATE INDEX idx_sessions_agent_name ON sessions(agent_name);
-- Partial index for active sessions
CREATE INDEX idx_sessions_running ON sessions(org_id, started_at)
    WHERE status = 'running';
```

---

### `events`
```sql
CREATE TABLE events (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    session_id   UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    event_type   TEXT NOT NULL,   -- llm_call | tool_call | error | custom
    sequence     INTEGER NOT NULL,
    timestamp    TIMESTAMPTZ NOT NULL,
    duration_ms  INTEGER,

    -- LLM call fields
    model        TEXT,
    input_text   TEXT,           -- PII-masked before storage
    output_text  TEXT,           -- PII-masked before storage
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd     NUMERIC(12, 6),

    -- Tool call fields
    tool_name    TEXT,
    tool_args    JSONB,          -- PII-masked
    tool_result  JSONB,          -- PII-masked

    -- Error fields
    error_type   TEXT,
    error_message TEXT,
    stack_trace  TEXT,

    -- Shared
    metadata     JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_events_session_id ON events(session_id);
CREATE INDEX idx_events_org_id     ON events(org_id);
CREATE INDEX idx_events_timestamp  ON events(timestamp DESC);
CREATE INDEX idx_events_type       ON events(event_type);
-- Composite for timeline queries
CREATE INDEX idx_events_session_seq ON events(session_id, sequence ASC);
```

---

### `billing_subscriptions`
```sql
CREATE TABLE billing_subscriptions (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                UUID NOT NULL UNIQUE REFERENCES organizations(id) ON DELETE CASCADE,
    stripe_customer_id    TEXT NOT NULL UNIQUE,
    stripe_subscription_id TEXT,
    plan                  TEXT NOT NULL DEFAULT 'free',   -- free | pro | enterprise
    status                TEXT NOT NULL DEFAULT 'active', -- active | past_due | canceled | trialing
    current_period_start  TIMESTAMPTZ,
    current_period_end    TIMESTAMPTZ,
    trial_end             TIMESTAMPTZ,
    events_used_this_period BIGINT NOT NULL DEFAULT 0,
    events_limit          BIGINT NOT NULL DEFAULT 50000,  -- per billing period
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_billing_org_id           ON billing_subscriptions(org_id);
CREATE INDEX idx_billing_stripe_customer  ON billing_subscriptions(stripe_customer_id);
```

---

### `usage_daily`  *(materialized summary for analytics)*
```sql
CREATE TABLE usage_daily (
    org_id        UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    project_id    UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    date          DATE NOT NULL,
    session_count INTEGER NOT NULL DEFAULT 0,
    event_count   INTEGER NOT NULL DEFAULT 0,
    total_cost    NUMERIC(12, 6) NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, project_id, date)
);
```

---

# API

## Design Principles

- Base path: `/api/v1`
- Two auth methods: **API Key** (for OSS client ingestion) and **JWT** (for dashboard/frontend)
- API keys are passed as `Authorization: Bearer abb_<token>`
- JWTs are issued by Supabase Auth and validated via `python-jose`
- All endpoints return `application/json`
- Errors follow RFC 7807 (`{"type": ..., "title": ..., "status": ..., "detail": ...}`)

---

## Endpoints

### Auth & Keys

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/auth/magic-link` | None | Request magic-link email |
| POST | `/auth/callback` | None | Supabase OAuth callback |
| GET  | `/api/v1/me` | JWT | Current user + org memberships |
| POST | `/api/v1/api-keys` | JWT | Create API key |
| GET  | `/api/v1/api-keys` | JWT | List org's API keys |
| DELETE | `/api/v1/api-keys/{id}` | JWT | Revoke key |

---

### Event Ingestion (OSS client uses these)

#### `POST /api/v1/events/batch`
**Auth:** API Key

**Request:**
```json
{
  "project_id": "proj_abc123",
  "session": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "agent_name": "research-agent",
    "started_at": "2026-04-28T10:00:00Z"
  },
  "events": [
    {
      "event_type": "llm_call",
      "sequence": 1,
      "timestamp": "2026-04-28T10:00:01Z",
      "duration_ms": 823,
      "model": "gpt-4o",
      "input_text": "Summarize the following document...",
      "output_text": "Here is a summary...",
      "input_tokens": 512,
      "output_tokens": 128,
      "cost_usd": 0.002048
    },
    {
      "event_type": "tool_call",
      "sequence": 2,
      "timestamp": "2026-04-28T10:00:02Z",
      "duration_ms": 210,
      "tool_name": "web_search",
      "tool_args": {"query": "latest AI news"},
      "tool_result": {"results": ["..."]}
    }
  ]
}
```

**Response `202 Accepted`:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "accepted": 2,
  "rejected": 0
}
```

---

#### `POST /api/v1/sessions/{session_id}/close`
**Auth:** API Key

**Request:**
```json
{
  "status": "completed",
  "ended_at": "2026-04-28T10:00:05Z",
  "duration_ms": 5000,
  "total_cost": 0.002048
}
```

**Response `200 OK`:**
```json
{ "session_id": "550e8400-...", "status": "completed" }
```

---

### Sessions & Events (Dashboard uses these)

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/projects/{project_id}/sessions` | JWT | List sessions (paginated) |
| GET | `/api/v1/sessions/{session_id}` | JWT | Session detail + summary |
| GET | `/api/v1/sessions/{session_id}/events` | JWT | All events for a session (timeline) |
| DELETE | `/api/v1/sessions/{session_id}` | JWT | Soft-delete session |

**`GET /api/v1/projects/{project_id}/sessions` Response:**
```json
{
  "items": [
    {
      "id": "550e8400-...",
      "agent_name": "research-agent",
      "status": "completed",
      "started_at": "2026-04-28T10:00:00Z",
      "duration_ms": 5000,
      "total_cost": 0.002048,
      "event_count": 12,
      "error_count": 0
    }
  ],
  "total": 42,
  "page": 1,
  "per_page": 20
}
```

---

### Analytics

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/projects/{project_id}/analytics/daily` | JWT | Daily cost + session counts |
| GET | `/api/v1/projects/{project_id}/analytics/models` | JWT | Cost breakdown by model |
| GET | `/api/v1/projects/{project_id}/analytics/agents` | JWT | Stats per agent name |

---

### Billing

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/api/v1/billing/subscription` | JWT | Current plan + usage |
| POST | `/api/v1/billing/checkout` | JWT | Create Stripe Checkout session |
| POST | `/api/v1/billing/portal` | JWT | Stripe Customer Portal link |
| POST | `/webhooks/stripe` | Stripe-Signature header | Stripe event webhook |

---

## Rate Limits

| Endpoint group | Limit | Window |
|---|---|---|
| `POST /api/v1/events/batch` | 300 req | 1 minute per API key |
| `POST /api/v1/events/batch` (payload) | max 500 events per batch | per request |
| Dashboard read endpoints | 120 req | 1 minute per JWT |
| Auth endpoints | 10 req | 1 minute per IP |

Rate limit headers returned on all responses:
```
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 297
X-RateLimit-Reset: 1714298460
```

---

## Security Notes

- **API keys**: stored as `SHA-256(key)` only; plaintext shown once at creation time
- **PII masking**: applied client-side before transmission AND server-side as a safety net
- **TLS**: all endpoints HTTPS-only; HSTS enforced
- **CORS**: dashboard origin allowlisted; API ingestion endpoints allow any origin (API key is the auth)
- **SQL injection**: all queries via SQLAlchemy ORM with parameterized queries
- **RLS**: Supabase Row-Level Security policies enforce org isolation at DB layer
- **Stripe webhooks**: signature validated via `stripe.Webhook.construct_event()`
- **Secrets rotation**: API key revocation is immediate (DB lookup on each request)

---

# Client Changes

## New Configuration Interface

```python
from agentblackbox import BlackBox

# Local-only mode (current default — unchanged)
BlackBox.configure(
    db_path="~/.agentblackbox/recordings.db"
)

# Cloud-only mode
BlackBox.configure(
    cloud_endpoint="https://api.agentblackbox.io",
    api_key="abb_1a2b3c4d5e6f7g8h",
    project_id="proj_abc123"
)

# Hybrid mode: write to local SQLite AND cloud simultaneously
BlackBox.configure(
    db_path="~/.agentblackbox/recordings.db",
    cloud_endpoint="https://api.agentblackbox.io",
    api_key="abb_1a2b3c4d5e6f7g8h",
    project_id="proj_abc123",
    mode="hybrid"   # "local" | "cloud" | "hybrid"
)
```

Environment variable alternative (for CI/CD, Docker):
```bash
export ABB_CLOUD_ENDPOINT="https://api.agentblackbox.io"
export ABB_API_KEY="abb_1a2b3c4d5e6f7g8h"
export ABB_PROJECT_ID="proj_abc123"
export ABB_MODE="hybrid"
```

`BlackBox.configure()` reads env vars as fallback if kwargs are not provided.

---

## CloudWriter Implementation Design

```python
class CloudWriter:
    """
    Buffers events in memory and flushes in batches to the cloud API.
    Runs a background thread; never blocks the agent's hot path.
    """

    MAX_QUEUE_SIZE = 1000        # ring buffer; oldest dropped if full
    BATCH_SIZE = 100             # events per HTTP POST
    FLUSH_INTERVAL_SEC = 5.0    # flush at least every N seconds
    
    # Retry / backoff
    MAX_RETRIES = 5
    BASE_BACKOFF_SEC = 1.0      # 1s, 2s, 4s, 8s, 16s
    MAX_BACKOFF_SEC = 30.0

    def enqueue(self, event: Event) -> None:
        """Non-blocking. Drops oldest event if queue is full (circuit breaker)."""
        ...

    def _flush_worker(self) -> None:
        """Background thread: collect batch, POST, retry on failure."""
        ...

    def _post_batch(self, events: list[Event]) -> bool:
        """
        POST /api/v1/events/batch
        Returns True on 2xx, raises on 4xx (no retry), retries on 5xx/network error.
        """
        ...

    def close(self) -> None:
        """Flush remaining events synchronously before process exit."""
        ...
```

### Retry & Backoff Policy

| Condition | Behavior |
|---|---|
| `2xx` | Success, clear retry counter |
| `400 Bad Request` | Log error, discard batch (no retry — malformed data) |
| `401 Unauthorized` | Log error, disable cloud writer, warn user |
| `402 Payment Required` | Log warning ("quota exceeded"), disable until next period |
| `429 Too Many Requests` | Respect `Retry-After` header, then exponential backoff |
| `5xx` / network error | Exponential backoff: 1s → 2s → 4s → 8s → 16s (max 5 retries) |
| All retries exhausted | Log warning, discard batch, continue (agent not blocked) |

### Hybrid Mode Behavior

- Local write happens synchronously (same as current OSS behavior) — always succeeds first.
- Cloud write happens asynchronously in background thread.
- A cloud failure never causes a local write failure.
- Sessions visible in local dashboard immediately, cloud dashboard after flush.

### Graceful Shutdown

```python
import atexit
atexit.register(BlackBox._shutdown)   # flush remaining events on process exit
```

`_shutdown()` gives the background thread up to 10 seconds to drain the queue before force-closing.
