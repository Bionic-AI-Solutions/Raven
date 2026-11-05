# letta-memory

A Raven memory backend plugin: Raven's session learnings live in a **Letta
server's archival memory** — one REST-queryable archive per deployment,
beside (not inside) the agent process.

## What it does

| Contract call | Letta wire |
|---|---|
| `recall(query, user_id \| agent_id)` | `GET /v1/agents/<letta_agent_id>/archival-memory/search?query=…&tags=…&tag_match_mode=all&top_k=…` — user track only; the agent track returns `[]` (flat-backend convention; transcripts are not skills) |
| `store(session_id, messages)` | `POST /v1/agents/<letta_agent_id>/archival-memory` — one passage per turn |
| `recall_session(session_id)` | same search, filtered to `raven:session:<id>` |
| `delete(memory_id)` | `DELETE /v1/agents/<letta_agent_id>/archival-memory/<id>` |
| `health()` | `GET /v1/agents/<letta_agent_id>` (bounded reachability + auth + existence) |
| `feedback(signal)` | no-op (logged once per kind at debug) |

Verified against Letta **0.16.7** (live OpenAPI schema, 2026-11-05).

Every passage carries tags so one Letta agent's archive stays separable:
`raven`, `raven:user:<id>`, `raven:agent:<id>`, `raven:session:<id>`. Store
tags both owner tracks (host defaults from `ServiceLocator`, per-call
`metadata` overrides replace them for that passage only) and the session; recall
searches the user track, and agent-track content stays reachable through
`recall_session` — the contract's own read-back for a sub-agent's run.

## Config

Slice at `plugins.letta-memory` (the resolver also accepts `plugins.letta`).
Each key falls back to the environment variable shown, so a container can be
configured entirely by env (Kubernetes `envFrom` an ESO secret) without
editing `config.json`:

| key | env fallback | default | meaning |
|---|---|---|---|
| `url` | `LETTA_URL` | — | Letta server base URL. In-cluster: `http://letta-server.letta.svc.cluster.local:8283` — never the NodePort (same overlay network; internal DNS is the short path). Outside: your ingress. |
| `token` | `LETTA_TOKEN` | — | Bearer token = the server's `server_auth_secret` / `LETTA_SERVER_PASSWORD`. Treat as a secret: env from a secret store, not committed manifests. |
| `letta_agent_id` | `LETTA_AGENT_ID` | — | The **Letta** agent id whose archive holds this deployment's passages. Create it once (see below); it is a container on the server, not a Raven identity. |
| `recall_timeout_s` | — | 3.5 | Per-call search budget; stricter than the host's 5 s turn bound by design. |
| `store_timeout_s` | — | 20.0 | Write/health-side budget; store sits off the answer path. |
| `health_timeout_s` | — | 5.0 | `health()` probe bound. |
| `store_max_chars` | — | 8000 | Passage size cap; a longer turn is truncated with `…`. |

Enable it: `memory.backend = "letta"` in `~/.raven/config.json` (or
`raven.config.update.set_memory_backend`), then `raven doctor` should show
three `ok` letta lines.

## Design decisions (and their costs)

- **One synchronous passage per turn — no server-side LLM extraction.**
  Letta can let its agent decide what to archival, but that lands only while
  a heartbeat runs and no response proves the write. Direct insertion keeps
  `store()` honest (True = on the server now, provable by an independent
  REST query) and `flush` trivially honoured. The cost: passages are turn
  transcripts, not distilled facts, so recall quality depends on Letta's
  embedding search over verbatim text. A later "extract" variant can batch
  turns through the agent loop without breaking this contract.
- **`score` is rank decay (`1/(rank+1)`)**, because the search endpoint
  returns ordered results but no relevance number. Honest and monotone;
  callers must not read it as a confidence.
- **`recall_session` via tag-filtered search** — the list endpoint has no tag
  filter, and the search endpoint needs a query string. The tag filter does
  the real selecting; the returned ranking is meaningless noise as documented
  in the code. Capped at 200 passages.
- **Failures never raise** where the contract forbids it, but never silent
  either: warning log per failed call, one `notify` sentence when `start()`
  finds the config broken, and a `missing` line in `raven doctor`.

## Tests

- Contract + behaviour (MockTransport, in CI): `uv run --no-sync pytest
  plugins-dist/letta-memory/tests/test_contract.py -q`
- **Live** conformance (real server, throwaway agent, token by file path):
  `LETTA_LIVE=1 LETTA_URL=… LETTA_TOKEN_FILE=/path/0600 LETTA_AGENT_ID=…
  uv run --no-sync pytest plugins-dist/letta-memory/tests/test_live_letta.py
  -m letta_live -q`

## Creating the Letta agent

The archive container needs a dedicated Letta agent with an embedding model
(archival search is embedding-backed). On this cluster that is the shared
Letta server in namespace `letta`; the agent is created once at deploy time
by `deploy-raven.sh` (or by hand against `/v1/agents`), and its id goes to
`LETTA_AGENT_ID`.
