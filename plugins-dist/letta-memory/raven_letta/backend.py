"""Letta memory backend for Raven.

Writes each turn into a Letta server's archival memory and reads recalls back
from it, so one Raven deployment's learnings live in one queryable Letta
archive beside the Subconscious memories, addressable by REST from anywhere.

The contract this satisfies is :class:`raven.contracts.memory.MemoryBackend`:
a flat per-turn read/write side. The wire is the Letta v1 REST API, verified
against a live server (Letta 0.16.7):

* ``GET  {base}/v1/agents/{letta_agent_id}/archival-memory/search``
  — semantic ``query`` plus ``tags`` / ``tag_match_mode`` filters, bounded by
  ``top_k``. This is ``recall`` and ``recall_session``.
* ``POST {base}/v1/agents/{letta_agent_id}/archival-memory``
  — ``{"text", "tags"}`` creates one passage. This is ``store``.
* ``DELETE {base}/v1/agents/{letta_agent_id}/archival-memory/{memory_id}``
  — removes one passage. This is ``delete``.
* ``GET  {base}/v1/agents/{letta_agent_id}``
  — bounded reachability + auth + existence probe. This is ``health``.

Every passage is tagged so the tracks and sessions stay separable within one
Letta agent's archive:

* ``raven`` — everything this plugin wrote (a filter for outside tools);
* ``raven:user:<id>`` / ``raven:agent:<id>`` — the owner track(s) of the
  content, per the contract's owner-track rule: a turn's default is the host's
  configured identities, and a per-call ``metadata`` override replaces the
  default for that call only. Store tags BOTH tracks (a sub-agent's write is
  attributed and stays readable through ``recall_session``); ``recall``
  answers the flat-backend convention the host documents for Letta: the user
  track searches, the agent track returns ``[]`` — turn transcripts are not
  skills and BackendSkillSource must not see them.
* ``raven:session:<id>`` — which conversation wrote the passage, which is how
  ``recall_session`` filters, since the list endpoint has no tag filter.

Failure policy is the contract's: ``store`` returns a bool and never raises,
``recall`` returns what resolved (``[]`` on any error) and never raises. The
turn that loses its memory must still answer — but it must not answer silently
either, so every failed call logs at warning with the Letta-side cause, and a
backend that was broken at ``start`` says so once through the host's
``notify``. ``raven doctor`` reports the state as a ``missing`` health line
rather than an absent one.

Config comes from the plugin's own slice (see ``raven-plugin.toml``), each key
falling back to an environment variable of the same intent so a Kubernetes
deployment can mount the credentials as env without editing config.json:
``url``/``LETTA_URL``, ``token``/``LETTA_TOKEN``,
``letta_agent_id``/``LETTA_AGENT_ID``. The in-cluster target is the internal
service DNS (``http://letta-server.letta.svc.cluster.local:8283``), never the
NodePort — same network, one fewer hop, and the NodePort only exists for
hosts outside the cluster.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from raven.contracts.memory import BackendHealth, HealthCheck, Memory
from raven.contracts.plugin_surface import ServiceLocator
from raven.plugins import PluginContext

__all__ = ["LettaBackend", "make_backend"]

__version__ = "0.1.0"

# Fallbacks under the manifest defaults. The manifest is authoritative for a
# build that goes through the registry (admission applies its defaults before
# the factory boards); these cover a directly-constructed backend (tests) and
# a manifest-less admission pass-through.
_DEFAULT_RECALL_TIMEOUT_S = 3.5
_DEFAULT_STORE_TIMEOUT_S = 20.0
_DEFAULT_HEALTH_TIMEOUT_S = 5.0
_DEFAULT_STORE_MAX_CHARS = 8000
# A connect that hangs for its full budget costs the turn that whole wait;
# two seconds finds a dead or wrong-address server long before that.
_CONNECT_TIMEOUT_S = 2.0

# One session's worth of passages returned by recall_session. The endpoint
# ranks by query similarity, which is meaningless once the tag filter has done
# the selecting; the cap only bounds the response, and 200 passages is far
# more than one conversation can store.
_SESSION_RECALL_LIMIT = 200

_TAG_NS = "raven"
_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")


def _num(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _tag_safe(value: str) -> str:
    """Coerce an identity into something tag-charset safe."""
    slug = _TAG_SAFE_RE.sub("-", value.strip())[:120].strip("-")
    return slug


class LettaBackend:
    """A :class:`MemoryBackend` over a Letta server's archival memory.

    ``letta_agent_id`` (config ``letta_agent_id`` / ``LETTA_AGENT_ID``) names
    the *Letta* agent whose archive holds this deployment's passages — a
    container on the server, not a Raven identity. Raven's owner identities
    (whose memories these are) come from the host's :class:`ServiceLocator`,
    the single source of truth, and ride as tags; never re-enter them here.

    ``_transport_factory`` exists for tests only: it swaps the transport (an
    ``httpx.MockTransport`` for the contract suite) while the client — auth
    header, timeouts — stays the plugin's own construction under test.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        user_id: str = "",
        agent_id: str = "",
        notify: Any = None,
        logger: logging.Logger | None = None,
        _transport_factory: Any = None,
    ) -> None:
        self._config: dict[str, Any] = dict(config or {})
        # Host identities captured at build time: the per-turn owner for
        # content stored without a per-call override. NOT duplicated config.
        self._default_user_id = user_id
        self._default_agent_id = agent_id
        self.log = logger or logging.getLogger("raven.plugins.letta-memory")
        self.notify = notify or self._log_notice
        self._transport_factory = _transport_factory
        self._client: httpx.AsyncClient | None = None
        self._stopping = False
        self._start_problem: str | None = None
        self._auth_problem_logged = False

    # ── config resolution (at use time, never at build time) ───────────

    def _resolve(self, key: str, env: str) -> str:
        raw = self._config.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return (os.environ.get(env) or "").strip()

    @property
    def _base_url(self) -> str:
        return self._resolve("url", "LETTA_URL").rstrip("/")

    @property
    def _token(self) -> str:
        return self._resolve("token", "LETTA_TOKEN")

    @property
    def _letta_agent_id(self) -> str:
        return self._resolve("letta_agent_id", "LETTA_AGENT_ID")

    def _timeout(self, key: str, fallback: float) -> httpx.Timeout:
        return httpx.Timeout(_num(self._config.get(key), fallback), connect=_CONNECT_TIMEOUT_S)

    # ── HTTP plumbing ──────────────────────────────────────────────────

    def _client_or_new(self) -> httpx.AsyncClient:
        """The shared client, (re)built when absent or previously stopped.

        Built here rather than in ``__init__`` because construction must not
        read the environment (the registry builds a backend just to health-
        check it), and rebuilt after ``stop`` so a restarted lifecycle works.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "User-Agent": f"raven-letta/{__version__}",
                },
                timeout=_CONNECT_TIMEOUT_S,
                # The live server answers no-trailing-slash paths with 307,
                # which preserves method+body; httpx does not follow by
                # default, and the schema's own examples carry both shapes.
                follow_redirects=True,
                transport=self._transport_factory() if self._transport_factory else None,
            )
        return self._client

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout: httpx.Timeout,
    ) -> tuple[int | None, Any, str | None]:
        """One bounded request → ``(status, parsed_json_or_None, error)``.

        ``error`` is a human-shaped cause (``None`` on a clean round trip);
        a status is still returned for HTTP errors, because 404 means
        different things to different callers. Never raises, never echoes
        the token.
        """
        base = self._base_url
        if not base:
            return None, None, "no Letta URL configured (url / LETTA_URL)"
        url = f"{base}{path}"
        try:
            resp = await self._client_or_new().request(
                method,
                url,
                params=dict(params or {}),
                json=dict(json_body or {}) if json_body is not None else None,
                timeout=timeout,
            )
        except httpx.TimeoutException:
            return None, None, f"timed out after {timeout.timeout if hasattr(timeout, 'timeout') else '?'}s: {base}"
        except httpx.HTTPError as exc:
            return None, None, f"cannot reach Letta at {base} ({type(exc).__name__})"
        payload: Any = None
        if resp.content:
            try:
                payload = resp.json()
            except (json.JSONDecodeError, ValueError):
                payload = None
        if resp.status_code in (401, 403):
            return resp.status_code, payload, "Letta rejected the token (HTTP %d)" % resp.status_code
        return resp.status_code, payload, None

    # ── tags ───────────────────────────────────────────────────────────

    @staticmethod
    def _track_tag(kind: str, owner_id: str) -> str:
        return f"{_TAG_NS}:{kind}:{_tag_safe(owner_id)}"

    def _session_tag(self, session_id: str) -> str:
        return f"{_TAG_NS}:session:{_tag_safe(session_id)}"

    # ── MemoryBackend ──────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        top_k: int = 5,
    ) -> list[Memory]:
        """Semantic search over the user track; ``[]`` for the agent track.

        Returns whatever resolved; empty on absence AND on failure (the
        contract's leniency) — the failure is logged at warning so the two
        differ in the log, and ``health`` tells the operator which.
        """
        if user_id is None and agent_id is None:
            self.log.warning("recall called without an owner track; skipping")
            return []
        if user_id is None and agent_id is not None:
            # Flat-backend convention (mem0 / MemOS / Letta, per the host's
            # own MemoryConfig docs): the agent track feeds
            # BackendSkillSource, and turn-transcript passages are not
            # skills. Passages tagged raven:agent:<id> are still real and
            # readable — via recall_session, the contract's own mechanism
            # for "what did this sub-agent's run leave behind".
            return []
        hits, err = await self._search(
            query,
            tags=[self._track_tag("user", user_id)],
            top_k=top_k,
            timeout=self._timeout("recall_timeout_s", _DEFAULT_RECALL_TIMEOUT_S),
        )
        if err:
            self.log.warning("recall failed (%d hits returned as empty): %s", len(hits), err)
        return hits

    async def store(
        self,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Append one turn as one passage. Always lands synchronously.

        There is no deferral to defer: the write happens on the call, so
        ``metadata["flush"]`` is honoured unconditionally (a backend that
        wrote later would have to honour it specially; this one has no
        later). A per-call ``user_id``/``agent_id`` in metadata replaces the
        host default for this passage's tags only — the turn an end-turn
        tool's write belongs to must not be indexed under the host owner.
        """
        meta = metadata or {}
        flush = bool(meta.get("flush", True))
        text = self._render(messages)
        if not text.strip():
            self.log.debug("store: nothing to record (empty turn); not writing")
            return True
        tags = [_TAG_NS, self._session_tag(session_id)]
        user_id = meta.get("user_id") or self._default_user_id
        agent_id = meta.get("agent_id") or self._default_agent_id
        if user_id:
            tags.append(self._track_tag("user", str(user_id)))
        if agent_id:
            tags.append(self._track_tag("agent", str(agent_id)))
        max_chars = int(_num(self._config.get("store_max_chars"), _DEFAULT_STORE_MAX_CHARS))
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        status, _, err = await self._request(
            "POST",
            f"/v1/agents/{self._letta_agent_id}/archival-memory",
            json_body={"text": text, "tags": tags},
            timeout=self._timeout("store_timeout_s", _DEFAULT_STORE_TIMEOUT_S),
        )
        if err or status is None or not (200 <= status < 300):
            self.log.warning(
                "store of session %s turn did not land (flush=%s): %s",
                session_id,
                flush,
                err or f"HTTP {status}",
            )
            return False
        return True

    async def feedback(self, signal: Mapping[str, Any]) -> None:
        """No-op. Raven's feedback signals have no Letta-side consumer yet.

        Logged once per kind at debug so the "nothing consumes this" fact is
        greppable rather than inferred from an empty method.
        """
        kind = signal.get("kind") or signal.get("signal_type") or "?"
        self.log.debug("feedback signal %r has no Letta-side handler; ignored", kind)

    async def start(self) -> None:
        """Probe the configured target once and make a broken setup known.

        Deliberately does NOT raise: raising strips the session of long-term
        memory with no per-turn explanation (the contract's warning). The
        user gets one actionable sentence; the operator gets the ``doctor``
        health line; every turn still works, with empty recall.
        """
        self._stopping = False
        if not self._base_url or not self._token or not self._letta_agent_id:
            missing = [
                name
                for name, have in (
                    ("LETTA_URL", self._base_url),
                    ("LETTA_TOKEN", self._token),
                    ("LETTA_AGENT_ID", self._letta_agent_id),
                )
                if not have
            ]
            self._start_problem = "no long-term memory this session: " + ", ".join(missing) + " not set"
        else:
            _, _, err = await self._request(
                "GET",
                f"/v1/agents/{self._letta_agent_id}",
                timeout=self._timeout("health_timeout_s", _DEFAULT_HEALTH_TIMEOUT_S),
            )
            self._start_problem = f"no long-term memory this session: {err}" if err else None
        if self._start_problem:
            self.log.warning(self._start_problem)
            self._notify_once(self._start_problem)

    async def stop(self) -> None:
        """Close the client. A backend whose start failed must still stop."""
        self._stopping = True
        client, self._client = self._client, None
        if client is not None and not client.is_closed:
            try:
                await client.aclose()
            except Exception as exc:  # pragma: no cover - closing is best-effort
                self.log.debug("closing the Letta client raised: %s", exc)

    async def health(self) -> BackendHealth:
        """A live bounded probe of URL + token + agent, as doctor shows it."""
        checks: list[HealthCheck] = []
        if not self._base_url or not self._token or not self._letta_agent_id:
            missing = [
                name
                for name, have in (
                    ("url", self._base_url),
                    ("token", self._token),
                    ("letta_agent_id", self._letta_agent_id),
                )
                if not have
            ]
            checks.append(
                HealthCheck(
                    label="letta:config",
                    status="missing",
                    hint="unset: " + ", ".join(missing),
                )
            )
            return BackendHealth(ready=False, checks=checks)
        status, _, err = await self._request(
            "GET",
            f"/v1/agents/{self._letta_agent_id}",
            timeout=self._timeout("health_timeout_s", _DEFAULT_HEALTH_TIMEOUT_S),
        )
        if err:
            checks.append(HealthCheck(label="letta:server", status="missing", hint=err))
            return BackendHealth(ready=False, checks=checks)
        checks.append(HealthCheck(label="letta:server", status="ok", hint=self._base_url))
        if status == 404:
            checks.append(
                HealthCheck(
                    label="letta:agent",
                    status="missing",
                    hint=(f"agent {self._letta_agent_id!r} not found on the server — create it or fix letta_agent_id"),
                )
            )
            return BackendHealth(ready=False, checks=checks)
        checks.append(HealthCheck(label="letta:agent", status="ok", hint=self._letta_agent_id))
        return BackendHealth(ready=True, checks=checks)

    async def delete(self, memory_id: str, *, kind: str | None = None) -> bool:
        """Remove one passage by its Letta id. ``False`` means it did not go.

        The caller is a person clicking a button; the two answers they must
        tell apart are "gone" and "this could not be removed here". A 404 is
        the ordinary shape of a second click on a row someone already
        removed: not deleted, not an error.
        """
        status, _, err = await self._request(
            "DELETE",
            f"/v1/agents/{self._letta_agent_id}/archival-memory/{memory_id}",
            timeout=self._timeout("store_timeout_s", _DEFAULT_STORE_TIMEOUT_S),
        )
        if err:
            self.log.warning("delete of %s failed: %s", memory_id, err)
            return False
        if status is not None and 200 <= status < 300:
            return True
        if status == 404:
            return False
        self.log.warning("delete of %s: unexpected HTTP %s", memory_id, status)
        return False

    async def recall_session(self, session_id: str) -> list[Memory]:
        """Passages tagged with this session, newest store first in rank order.

        The tag filter is the real selector; the query only has to be there
        for the endpoint to answer, so the ranking it produces is noise the
        caller treats as "this session wrote these".
        """
        hits, err = await self._search(
            "conversation",
            tags=[self._session_tag(session_id)],
            top_k=_SESSION_RECALL_LIMIT,
            timeout=self._timeout("recall_timeout_s", _DEFAULT_RECALL_TIMEOUT_S),
        )
        if err:
            self.log.warning("recall_session(%s) returned empty: %s", session_id, err)
        return hits

    # ── shared pieces ──────────────────────────────────────────────────

    async def _search(
        self,
        query: str,
        *,
        tags: Sequence[str],
        top_k: int,
        timeout: httpx.Timeout,
    ) -> tuple[list[Memory], str | None]:
        """Search + project to :class:`Memory`; ``(hits, error)``."""
        status, payload, err = await self._request(
            "GET",
            f"/v1/agents/{self._letta_agent_id}/archival-memory/search",
            params={
                "query": query,
                "top_k": max(1, int(top_k)),
                "tags": list(tags),
                "tag_match_mode": "all",
            },
            timeout=timeout,
        )
        if err:
            return [], err
        if status is None or not (200 <= status < 300):
            return [], f"search returned HTTP {status}"
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            return [], f"unexpected search payload shape: {type(payload).__name__}"
        hits: list[Memory] = []
        for rank, item in enumerate(results[: max(1, int(top_k))]):
            if not isinstance(item, dict):
                continue
            text = item.get("content") or item.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            item_tags = item.get("tags")
            hits.append(
                Memory(
                    text=text,
                    # The search endpoint ranks but exposes no score, so the
                    # rank IS the signal: 1/(rank+1) keeps it in the contract's
                    # 0..1 normalisation and ordered, and says so rather than
                    # faking a relevance the server never sent.
                    score=round(1.0 / (rank + 1), 4),
                    metadata={
                        "id": item.get("id"),
                        "tags": item_tags if isinstance(item_tags, list) else [],
                        "source": "letta",
                    },
                )
            )
        return hits, None

    def _render(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Project the turn's messages onto plain text, one labelled line each.

        Multimodal parts collapse to their text; a part with no text (an
        image) is labelled so the passage says so rather than silently
        dropping it.
        """
        lines: list[str] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or "user")
            content = message.get("content")
            if isinstance(content, str):
                body = content.strip()
            elif isinstance(content, Sequence):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                    elif isinstance(part, Mapping):
                        parts.append(f"[{part.get('type', 'part')}]")
                    else:
                        parts.append(str(part))
                body = " ".join(p.strip() for p in parts if str(p).strip()).strip()
            elif content is None:
                body = ""
            else:
                body = str(content).strip()
            if body:
                lines.append(f"{role}: {body}")
        return "\n".join(lines)

    def _log_notice(self, msg: str) -> None:
        self.log.warning(msg)

    def _notify_once(self, msg: str) -> None:
        if self._auth_problem_logged:
            return
        self._auth_problem_logged = True
        try:
            self.notify(msg)
        except Exception:  # pragma: no cover - a host renderer must not break us
            self.log.debug("notify raised; the warning above is the record")


def make_backend(ctx: PluginContext) -> LettaBackend:
    """Registry entry for the ``letta`` memory backend (manifest contribution).

    Read-only and zero IO by the registry's contract: the config slice is
    copied and the host's owner identities captured from the locator — the
    ONE source for those ids; reading them from this plugin's own config
    would put the same value in two files, and a one-sided edit makes every
    written memory unrecallable with no warning. The first connection
    happens at ``start`` or on the first turn, never here.
    """
    services: ServiceLocator | None = getattr(ctx, "services", None)
    return LettaBackend(
        ctx.config,
        user_id=getattr(services, "user_id", "") or "",
        agent_id=getattr(services, "agent_id", "") or "",
        notify=getattr(services, "notify", None),
        logger=ctx.logger,
    )
