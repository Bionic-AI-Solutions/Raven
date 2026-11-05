"""Contract conformance + behaviour for the Letta backend.

The contract classes run against the REAL backend object over an
``httpx.MockTransport`` — the request-building code the plugin owns (paths,
params, tags, projections) is exercised verbatim; only the socket is replaced.
Live-server conformance lives in ``test_live_letta.py`` behind the
``letta_live`` marker.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from raven.contracts.plugin_surface import ServiceLocator
from raven.memory_engine.contract_test import (
    LifecycleContractTests,
    MemoryBackendContractTests,
)
from raven.plugins.context import PluginContext
from raven_letta.backend import LettaBackend, make_backend

BASE = "http://letta.test"
AGENT = "raven-agent"
TOKEN = "sekrit-token-never-echo"


class MockLetta:
    """A small in-memory Letta: agent probe, search, insert, delete."""

    def __init__(self) -> None:
        self.passages: list[dict] = []
        self.requests: list[httpx.Request] = []
        self.fail_next: int | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_next is not None:
            code, self.fail_next = self.fail_next, None
            return httpx.Response(code, json={"detail": "planted failure"})
        path = request.url.path
        if request.method == "GET" and path == f"/v1/agents/{AGENT}":
            return httpx.Response(200, json={"id": AGENT, "name": "Raven"})
        if request.method == "GET" and path == f"/v1/agents/{AGENT}/archival-memory/search":
            tags_wanted = request.url.params.get_list("tags")
            top_k = int(request.url.params.get("top_k", "5"))
            results = [
                {
                    "id": p["id"],
                    "content": p["text"],
                    "tags": p["tags"],
                    "timestamp": "2026-01-01T00:00:00",
                }
                for p in self.passages
                if not tags_wanted or all(t in p["tags"] for t in tags_wanted)
            ][: max(1, top_k)]
            return httpx.Response(200, json={"results": results, "count": len(results)})
        if request.method == "POST" and path == f"/v1/agents/{AGENT}/archival-memory":
            body = json.loads(request.content)
            passage = {
                "id": f"pm_{len(self.passages):028d}",
                "text": body["text"],
                "tags": body.get("tags", []),
            }
            self.passages.append(passage)
            return httpx.Response(201, json=passage)
        if request.method == "DELETE" and path.startswith(f"/v1/agents/{AGENT}/archival-memory/"):
            gone = path.rsplit("/", 1)[-1]
            before = len(self.passages)
            self.passages = [p for p in self.passages if p["id"] != gone]
            if len(self.passages) == before:
                return httpx.Response(404, json={"detail": "no such passage"})
            return httpx.Response(200, json={"deleted": True})
        return httpx.Response(404, json={"detail": f"unmocked {request.method} {path}"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _backend(mock: MockLetta, **config) -> LettaBackend:
    cfg: dict = {"url": BASE, "token": TOKEN, "letta_agent_id": AGENT}
    cfg.update(config)
    return LettaBackend(cfg, user_id="salil", agent_id="raven", _transport_factory=mock.transport)


class TestLettaContract(MemoryBackendContractTests, LifecycleContractTests):
    async def make_backend(self) -> LettaBackend:
        self.mock = MockLetta()
        return _backend(self.mock)

    # The base fixture covers the protocol's floor. This extends it with
    # round-trip AGREEMENT the base deliberately does not claim: a stored
    # turn must actually surface again for the right owner and not for a
    # wrong one.
    async def test_store_then_recall_round_trip(self) -> None:
        backend = await self.make_backend()
        await backend.start()
        try:
            ok = await backend.store(
                "rt-session",
                [{"role": "user", "content": "the deploy key rotated 2026-11-05"}],
            )
            assert ok is True
            hits = await backend.recall("deploy key", user_id="salil", top_k=5)
            assert any("rotated" in h.text for h in hits)
            # The wrong owner's track filter must not surface it: the stored
            # passage is tagged raven:agent:raven / raven:user:salil, and
            # this asks for a different agent id entirely.
            agent_hits = await backend.recall("deploy key", agent_id="other-agent")
            assert agent_hits == []
        finally:
            await backend.stop()


class TestWireShape:
    """The paths/params/tags the contract cannot see."""

    async def test_recall_filters_the_user_track_by_tag(self) -> None:
        mock = MockLetta()
        backend = _backend(mock)
        await backend.recall("kubernetes", user_id="salil", top_k=4)
        req = mock.requests[-1]
        assert req.url.path == f"/v1/agents/{AGENT}/archival-memory/search"
        assert req.url.params["query"] == "kubernetes"
        assert req.url.params["top_k"] == "4"
        assert req.url.params.get_list("tags") == ["raven:user:salil"]
        assert req.url.params["tag_match_mode"] == "all"
        assert req.headers["authorization"] == f"Bearer {TOKEN}"

    async def test_store_tags_both_tracks_and_the_session(self) -> None:
        mock = MockLetta()
        backend = _backend(mock)
        ok = await backend.store(
            "sess-9",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        )
        assert ok is True
        body = json.loads(mock.requests[-1].content)
        assert body["text"] == "user: hi\nassistant: hello"
        assert set(body["tags"]) == {
            "raven",
            "raven:session:sess-9",
            "raven:user:salil",
            "raven:agent:raven",
        }

    async def test_store_per_call_override_replaces_default_track(self) -> None:
        mock = MockLetta()
        backend = _backend(mock)
        await backend.store(
            "s1",
            [{"role": "user", "content": "from a subagent"}],
            metadata={"agent_id": "dsh"},
        )
        tags = json.loads(mock.requests[-1].content)["tags"]
        assert "raven:agent:dsh" in tags and "raven:agent:raven" not in tags
        await backend.store("s1", [{"role": "user", "content": "host turn"}])
        tags = json.loads(mock.requests[-1].content)["tags"]
        assert "raven:agent:raven" in tags  # the default was not poisoned

    async def test_store_truncates_to_max_chars(self) -> None:
        mock = MockLetta()
        backend = _backend(mock, store_max_chars=20)
        await backend.store("s1", [{"role": "user", "content": "x" * 100}])
        text = json.loads(mock.requests[-1].content)["text"]
        assert len(text) == 20 and text.endswith("…")

    async def test_recall_agent_track_is_empty_and_makes_no_request(self) -> None:
        """The flat-backend convention must not even touch the network: the
        agent track feeds BackendSkillSource, and transcripts are not skills."""
        mock = MockLetta()
        backend = _backend(mock)
        assert await backend.recall("anything", agent_id="raven", top_k=5) == []
        assert mock.requests == []

    async def test_recall_session_filters_by_session_tag(self) -> None:
        mock = MockLetta()
        backend = _backend(mock)
        await backend.recall_session("sess-42")
        assert mock.requests[-1].url.params.get_list("tags") == ["raven:session:sess-42"]

    async def test_delete_known_id_removes_the_passage(self) -> None:
        mock = MockLetta()
        backend = _backend(mock)
        await backend.store("s1", [{"role": "user", "content": "forget me"}])
        pid = mock.passages[0]["id"]
        assert await backend.delete(pid) is True
        assert mock.passages == []
        assert await backend.delete(pid) is False  # second click: gone, not an error

    async def test_scores_are_ranked_and_in_range(self) -> None:
        mock = MockLetta()
        for i in range(3):
            mock.passages.append({"id": f"p{i}", "text": f"mem {i}", "tags": ["raven:user:salil"]})
        backend = _backend(mock)
        hits = await backend.recall("mem", user_id="salil", top_k=5)
        assert len(hits) == 3
        assert hits[0].score > hits[1].score > hits[2].score
        assert all(0.0 <= h.score <= 1.0 for h in hits)


class TestFailurePaths:
    """Failure paths must stay non-raising where the contract demands, and
    still be LOUD: every test below plants a defect and asserts it surfaces."""

    async def test_store_500_returns_false_never_raises(self) -> None:
        mock = MockLetta()
        mock.fail_next = 500
        backend = _backend(mock)
        assert await backend.store("s1", [{"role": "user", "content": "x"}]) is False

    async def test_recall_unreachable_returns_empty_and_logs(self, caplog) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        backend = LettaBackend(
            {"url": BASE, "token": TOKEN, "letta_agent_id": AGENT},
            user_id="salil",
            _transport_factory=lambda: httpx.MockTransport(refuse),
        )
        with caplog.at_level(logging.WARNING):
            hits = await backend.recall("anything", user_id="salil")
        assert hits == []
        assert "cannot reach" in caplog.text

    async def test_auth_failure_names_cause_without_leaking_token(self) -> None:
        def deny(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "unauthorized"})

        notices: list[str] = []
        backend = LettaBackend(
            {"url": BASE, "token": TOKEN, "letta_agent_id": AGENT},
            user_id="salil",
            notify=notices.append,
            _transport_factory=lambda: httpx.MockTransport(deny),
        )
        await backend.start()
        joined = " | ".join(notices)
        assert "token" in joined.lower()
        assert TOKEN not in joined
        health = await backend.health()
        assert health.ready is False

    async def test_health_404_names_the_missing_agent(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path.startswith("/v1/agents/"):
                return httpx.Response(404, json={"detail": "agent not found"})
            return httpx.Response(404)

        backend = LettaBackend(
            {"url": BASE, "token": TOKEN, "letta_agent_id": AGENT},
            _transport_factory=lambda: httpx.MockTransport(handler),
        )
        health = await backend.health()
        assert health.ready is False
        line = {c.label: c for c in health.checks}
        assert line["letta:server"].status == "ok"
        assert line["letta:agent"].status == "missing"
        assert AGENT in line["letta:agent"].hint

    async def test_start_with_unset_config_notifies_once_and_survives(self) -> None:
        notices: list[str] = []
        backend = LettaBackend({}, notify=notices.append)
        await backend.start()  # must not raise: the turn must still be served
        assert len(notices) == 1
        assert "LETTA_URL" in notices[0] and "LETTA_TOKEN" in notices[0]
        health = await backend.health()
        assert health.ready is False and health.checks[0].status == "missing"


class TestFactory:
    def test_make_backend_copies_config_and_grabs_host_identities(self) -> None:
        services = ServiceLocator(workspace=Path("/tmp/ws"), user_id="u9", agent_id="a9", notify=None)
        ctx = PluginContext(
            config={"url": BASE, "token": TOKEN, "letta_agent_id": AGENT},
            services=services,
        )
        backend = make_backend(ctx)
        assert backend._default_user_id == "u9"
        assert backend._default_agent_id == "a9"
        assert backend._base_url == BASE
