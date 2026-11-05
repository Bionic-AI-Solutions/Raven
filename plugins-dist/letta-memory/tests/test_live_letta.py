"""Conformance against the REAL Letta server.

Run only where a live server + throwaway agent are configured (never in
shared CI):

    LETTA_LIVE=1 LETTA_URL=... LETTA_TOKEN_FILE=/path/0600 \\
    LETTA_AGENT_ID=<throwaway-agent> uv run pytest \\
        plugins-dist/letta-memory/tests/test_live_letta.py -m letta_live -q

The token is read from a file path (env ``LETTA_TOKEN_FILE``), never from a
command line or this transcript. Every test cleans up its own passages.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

from raven_letta.backend import LettaBackend

LIVE = os.environ.get("LETTA_LIVE") == "1"
BASE = os.environ.get("LETTA_URL", "").rstrip("/")
AGENT = os.environ.get("LETTA_AGENT_ID", "")
_TOKEN_FILE = os.environ.get("LETTA_TOKEN_FILE", "")
TOKEN = open(_TOKEN_FILE).read().strip() if _TOKEN_FILE else ""

pytestmark = [
    pytest.mark.letta_live,
    pytest.mark.skipif(
        not (LIVE and BASE and AGENT and TOKEN),
        reason="set LETTA_LIVE=1, LETTA_URL, LETTA_TOKEN_FILE, LETTA_AGENT_ID",
    ),
]


def _backend() -> LettaBackend:
    return LettaBackend(
        {"url": BASE, "token": TOKEN, "letta_agent_id": AGENT},
        user_id="letta-live-test",
        agent_id="letta-live-test",
    )


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


async def _raw_list_tags() -> list[dict]:
    """Independent query: list passages straight from the list endpoint —
    not through the plugin — to prove what actually landed."""
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{BASE}/v1/agents/{AGENT}/archival-memory?limit=100", headers=_auth())
        resp.raise_for_status()
        return resp.json()


@pytest.fixture
async def marker() -> str:
    token = f"letta-live-{uuid.uuid4().hex[:12]}"
    yield token
    # Cleanup: delete any passage carrying this run's marker.
    for p in await _raw_list_tags():
        tags = p.get("tags") or []
        text = p.get("text") or ""
        if any(token in t for t in tags) or token in text:
            async with httpx.AsyncClient(timeout=15) as client:
                await client.delete(
                    f"{BASE}/v1/agents/{AGENT}/archival-memory/{p['id']}",
                    headers=_auth(),
                )


async def test_round_trip_agreement_and_removal(marker: str) -> None:
    backend = _backend()
    await backend.start()
    try:
        session_id = f"live-{marker}"
        content = f"{marker}: the cluster gateway rotates its consumer key quarterly"
        assert await backend.store(session_id, [{"role": "user", "content": content}])

        # Agreement (removal comes below): an INDEPENDENT list query shows
        # the passage with the plugin's tags — not just that store() said True.
        raw = await _raw_list_tags()
        landed = [p for p in raw if marker in (p.get("text") or "")]
        assert len(landed) == 1, f"expected exactly one passage for {marker}, got {len(landed)}"
        assert f"raven:session:{session_id}" in (landed[0].get("tags") or [])

        hits = await backend.recall("gateway consumer key rotation", user_id="letta-live-test", top_k=5)
        assert any(marker in h.text for h in hits), "stored turn did not surface in recall"

        session_hits = await backend.recall_session(session_id)
        assert any(marker in h.text for h in session_hits)

        # Removal AND agreement: delete, then the independent query must not
        # show it any more.
        pid = landed[0]["id"]
        assert await backend.delete(pid) is True
        raw_after = await _raw_list_tags()
        assert not any(marker in (p.get("text") or "") for p in raw_after)
    finally:
        await backend.stop()


async def test_health_ready_against_live_server() -> None:
    backend = _backend()
    health = await backend.health()
    assert health.ready is True, health.checks
    assert {c.status for c in health.checks} == {"ok"}
