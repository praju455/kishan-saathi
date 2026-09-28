"""Mem0 REST adapter contract tests for scope isolation and idempotency."""

import json
from typing import Any
from uuid import UUID

import httpx
import pytest

from app.integrations.memory.mem0 import Mem0MemoryProvider
from app.integrations.memory.provider import MemoryScope, MemoryScopeType

FARMER = UUID("00000000-0000-0000-0000-000000000081")
PLOT = UUID("00000000-0000-0000-0000-000000000082")
FACT = UUID("00000000-0000-0000-0000-000000000083")


@pytest.mark.asyncio
async def test_mem0_uses_rest_api_with_scope_metadata_and_idempotency() -> None:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    existing = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal existing
        assert request.headers["authorization"] == "Token test"
        payload = json.loads(request.content) if request.content else {}
        calls.append((request.method, request.url.path, payload))
        if request.url.path == "/v3/memories/search/":
            return httpx.Response(200, json={"results": [{"id": "remote-1", "memory": "Fact"}]})
        if request.url.path == "/v3/memories/":
            return httpx.Response(
                200,
                json={
                    "results": (
                        [{"id": "remote-1", "memory": "Irrigation completed"}] if existing else []
                    )
                },
            )
        if request.url.path == "/v3/memories/add/":
            existing = True
            return httpx.Response(
                200,
                json={"results": [{"id": "remote-1", "memory": "Irrigation completed"}]},
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"message": "deleted"})
        return httpx.Response(500)

    http = httpx.AsyncClient(
        base_url="https://api.mem0.ai",
        headers={"Authorization": "Token test"},
        transport=httpx.MockTransport(handler),
    )
    provider = Mem0MemoryProvider("test", client=http)
    scope = MemoryScope(FARMER, MemoryScopeType.PLOT, PLOT)

    result = await provider.search(scope=scope, query="irrigation", limit=5)
    indexed = await provider.index_fact(
        scope=scope,
        canonical_fact_id=FACT,
        text="Irrigation completed",
    )
    reused = await provider.index_fact(
        scope=scope,
        canonical_fact_id=FACT,
        text="Irrigation completed",
    )
    await provider.delete_fact(memory_id="remote-1")

    assert result[0].id == "remote-1"
    assert indexed.id == reused.id == "remote-1"
    search_payload = calls[0][2]
    assert search_payload["filters"] == {
        "AND": [
            {"user_id": str(FARMER)},
            {"metadata": {"scope_key": f"plot:{PLOT}"}},
        ]
    }
    add_payload = next(payload for _, path, payload in calls if path.endswith("/add/"))
    assert add_payload["infer"] is False
    assert add_payload["metadata"]["canonical_fact_id"] == str(FACT)
    assert sum(path.endswith("/add/") for _, path, _ in calls) == 1
    await http.aclose()
