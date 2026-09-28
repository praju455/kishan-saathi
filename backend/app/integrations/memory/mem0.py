"""Minimal async Mem0 Platform REST adapter."""

from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx

from app.core.errors import ApplicationError
from app.integrations.memory.provider import MemoryFact, MemoryProvider, MemoryScope


class Mem0MemoryProvider(MemoryProvider):
    """Use one Mem0 entity plus metadata to isolate farmer workspace memories."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url="https://api.mem0.ai",
            headers={"Authorization": f"Token {api_key}"},
            timeout=30,
        )
        self._owns_client = client is None

    async def search(
        self,
        *,
        scope: MemoryScope,
        query: str,
        limit: int,
    ) -> tuple[MemoryFact, ...]:
        payload = await self._request(
            "POST",
            "/v3/memories/search/",
            json={
                "query": query,
                "filters": self._scope_filters(scope),
                "top_k": limit,
            },
        )
        return self._facts(payload)

    async def index_fact(
        self,
        *,
        scope: MemoryScope,
        canonical_fact_id: UUID,
        text: str,
    ) -> MemoryFact:
        """Index idempotently using the canonical PostgreSQL fact ID as metadata."""

        existing = await self._request(
            "POST",
            "/v3/memories/",
            params={"page": 1, "page_size": 2},
            json={"filters": self._fact_filters(scope, canonical_fact_id)},
        )
        facts = self._facts(existing)
        if facts:
            return facts[0]
        payload = await self._request(
            "POST",
            "/v3/memories/add/",
            json={
                "messages": [{"role": "user", "content": text}],
                "user_id": str(scope.farmer_id),
                "metadata": {
                    "scope_key": self._scope_key(scope),
                    "canonical_fact_id": str(canonical_fact_id),
                },
                "infer": False,
                "async_mode": False,
            },
        )
        added = self._facts(payload)
        if len(added) != 1:
            raise ApplicationError(code="MEMORY_PROVIDER_INVALID_RESPONSE", status_code=502)
        return added[0]

    async def delete_fact(self, *, memory_id: str) -> None:
        await self._request(
            "DELETE",
            f"/v1/memories/{quote(memory_id, safe='')}/",
            allow_not_found=True,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_not_found: bool = False,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(method, path, json=json, params=params)
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ApplicationError(code="MEMORY_PROVIDER_UNAVAILABLE", status_code=503) from exc
        if allow_not_found and response.status_code == 404:
            return {}
        if response.status_code in {401, 403}:
            raise ApplicationError(code="MEMORY_AUTHENTICATION_FAILED", status_code=503)
        if response.status_code == 429 or response.status_code >= 500:
            raise ApplicationError(code="MEMORY_PROVIDER_UNAVAILABLE", status_code=503)
        if not response.is_success:
            raise ApplicationError(code="MEMORY_PROVIDER_REJECTED", status_code=502)
        if response.status_code == 204 or not response.content:
            return {}
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApplicationError(
                code="MEMORY_PROVIDER_INVALID_RESPONSE", status_code=502
            ) from exc
        if not isinstance(payload, dict):
            raise ApplicationError(code="MEMORY_PROVIDER_INVALID_RESPONSE", status_code=502)
        return payload

    @classmethod
    def _scope_filters(cls, scope: MemoryScope) -> dict[str, Any]:
        return {
            "AND": [
                {"user_id": str(scope.farmer_id)},
                {"metadata": {"scope_key": cls._scope_key(scope)}},
            ]
        }

    @classmethod
    def _fact_filters(cls, scope: MemoryScope, canonical_fact_id: UUID) -> dict[str, Any]:
        return {
            "AND": [
                {"user_id": str(scope.farmer_id)},
                {"metadata": {"scope_key": cls._scope_key(scope)}},
                {"metadata": {"canonical_fact_id": str(canonical_fact_id)}},
            ]
        }

    @staticmethod
    def _scope_key(scope: MemoryScope) -> str:
        return f"{scope.scope_type.value}:{scope.scope_id}"

    @staticmethod
    def _facts(payload: dict[str, Any]) -> tuple[MemoryFact, ...]:
        values = payload.get("results", [])
        if not isinstance(values, list):
            raise ApplicationError(code="MEMORY_PROVIDER_INVALID_RESPONSE", status_code=502)
        return tuple(
            MemoryFact(id=str(item["id"]), text=str(item["memory"]))
            for item in values
            if isinstance(item, dict) and item.get("id") and item.get("memory")
        )
