"""Small async transport for Amazon Bedrock's provider-neutral Converse API."""

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import Settings


class BedrockFailureKind(StrEnum):
    AUTHENTICATION = "authentication"
    ACCESS_DENIED = "access_denied"
    CONFIGURATION = "configuration"
    TEMPORARY = "temporary"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER = "provider"


class BedrockFailure(Exception):
    """Provider-neutral failure that feature adapters map to their own API codes."""

    def __init__(self, kind: BedrockFailureKind) -> None:
        super().__init__(kind.value)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class BedrockResponse:
    payload: dict[str, Any]
    request_id: str
    request_count: int


class BedrockConverseClient:
    """Call Converse with a Bedrock API key without process-global credential state."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = f"https://bedrock-runtime.{settings.bedrock_region}.amazonaws.com"
        self._headers = {
            "Authorization": f"Bearer {settings.bedrock_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        self._http = http_client or httpx.AsyncClient(
            timeout=settings.bedrock_request_timeout_seconds
        )
        self._owns_http_client = http_client is None
        self._max_retries = settings.bedrock_max_retries

    async def converse(self, *, model_id: str, body: dict[str, Any]) -> BedrockResponse:
        url = f"{self._base_url}/model/{quote(model_id, safe='')}/converse"
        request_count = 0
        for attempt in range(self._max_retries + 1):
            request_count += 1
            try:
                response = await self._http.post(url, json=body, headers=self._headers)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < self._max_retries:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                raise BedrockFailure(BedrockFailureKind.TEMPORARY) from exc
            failure = self._failure_for_response(response)
            if failure is not None:
                if failure is BedrockFailureKind.TEMPORARY and attempt < self._max_retries:
                    await asyncio.sleep(0.25 * (2**attempt))
                    continue
                raise BedrockFailure(failure)
            try:
                payload = response.json()
            except ValueError as exc:
                raise BedrockFailure(BedrockFailureKind.INVALID_RESPONSE) from exc
            if not isinstance(payload, dict):
                raise BedrockFailure(BedrockFailureKind.INVALID_RESPONSE)
            return BedrockResponse(
                payload=payload,
                request_id=response.headers.get("x-amzn-requestid", "not-provided"),
                request_count=request_count,
            )
        raise BedrockFailure(BedrockFailureKind.TEMPORARY)

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()

    @staticmethod
    def _failure_for_response(response: httpx.Response) -> BedrockFailureKind | None:
        if response.is_success:
            return None
        if response.status_code == 401:
            return BedrockFailureKind.AUTHENTICATION
        if response.status_code == 403:
            return BedrockFailureKind.ACCESS_DENIED
        if response.status_code in {400, 404}:
            return BedrockFailureKind.CONFIGURATION
        if response.status_code in {408, 409, 429} or response.status_code >= 500:
            return BedrockFailureKind.TEMPORARY
        return BedrockFailureKind.PROVIDER
