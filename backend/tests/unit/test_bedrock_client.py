"""HTTP contract tests for the Bedrock API-key transport."""

import httpx
import pytest

from app.core.config import Settings
from app.integrations.bedrock.client import (
    BedrockConverseClient,
    BedrockFailure,
    BedrockFailureKind,
)


@pytest.mark.asyncio
async def test_transport_sends_bearer_key_without_logging_or_global_state() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret-key"
        assert request.url.path.endswith("/converse")
        return httpx.Response(
            200,
            headers={"x-amzn-requestid": "request-1"},
            json={"output": {"message": {"content": [{"text": "{}"}]}}},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = BedrockConverseClient(
        Settings(_env_file=None, bedrock_api_key="secret-key", bedrock_max_retries=0),
        http_client=http,
    )

    response = await client.converse(
        model_id="global.anthropic.claude-sonnet-4-6",
        body={"messages": []},
    )

    assert response.request_id == "request-1"
    await http.aclose()


@pytest.mark.asyncio
async def test_transport_maps_access_denied() -> None:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(403, json={}))
    )
    client = BedrockConverseClient(
        Settings(_env_file=None, bedrock_api_key="bad", bedrock_max_retries=0),
        http_client=http,
    )

    with pytest.raises(BedrockFailure) as raised:
        await client.converse(model_id="model", body={"messages": []})

    assert raised.value.kind is BedrockFailureKind.ACCESS_DENIED
    await http.aclose()
