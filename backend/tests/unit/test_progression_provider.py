"""Unit tests for the Bedrock visual-progression boundary."""

import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.integrations.bedrock.client import (
    BedrockConverseClient,
    BedrockFailure,
    BedrockFailureKind,
    BedrockResponse,
)
from app.integrations.progression.bedrock_converse import BedrockProgressionProvider
from app.integrations.progression.provider import (
    ProgressionAnalysis,
    ProgressionDiagnosisContext,
    ProgressionImage,
    ProgressionRequest,
)

FARMER = UUID("00000000-0000-0000-0000-000000000001")


class FakeClient:
    def __init__(self, response: BedrockResponse | Exception) -> None:
        self.response = response
        self.model_id = ""
        self.body: dict[str, Any] = {}

    async def converse(self, *, model_id: str, body: dict[str, Any]) -> BedrockResponse:
        self.model_id = model_id
        self.body = body
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_provider_uses_bedrock_images_and_structured_output() -> None:
    payload = {
        "trend": "improving",
        "confidence": 0.78,
        "evidence": ["affected_area_reduced"],
        "limitations": ["different_lighting"],
        "recommendations": ["monitor_same_leaf"],
        "image_quality": {
            "sufficient_for_comparison": True,
            "earlier_issues": [],
            "later_issues": [],
            "retake_guidance": [],
        },
    }
    client = FakeClient(_response(payload))
    provider = BedrockProgressionProvider(
        Settings(_env_file=None, bedrock_api_key="test"),
        client=cast(BedrockConverseClient, client),
    )

    result = await provider.compare(_request())

    assert result.analysis.trend.value == "improving"
    assert result.provider_response_id == "progression-request"
    assert client.model_id == "global.anthropic.claude-sonnet-4-6"
    content = client.body["messages"][0]["content"]
    image_parts = [item for item in content if "image" in item]
    assert len(image_parts) == 2
    assert all(item["image"]["source"]["bytes"] for item in image_parts)
    text_parts = [item["text"] for item in content if "text" in item]
    assert any("TRAINED_CLASSIFIER_CONTEXT" in text for text in text_parts)
    assert "outputConfig" in client.body


@pytest.mark.asyncio
async def test_provider_rejects_invalid_structured_output() -> None:
    client = FakeClient(_response({"trend": "not-an-enum"}))
    provider = BedrockProgressionProvider(
        Settings(_env_file=None, bedrock_api_key="test"),
        client=cast(BedrockConverseClient, client),
    )

    with pytest.raises(ApplicationError) as raised:
        await provider.compare(_request())

    assert raised.value.code == "PROGRESSION_PROVIDER_INVALID_RESPONSE"


@pytest.mark.asyncio
async def test_provider_maps_timeout_without_provider_details() -> None:
    client = FakeClient(BedrockFailure(BedrockFailureKind.TEMPORARY))
    provider = BedrockProgressionProvider(
        Settings(_env_file=None, bedrock_api_key="test"),
        client=cast(BedrockConverseClient, client),
    )

    with pytest.raises(ApplicationError) as raised:
        await provider.compare(_request())

    assert raised.value.code == "PROGRESSION_TEMPORARILY_UNAVAILABLE"
    assert raised.value.status_code == 503


def test_contract_accepts_only_controlled_codes_and_false_quality_certainty() -> None:
    base = {
        "confidence": 0.5,
        "evidence": ["affected_area_increased"],
        "limitations": [],
        "image_quality": {"sufficient_for_comparison": True},
    }
    with pytest.raises(ValidationError):
        ProgressionAnalysis.model_validate(
            {**base, "trend": "worsening", "evidence": ["This diagnoses early blight."]}
        )
    with pytest.raises(ValidationError):
        ProgressionAnalysis.model_validate(
            {**base, "trend": "worsening", "recommendations": ["Spray 2 ml per litre."]}
        )
    with pytest.raises(ValidationError):
        ProgressionAnalysis.model_validate(
            {
                **base,
                "trend": "improving",
                "image_quality": {"sufficient_for_comparison": False},
            }
        )


def _response(value: dict[str, Any]) -> BedrockResponse:
    return BedrockResponse(
        payload={
            "output": {"message": {"content": [{"text": json.dumps(value)}]}},
            "usage": {},
        },
        request_id="progression-request",
        request_count=1,
    )


def _request() -> ProgressionRequest:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    return ProgressionRequest(
        farmer_id=FARMER,
        earlier_images=(ProgressionImage(UUID(int=10), b"earlier", now),),
        later_images=(ProgressionImage(UUID(int=11), b"later", now),),
        response_language="hi",
        classifier_context=ProgressionDiagnosisContext(
            assessment_id=UUID(int=12),
            crop_name="tomato",
            disease_name="tomato early blight",
            confidence_label="high",
        ),
    )
