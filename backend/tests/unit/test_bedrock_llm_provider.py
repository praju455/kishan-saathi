"""Focused tests for the Bedrock Converse LLM boundary."""

import json
from typing import Any, cast
from uuid import UUID

import pytest

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.integrations.bedrock.client import (
    BedrockConverseClient,
    BedrockFailure,
    BedrockFailureKind,
    BedrockResponse,
)
from app.integrations.llm.bedrock_converse import BedrockConverseProvider
from app.integrations.llm.provider import LLMRequest, LLMTask, LLMTool

FARMER = UUID("00000000-0000-0000-0000-000000000001")


class FakeBedrockClient:
    def __init__(self, responses: list[BedrockResponse | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def converse(self, *, model_id: str, body: dict[str, Any]) -> BedrockResponse:
        self.calls.append({"model_id": model_id, "body": body})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_provider_uses_structured_output_and_safety_review() -> None:
    client = FakeBedrockClient(
        [
            _core_response({"short_answer": "Share the crop, symptoms, and timing."}),
            _actions_response(),
            _response({"unsafe": False, "reason_codes": []}, "safety"),
        ]
    )
    provider = BedrockConverseProvider(
        Settings(_env_file=None, bedrock_api_key="test"),
        client=cast(BedrockConverseClient, client),
    )

    result = await provider.respond(
        LLMRequest(
            task=LLMTask.ROUTINE_CHAT,
            instructions="Answer safely.",
            input_text="What should I share?",
            farmer_id=FARMER,
        ),
        model="global.anthropic.claude-sonnet-4-6",
    )

    assert result.reply.short_answer.startswith("Share")
    assert result.policy_reviewed is True
    assert result.provider_response_id == "chat"
    output_config = client.calls[0]["body"]["outputConfig"]
    schema = json.loads(output_config["textFormat"]["structure"]["jsonSchema"]["schema"])
    assert schema["additionalProperties"] is False
    assert "maxLength" not in json.dumps(schema)
    assert client.calls[2]["body"]["system"][0]["text"].startswith("Review only")


@pytest.mark.asyncio
async def test_provider_executes_only_declared_empty_argument_tools() -> None:
    called: list[str] = []

    async def weather() -> str:
        called.append("weather")
        return '{"temperature_c":28}'

    client = FakeBedrockClient(
        [
            BedrockResponse(
                payload={
                    "output": {
                        "message": {
                            "content": [
                                {
                                    "toolUse": {
                                        "toolUseId": "tool-1",
                                        "name": "get_plot_weather",
                                        "input": {},
                                    }
                                }
                            ]
                        }
                    },
                    "usage": {},
                },
                request_id="tool",
                request_count=1,
            ),
            _core_response({"short_answer": "It is warm at the linked plot."}),
            _actions_response(),
            _response({"unsafe": False, "reason_codes": []}, "safety"),
        ]
    )
    provider = BedrockConverseProvider(
        Settings(_env_file=None, bedrock_api_key="test"),
        client=cast(BedrockConverseClient, client),
    )

    await provider.respond(
        LLMRequest(
            task=LLMTask.AGRICULTURAL_GUIDANCE,
            instructions="Use the trusted tool.",
            input_text="How is the plot?",
            farmer_id=FARMER,
            tools=(
                LLMTool(
                    name="get_plot_weather",
                    description="Get owner-authorized plot weather",
                    execute=weather,
                ),
            ),
        ),
        model="global.anthropic.claude-sonnet-4-6",
    )

    assert called == ["weather"]
    tool_result = client.calls[1]["body"]["messages"][-1]["content"][0]["toolResult"]
    assert tool_result["toolUseId"] == "tool-1"


@pytest.mark.asyncio
async def test_provider_rebuilds_nested_contract_from_flat_bedrock_wire_reply() -> None:
    assessment_id = UUID("00000000-0000-0000-0000-000000000099")
    client = FakeBedrockClient(
        [
            _core_response(
                {
                    "short_answer": "The saved assessment is compatible with early blight.",
                    "certainty": "confirmed_context",
                }
            ),
            _actions_response(
                {
                    "diagnosis_assessment_id": str(assessment_id),
                    "diagnosis_crop_name": "Tomato",
                    "diagnosis_disease_name": "tomato early blight",
                    "diagnosis_confidence_label": "medium",
                    "retake_reason_codes": ["blur"],
                    "retake_instructions": ["Take one sharp close photo."],
                    "reminder_title": "Take another leaf photo",
                    "reminder_due_at": "2030-01-02T03:04:05Z",
                    "reminder_recurrence_days": 0,
                }
            ),
            _response({"unsafe": False, "reason_codes": []}, "safety"),
        ]
    )
    provider = BedrockConverseProvider(
        Settings(_env_file=None, bedrock_api_key="test"),
        client=cast(BedrockConverseClient, client),
    )

    result = await provider.respond(
        LLMRequest(
            task=LLMTask.AGRICULTURAL_GUIDANCE,
            instructions="Use only the supplied classifier evidence.",
            input_text="Summarize the saved assessment.",
            farmer_id=FARMER,
            allow_diagnosis=True,
        ),
        model="global.anthropic.claude-sonnet-4-6",
    )

    assert result.reply.diagnosis_discussion is not None
    assert result.reply.diagnosis_discussion.assessment_id == assessment_id
    assert result.reply.retake_advice is not None
    assert result.reply.reminder_proposal is not None
    assert result.reply.reminder_proposal.title == "Take another leaf photo"


@pytest.mark.asyncio
async def test_provider_maps_bedrock_authentication_failure() -> None:
    client = FakeBedrockClient([BedrockFailure(BedrockFailureKind.AUTHENTICATION)])
    provider = BedrockConverseProvider(
        Settings(_env_file=None, bedrock_api_key="test"),
        client=cast(BedrockConverseClient, client),
    )

    with pytest.raises(ApplicationError) as raised:
        await provider.classify_chat_risk(
            content="hello",
            farmer_id=FARMER,
            model="global.anthropic.claude-sonnet-4-6",
        )

    assert raised.value.code == "LLM_ROUTING_UNAVAILABLE"
    assert raised.value.status_code == 503


def _response(value: dict[str, Any], request_id: str) -> BedrockResponse:
    return BedrockResponse(
        payload={
            "output": {"message": {"content": [{"text": json.dumps(value)}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        },
        request_id=request_id,
        request_count=1,
    )


def _core_response(overrides: dict[str, Any]) -> BedrockResponse:
    value: dict[str, Any] = {
        "short_answer": "A careful answer.",
        "disposition": "in_scope",
        "details": "",
        "next_steps": [],
        "follow_up_questions": [],
        "certainty": "possible",
        "general_precautions": [],
        "consult_local_expert": False,
    }
    value.update(overrides)
    return _response(value, "chat")


def _actions_response(overrides: dict[str, Any] | None = None) -> BedrockResponse:
    value: dict[str, Any] = {
        "diagnosis_assessment_id": "",
        "diagnosis_crop_name": "",
        "diagnosis_disease_name": "",
        "diagnosis_confidence_label": "",
        "retake_reason_codes": [],
        "retake_instructions": [],
        "reminder_title": "",
        "reminder_due_at": "",
        "reminder_recurrence_days": 0,
    }
    value.update(overrides or {})
    return _response(value, "actions")
