"""Run non-destructive connectivity checks against configured external providers."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from uuid import UUID

import httpx
from PIL import Image, ImageDraw

from app.core.config import get_settings
from app.integrations.llm.bedrock_converse import BedrockConverseProvider
from app.integrations.llm.provider import LLMRequest, LLMTask
from app.integrations.memory.mem0 import Mem0MemoryProvider
from app.integrations.memory.provider import MemoryScope, MemoryScopeType
from app.integrations.progression.bedrock_converse import BedrockProgressionProvider
from app.integrations.progression.provider import ProgressionImage, ProgressionRequest
from app.integrations.weather.openweather import OpenWeatherCurrentProvider

FARMER_ID = UUID("00000000-0000-0000-0000-000000000001")
SCOPE_ID = UUID("00000000-0000-0000-0000-000000000002")


async def _check_cognito() -> dict[str, object]:
    settings = get_settings()
    issuer = (
        f"https://cognito-idp.{settings.cognito_region}.amazonaws.com/"
        f"{settings.cognito_user_pool_id}"
    )
    async with httpx.AsyncClient(timeout=settings.external_request_timeout_seconds) as client:
        jwks = await client.get(f"{issuer}/.well-known/jwks.json")
        jwks.raise_for_status()
        keys = jwks.json().get("keys", [])
        probe = await client.post(
            f"https://cognito-idp.{settings.cognito_region}.amazonaws.com/",
            headers={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
            },
            json={
                "AuthFlow": "USER_PASSWORD_AUTH",
                "ClientId": settings.cognito_app_client_id,
                "AuthParameters": {
                    "USERNAME": "provider-smoke@example.invalid",
                    "PASSWORD": "NotARealPassword-42!",
                },
            },
        )
        payload = probe.json()
    return {
        "jwks_key_count": len(keys),
        "app_client_probe_status": probe.status_code,
        "app_client_probe_code": str(payload.get("__type", "unknown")).split("#")[-1],
    }


async def _check_weather() -> dict[str, object]:
    provider = OpenWeatherCurrentProvider(get_settings())
    try:
        result = await provider.current(latitude=12.9716, longitude=77.5946)
        return {
            "location": result.location_name,
            "condition": result.condition,
            "temperature_c": result.temperature_c,
        }
    finally:
        await provider.close()


async def _check_mem0() -> dict[str, object]:
    provider = Mem0MemoryProvider(get_settings().mem0_api_key)
    result = await provider.search(
        scope=MemoryScope(FARMER_ID, MemoryScopeType.PLOT, SCOPE_ID),
        query="provider connectivity check",
        limit=1,
    )
    return {"reachable": True, "result_count": len(result)}


async def _check_bedrock() -> dict[str, object]:
    settings = get_settings()
    provider = BedrockConverseProvider(settings)
    try:
        routing = await provider.classify_chat_risk(
            content=json.dumps(
                {
                    "effective_scope": "general",
                    "has_farm_context": False,
                    "has_plot_context": False,
                    "farmer_message": "Hello, what can you help me with?",
                }
            ),
            farmer_id=FARMER_ID,
            model=settings.bedrock_model_id,
        )
        response = await provider.respond(
            LLMRequest(
                task=LLMTask.ROUTINE_CHAT,
                instructions=(
                    "You are a careful farmer assistant. Answer in simple English. "
                    "Do not provide chemical products, dosage, or application schedules."
                ),
                input_text="What information should I include when asking about a crop problem?",
                farmer_id=FARMER_ID,
            ),
            model=settings.bedrock_model_id,
        )
        return {
            "routing_model": settings.bedrock_model_id,
            "requires_primary_model": routing.requires_primary_model,
            "response_model": response.model,
            "short_answer": response.reply.short_answer,
            "follow_up_question_count": len(response.reply.follow_up_questions),
        }
    finally:
        await provider.close()


def _test_leaf_image(*, affected_radius: int) -> bytes:
    image = Image.new("RGB", (256, 256), (50, 130, 55))
    ImageDraw.Draw(image).ellipse(
        (
            128 - affected_radius,
            128 - affected_radius,
            128 + affected_radius,
            128 + affected_radius,
        ),
        fill=(115, 70, 30),
    )
    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


async def _check_bedrock_progression() -> dict[str, object]:
    provider = BedrockProgressionProvider(get_settings())
    captured = datetime.now(tz=UTC)
    try:
        result = await provider.compare(
            ProgressionRequest(
                farmer_id=FARMER_ID,
                earlier_images=(
                    ProgressionImage(UUID(int=10), _test_leaf_image(affected_radius=54), captured),
                ),
                later_images=(
                    ProgressionImage(
                        UUID(int=11),
                        _test_leaf_image(affected_radius=30),
                        captured + timedelta(days=3),
                    ),
                ),
                response_language="en",
            )
        )
        return {
            "model": result.model,
            "trend": result.analysis.trend,
            "confidence": result.analysis.confidence,
            "scope": "visible_symptom_progression_only",
        }
    finally:
        await provider.close()


async def main() -> None:
    checks = {
        "cognito": _check_cognito,
        "openweather": _check_weather,
        "mem0": _check_mem0,
        "bedrock": _check_bedrock,
        "bedrock_progression": _check_bedrock_progression,
    }
    results: dict[str, object] = {}
    for name, check in checks.items():
        try:
            results[name] = {"ok": True, "details": await check()}
        except Exception as exc:  # this is a diagnostic entry point
            results[name] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
