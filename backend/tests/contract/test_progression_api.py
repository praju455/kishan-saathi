"""HTTP contract tests for visible-progression comparison."""

from typing import Any
from uuid import UUID

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from app.core.dependencies import get_paid_operation_rate_limiter
from app.core.rate_limits import PaidOperationRateLimiter
from app.main import create_app
from app.modules.diagnoses.dependencies import get_diagnosis_progression_service
from app.modules.users.dependencies import get_current_farmer_id
from tests.contract.test_health_api import FakeDatabase, _test_settings

FARMER = UUID("00000000-0000-0000-0000-000000000001")


class FakeProgressionService:
    async def compare(self, farmer_id: UUID, case_id: UUID, selection: Any) -> dict[str, Any]:
        assert farmer_id == FARMER
        assert selection.response_language == "en"
        return {
            "comparison_id": UUID(int=9),
            "case_id": case_id,
            "earlier_assessment_id": UUID(int=10),
            "later_assessment_id": UUID(int=11),
            "earlier_captured_at": "2026-08-01T00:00:00Z",
            "later_captured_at": "2026-08-05T00:00:00Z",
            "earlier_image_ids": [UUID(int=20)],
            "later_image_ids": [UUID(int=21)],
            "trend": "unchanged",
            "confidence": 0.6,
            "evidence": ["affected_area_similar"],
            "limitations": [],
            "recommendations": ["monitor_same_leaf"],
            "image_quality": {"sufficient_for_comparison": True},
            "diagnosis_context": {
                "assessment_id": UUID(int=11),
                "crop_name": "Tomato",
                "disease_name": "tomato early blight",
                "confidence_label": "medium",
            },
            "model_name": "global.anthropic.claude-sonnet-4-6",
            "generated_at": "2026-08-13T00:00:00Z",
            "scope": "visible_symptom_progression_only",
        }


@pytest.mark.asyncio
async def test_progression_endpoint_requires_authentication() -> None:
    app = create_app(_test_settings(), FakeDatabase())
    async with (
        LifespanManager(app),
        AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client,
    ):
        response = await client.post(f"/api/v1/diagnoses/{UUID(int=1)}/progression", json={})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTH_TOKEN_REQUIRED"


@pytest.mark.asyncio
async def test_progression_endpoint_returns_typed_non_diagnostic_contract() -> None:
    app = create_app(_test_settings(), FakeDatabase())
    app.dependency_overrides[get_current_farmer_id] = lambda: FARMER
    app.dependency_overrides[get_diagnosis_progression_service] = FakeProgressionService
    app.dependency_overrides[get_paid_operation_rate_limiter] = lambda: PaidOperationRateLimiter(
        _test_settings()
    )
    async with (
        LifespanManager(app),
        AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client,
    ):
        response = await client.post(f"/api/v1/diagnoses/{UUID(int=1)}/progression", json={})

    assert response.status_code == 200
    assert response.json()["comparison_id"] == str(UUID(int=9))
    assert response.json()["trend"] == "unchanged"
    assert response.json()["recommendations"] == ["monitor_same_leaf"]
    assert response.json()["diagnosis_context"]["disease_name"] == "tomato early blight"
    assert response.json()["scope"] == "visible_symptom_progression_only"
    assert "diagnosis" not in response.json()
