"""Integration tests for owner-scoped diagnosis progression comparison."""

from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.database.base import Base
from app.integrations.progression.provider import (
    ProgressionAnalysis,
    ProgressionEvidenceCode,
    ProgressionImageQuality,
    ProgressionLimitationCode,
    ProgressionProvider,
    ProgressionRecommendationCode,
    ProgressionRequest,
    ProgressionResult,
    ProgressionTrend,
)
from app.integrations.storage.local import LocalObjectStorage
from app.modules.diagnoses.models import (
    DiagnosisAssessment,
    DiagnosisCase,
    DiagnosisImage,
    DiagnosisPrediction,
)
from app.modules.diagnoses.progression import DiagnosisProgressionService
from app.modules.diagnoses.repository import DiagnosisRepository
from app.modules.diagnoses.schemas import ProgressionComparisonRequest
from app.modules.farms import models as _farm_models  # noqa: F401
from app.modules.users.models import FarmerProfile

FARMER = UUID("00000000-0000-0000-0000-000000000001")
OTHER = UUID("00000000-0000-0000-0000-000000000002")


class RecordingProgressionProvider(ProgressionProvider):
    def __init__(self) -> None:
        self.requests: list[ProgressionRequest] = []

    async def compare(self, request: ProgressionRequest) -> ProgressionResult:
        self.requests.append(request)
        return ProgressionResult(
            analysis=ProgressionAnalysis(
                trend=ProgressionTrend.IMPROVING,
                confidence=0.81,
                evidence=[ProgressionEvidenceCode.AFFECTED_AREA_REDUCED],
                limitations=[ProgressionLimitationCode.DIFFERENT_VIEWPOINT],
                recommendations=[ProgressionRecommendationCode.MONITOR_SAME_LEAF],
                image_quality=ProgressionImageQuality(sufficient_for_comparison=True),
            ),
            provider_response_id="resp_test",
            model="global.anthropic.claude-sonnet-4-6",
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_progression_success_uses_two_batches_without_diagnosis_labels(
    tmp_path: Path,
) -> None:
    context = await _context(tmp_path, include_later=True)
    try:
        response = await context.service.compare(
            FARMER,
            context.case_id,
            ProgressionComparisonRequest(response_language="hi"),
        )
    finally:
        await context.close()

    assert response.trend == "improving"
    assert response.scope == "visible_symptom_progression_only"
    assert response.earlier_captured_at < response.later_captured_at
    assert response.model_name == "global.anthropic.claude-sonnet-4-6"
    assert len(context.provider.requests) == 1
    request = context.provider.requests[0]
    assert request.response_language == "hi"
    assert request.classifier_context is not None
    assert request.classifier_context.crop_name == "Tomato"
    assert response.diagnosis_context is not None
    assert response.diagnosis_context.crop_name == "Tomato"
    assert response.recommendations == ["monitor_same_leaf"]
    assert request.earlier_images[0].content != request.later_images[0].content


@pytest.mark.asyncio
async def test_progression_hides_other_owner_and_rejects_insufficient_history(
    tmp_path: Path,
) -> None:
    context = await _context(tmp_path, include_later=False)
    try:
        with pytest.raises(ApplicationError) as hidden:
            await context.service.compare(OTHER, context.case_id, ProgressionComparisonRequest())
        with pytest.raises(ApplicationError) as insufficient:
            await context.service.compare(FARMER, context.case_id, ProgressionComparisonRequest())
    finally:
        await context.close()

    assert hidden.value.code == "DIAGNOSIS_CASE_NOT_FOUND"
    assert insufficient.value.code == "PROGRESSION_HISTORY_INSUFFICIENT"
    assert context.provider.requests == []


@pytest.mark.asyncio
async def test_progression_rejects_unusable_quality_before_provider(tmp_path: Path) -> None:
    context = await _context(tmp_path, include_later=True, later_quality=0.1)
    try:
        with pytest.raises(ApplicationError) as raised:
            await context.service.compare(FARMER, context.case_id, ProgressionComparisonRequest())
    finally:
        await context.close()

    assert raised.value.code == "PROGRESSION_IMAGE_QUALITY_INSUFFICIENT"
    assert raised.value.details == [{"timepoint": "LATER"}]
    assert context.provider.requests == []


class ProgressionHarness:
    def __init__(
        self,
        service: DiagnosisProgressionService,
        provider: RecordingProgressionProvider,
        case_id: UUID,
        engine: AsyncEngine,
        session: AsyncSession,
    ) -> None:
        self.service = service
        self.provider = provider
        self.case_id = case_id
        self.engine = engine
        self.session = session

    async def close(self) -> None:
        await self.session.close()
        await self.engine.dispose()


async def _context(
    tmp_path: Path,
    *,
    include_later: bool,
    later_quality: float = 0.9,
) -> ProgressionHarness:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session = async_sessionmaker(engine, expire_on_commit=False)()
    session.add_all(
        [
            FarmerProfile(
                id=FARMER,
                cognito_sub="farmer",
                cognito_username="farmer@example.com",
                email="farmer@example.com",
            ),
            FarmerProfile(
                id=OTHER,
                cognito_sub="other",
                cognito_username="other@example.com",
                email="other@example.com",
            ),
        ]
    )
    case_id = uuid4()
    session.add(
        DiagnosisCase(
            id=case_id,
            farmer_id=FARMER,
            title="Leaf comparison",
            status="completed",
        )
    )
    storage = LocalObjectStorage(tmp_path)
    now = datetime(2026, 8, 1, tzinfo=UTC)
    await _add_timepoint(session, storage, case_id, now, (40, 120, 30), 0.9)
    if include_later:
        await _add_timepoint(
            session,
            storage,
            case_id,
            now + timedelta(days=5),
            (80, 100, 25),
            later_quality,
        )
    await session.commit()
    provider = RecordingProgressionProvider()
    service = DiagnosisProgressionService(
        settings=Settings(_env_file=None),
        repository=DiagnosisRepository(session),
        storage=storage,
        provider=provider,
    )
    return ProgressionHarness(service, provider, case_id, engine, session)


async def _add_timepoint(
    session: AsyncSession,
    storage: LocalObjectStorage,
    case_id: UUID,
    captured_at: datetime,
    color: tuple[int, int, int],
    quality: float,
) -> None:
    data = BytesIO()
    Image.new("RGB", (256, 256), color=color).save(data, format="JPEG")
    content = data.getvalue()
    stored = await storage.put_private_image(owner_id=FARMER, category="diagnoses", content=content)
    image_id, assessment_id = uuid4(), uuid4()
    session.add(
        DiagnosisImage(
            id=image_id,
            farmer_id=FARMER,
            case_id=case_id,
            object_key=stored.key,
            size_bytes=len(content),
            width=256,
            height=256,
            captured_or_uploaded_at=captured_at,
            quality_score=quality,
            quality_flags=[],
        )
    )
    session.add(
        DiagnosisAssessment(
            id=assessment_id,
            farmer_id=FARMER,
            case_id=case_id,
            predicted_crop="Tomato",
            primary_disease="Classifier result",
            confidence=0.7,
            confidence_label="medium",
            model_name="trained-classifier",
            model_version="1",
            is_active=False,
            created_at=captured_at,
        )
    )
    session.add(
        DiagnosisPrediction(
            assessment_id=assessment_id,
            image_id=image_id,
            scope="image",
            rank=1,
            crop_name="Tomato",
            disease_name="Classifier result",
            confidence=0.7,
        )
    )
