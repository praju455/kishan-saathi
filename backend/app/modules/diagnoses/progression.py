"""Owner-scoped orchestration for visible symptom progression comparison."""

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.integrations.progression.provider import (
    ProgressionDiagnosisContext,
    ProgressionImage,
    ProgressionProvider,
    ProgressionRequest,
)
from app.integrations.storage.provider import ObjectStorageProvider
from app.modules.diagnoses.models import (
    DiagnosisAssessment,
    DiagnosisImage,
    DiagnosisProgressionComparison,
)
from app.modules.diagnoses.repository import DiagnosisRepository
from app.modules.diagnoses.schemas import (
    ProgressionComparisonRequest,
    ProgressionComparisonResponse,
    ProgressionDiagnosisContextResponse,
)

_BLOCKING_QUALITY_FLAGS = frozenset(
    {
        "too_small",
        "blurry",
        "very_blurry",
        "underexposed",
        "overexposed",
        "extreme_aspect_ratio",
        "not_a_leaf",
    }
)
_PROGRESSION_PROMPT_VERSION = "2026-08-15"
_PROGRESSION_SCHEMA_VERSION = "1"


class DiagnosisProgressionService:
    """Validate history and limits before sending private bytes to the provider."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: DiagnosisRepository,
        storage: ObjectStorageProvider,
        provider: ProgressionProvider,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._storage = storage
        self._provider = provider

    async def compare(
        self,
        farmer_id: UUID,
        case_id: UUID,
        selection: ProgressionComparisonRequest,
    ) -> ProgressionComparisonResponse:
        # Owner scoping happens before provider work, and returns the same opaque 404 as case APIs.
        if await self._repository.get_case(farmer_id, case_id) is None:
            raise ApplicationError(code="DIAGNOSIS_CASE_NOT_FOUND", status_code=404)

        earlier, later = await self._timepoints(farmer_id, case_id, selection)
        earlier_images = await self._repository.assessment_images(farmer_id, case_id, earlier.id)
        later_images = await self._repository.assessment_images(farmer_id, case_id, later.id)
        earlier_images = self._usable_timepoint_images(earlier_images, "EARLIER")
        later_images = self._usable_timepoint_images(later_images, "LATER")
        if max(item.captured_or_uploaded_at for item in earlier_images) >= min(
            item.captured_or_uploaded_at for item in later_images
        ):
            raise ApplicationError(code="PROGRESSION_TIMEPOINT_ORDER_INVALID", status_code=422)

        earlier_images = earlier_images[: self._settings.progression_max_images_per_timepoint]
        later_images = later_images[: self._settings.progression_max_images_per_timepoint]
        request_hash = self._request_hash(
            earlier.id,
            later.id,
            [item.id for item in earlier_images],
            [item.id for item in later_images],
            selection.response_language.value,
        )
        existing = await self._repository.get_progression_by_hash(farmer_id, case_id, request_hash)
        if existing is not None:
            return self._response(existing)
        selected_images = [*earlier_images, *later_images]
        if any(image.size_bytes > self._settings.max_image_bytes for image in selected_images):
            raise ApplicationError(code="PROGRESSION_IMAGE_INVALID", status_code=422)
        if sum(image.size_bytes for image in selected_images) > (
            self._settings.progression_max_total_image_bytes
        ):
            raise ApplicationError(code="PROGRESSION_IMAGES_TOO_LARGE", status_code=413)
        # Release metadata reads before storage and Bedrock calls. Persistence below
        # starts a new, short transaction.
        await self._repository.commit()
        earlier_payload = await self._load_images(farmer_id, earlier_images)
        later_payload = await self._load_images(farmer_id, later_images)
        if sum(len(image.content) for image in (*earlier_payload, *later_payload)) > (
            self._settings.progression_max_total_image_bytes
        ):
            raise ApplicationError(code="PROGRESSION_IMAGES_TOO_LARGE", status_code=413)

        result = await self._provider.compare(
            ProgressionRequest(
                farmer_id=farmer_id,
                earlier_images=earlier_payload,
                later_images=later_payload,
                response_language=selection.response_language.value,
                classifier_context=ProgressionDiagnosisContext(
                    assessment_id=later.id,
                    crop_name=later.predicted_crop,
                    disease_name=later.primary_disease,
                    confidence_label=later.confidence_label,
                ),
            )
        )
        comparison = DiagnosisProgressionComparison(
            farmer_id=farmer_id,
            case_id=case_id,
            earlier_assessment_id=earlier.id,
            later_assessment_id=later.id,
            earlier_captured_at=min(item.captured_or_uploaded_at for item in earlier_images),
            later_captured_at=max(item.captured_or_uploaded_at for item in later_images),
            request_hash=request_hash,
            response_language=selection.response_language.value,
            earlier_image_ids=[str(item.id) for item in earlier_images],
            later_image_ids=[str(item.id) for item in later_images],
            trend=result.analysis.trend.value,
            confidence=result.analysis.confidence,
            evidence=[item.value for item in result.analysis.evidence],
            limitations=[item.value for item in result.analysis.limitations],
            recommendations=[item.value for item in result.analysis.recommendations],
            image_quality=result.analysis.image_quality.model_dump(mode="json"),
            diagnosis_context={
                "assessment_id": str(later.id),
                "crop_name": later.predicted_crop,
                "disease_name": later.primary_disease,
                "confidence_label": later.confidence_label,
            },
            model_name=result.model,
            provider_response_id=result.provider_response_id,
            prompt_version=_PROGRESSION_PROMPT_VERSION,
            schema_version=_PROGRESSION_SCHEMA_VERSION,
            generated_at=datetime.now(tz=UTC),
        )
        if await self._repository.get_case(farmer_id, case_id) is None:
            raise ApplicationError(code="DIAGNOSIS_CASE_NOT_FOUND", status_code=404)
        self._repository.add(comparison)
        try:
            await self._repository.commit()
        except IntegrityError:
            await self._repository.rollback()
            existing = await self._repository.get_progression_by_hash(
                farmer_id, case_id, request_hash
            )
            if existing is None:
                raise ApplicationError(
                    code="PROGRESSION_PERSISTENCE_FAILED", status_code=503
                ) from None
            return self._response(existing)
        await self._repository.refresh(comparison)
        return self._response(comparison)

    async def history(
        self, farmer_id: UUID, case_id: UUID, *, limit: int, offset: int
    ) -> list[ProgressionComparisonResponse]:
        if await self._repository.get_case(farmer_id, case_id) is None:
            raise ApplicationError(code="DIAGNOSIS_CASE_NOT_FOUND", status_code=404)
        return [
            self._response(item)
            for item in await self._repository.list_progression_comparisons(
                farmer_id, case_id, limit=limit, offset=offset
            )
        ]

    @staticmethod
    def _request_hash(
        earlier_assessment_id: UUID,
        later_assessment_id: UUID,
        earlier_image_ids: list[UUID],
        later_image_ids: list[UUID],
        response_language: str,
    ) -> str:
        payload = json.dumps(
            {
                "earlier_assessment_id": str(earlier_assessment_id),
                "later_assessment_id": str(later_assessment_id),
                "earlier_image_ids": [str(value) for value in earlier_image_ids],
                "later_image_ids": [str(value) for value in later_image_ids],
                "response_language": response_language,
                "prompt_version": _PROGRESSION_PROMPT_VERSION,
                "schema_version": _PROGRESSION_SCHEMA_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _response(value: DiagnosisProgressionComparison) -> ProgressionComparisonResponse:
        return ProgressionComparisonResponse(
            comparison_id=value.id,
            case_id=value.case_id,
            earlier_assessment_id=value.earlier_assessment_id,
            later_assessment_id=value.later_assessment_id,
            earlier_captured_at=value.earlier_captured_at,
            later_captured_at=value.later_captured_at,
            earlier_image_ids=[UUID(item) for item in value.earlier_image_ids],
            later_image_ids=[UUID(item) for item in value.later_image_ids],
            trend=value.trend,
            confidence=value.confidence,
            evidence=value.evidence,
            limitations=value.limitations,
            recommendations=value.recommendations,
            image_quality=value.image_quality,
            diagnosis_context=(
                ProgressionDiagnosisContextResponse.model_validate(value.diagnosis_context)
                if value.diagnosis_context is not None
                else None
            ),
            model_name=value.model_name,
            generated_at=value.generated_at,
        )

    async def _timepoints(
        self,
        farmer_id: UUID,
        case_id: UUID,
        selection: ProgressionComparisonRequest,
    ) -> tuple[DiagnosisAssessment, DiagnosisAssessment]:
        if selection.earlier_assessment_id is not None:
            if selection.later_assessment_id is None:
                raise ApplicationError(code="PROGRESSION_TIMEPOINT_PAIR_REQUIRED", status_code=422)
            earlier = await self._repository.get_assessment(
                farmer_id, case_id, selection.earlier_assessment_id
            )
            later = await self._repository.get_assessment(
                farmer_id, case_id, selection.later_assessment_id
            )
            if earlier is None or later is None:
                raise ApplicationError(code="PROGRESSION_TIMEPOINT_NOT_FOUND", status_code=404)
        else:
            newest = await self._repository.list_assessments(
                farmer_id,
                case_id,
                limit=self._settings.progression_history_scan_limit,
            )
            valid: list[tuple[datetime, DiagnosisAssessment]] = []
            for assessment in newest:
                images = await self._repository.assessment_images(farmer_id, case_id, assessment.id)
                if images:
                    valid.append(
                        (
                            max(image.captured_or_uploaded_at for image in images),
                            assessment,
                        )
                    )
            if len(valid) < 2:
                raise ApplicationError(code="PROGRESSION_HISTORY_INSUFFICIENT", status_code=422)
            valid.sort(key=lambda item: item[0], reverse=True)
            later, earlier = valid[0][1], valid[1][1]

        return earlier, later

    def _usable_timepoint_images(
        self, images: list[DiagnosisImage], label: str
    ) -> list[DiagnosisImage]:
        if not images:
            raise ApplicationError(code="PROGRESSION_HISTORY_INSUFFICIENT", status_code=422)
        usable = [image for image in images if self._is_usable(image)]
        if not usable:
            raise ApplicationError(
                code="PROGRESSION_IMAGE_QUALITY_INSUFFICIENT",
                status_code=422,
                details=[{"timepoint": label}],
            )
        return usable

    def _is_usable(self, image: DiagnosisImage) -> bool:
        if min(image.width, image.height) < self._settings.progression_min_image_dimension:
            return False
        if image.quality_score is not None and (
            image.quality_score < self._settings.progression_min_image_quality_score
        ):
            return False
        return not _BLOCKING_QUALITY_FLAGS.intersection(image.quality_flags)

    async def _load_images(
        self, farmer_id: UUID, images: list[DiagnosisImage]
    ) -> tuple[ProgressionImage, ...]:
        payload: list[ProgressionImage] = []
        for image in images:
            try:
                content = await self._storage.read_private(
                    owner_id=farmer_id,
                    key=image.object_key,
                )
            except (OSError, ValueError) as exc:
                raise ApplicationError(
                    code="PROGRESSION_IMAGE_UNAVAILABLE", status_code=503
                ) from exc
            if len(content) != image.size_bytes or len(content) > self._settings.max_image_bytes:
                raise ApplicationError(code="PROGRESSION_IMAGE_INVALID", status_code=422)
            payload.append(
                ProgressionImage(
                    image_id=image.id,
                    content=content,
                    captured_at=image.captured_or_uploaded_at,
                )
            )
        return tuple(payload)
