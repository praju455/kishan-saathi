"""Task-policy based model routing without feature-specific condition chains."""

from dataclasses import dataclass
from uuid import UUID

from app.core.config import Settings
from app.integrations.llm.provider import (
    ChatRiskClassification,
    GeneratedTitle,
    LLMProvider,
    LLMRequest,
    LLMResult,
    LLMTask,
    MemoryExtraction,
    MemoryExtractionRequest,
)


@dataclass(frozen=True, slots=True)
class ModelPolicy:
    model: str


class LLMRouter:
    """Map typed tasks to configuration-owned model policies."""

    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        configured = ModelPolicy(settings.bedrock_model_id)
        self._provider = provider
        self._policies = {
            LLMTask.TITLE: configured,
            LLMTask.ROUTING: configured,
            LLMTask.EXTRACTION: configured,
            LLMTask.ROUTINE_CHAT: configured,
            LLMTask.AGRICULTURAL_GUIDANCE: configured,
        }

    async def respond(self, request: LLMRequest) -> LLMResult:
        return await self._provider.respond(request, model=self._policies[request.task].model)

    async def extract_memories(self, request: MemoryExtractionRequest) -> MemoryExtraction:
        return await self._provider.extract_memories(
            request, model=self._policies[LLMTask.EXTRACTION].model
        )

    async def generate_title(
        self, *, content: str, language: str, farmer_id: UUID
    ) -> GeneratedTitle:
        return await self._provider.generate_title(
            content=content,
            language=language,
            farmer_id=farmer_id,
            model=self._policies[LLMTask.TITLE].model,
        )

    async def classify_chat_risk(self, *, content: str, farmer_id: UUID) -> ChatRiskClassification:
        return await self._provider.classify_chat_risk(
            content=content,
            farmer_id=farmer_id,
            model=self._policies[LLMTask.ROUTING].model,
        )
