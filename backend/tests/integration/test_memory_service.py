"""Integration tests for filtered, farmer-controlled scoped memory."""

from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.database.base import Base
from app.integrations.llm.provider import (
    AssistantReply,
    ChatRiskClassification,
    ChatRiskReason,
    GeneratedTitle,
    LLMProvider,
    LLMRequest,
    LLMResult,
    MemoryCandidate,
    MemoryExtraction,
    MemoryExtractionRequest,
)
from app.integrations.llm.router import LLMRouter
from app.integrations.memory.provider import MemoryFact, MemoryProvider, MemoryScope
from app.modules.chats.models import ChatMessage, ChatSession
from app.modules.chats.repository import ChatRepository
from app.modules.diagnoses.models import DiagnosisCase
from app.modules.farms.models import Farm, Plot
from app.modules.farms.repository import FarmRepository
from app.modules.memories.cleaner import MemoryDiagnosisContextCleaner
from app.modules.memories.models import MemoryCaptureJob, ScopedMemoryFact
from app.modules.memories.repository import MemoryRepository
from app.modules.memories.schemas import (
    ChatMemoryConnectionCreate,
    MemoryConnectionTarget,
)
from app.modules.memories.service import MemoryService
from app.modules.users.models import FarmerProfile

FARMER = UUID("00000000-0000-0000-0000-000000000061")
OTHER = UUID("00000000-0000-0000-0000-000000000062")


class ExtractingLLM(LLMProvider):
    def __init__(self) -> None:
        self.requests: list[tuple[MemoryExtractionRequest, str]] = []

    async def respond(self, request: LLMRequest, *, model: str) -> LLMResult:
        del request
        return LLMResult(AssistantReply(short_answer="ok"), "id", model, True)

    async def extract_memories(
        self, request: MemoryExtractionRequest, *, model: str
    ) -> MemoryExtraction:
        self.requests.append((request, model))
        first_line = next(line for line in request.transcript.splitlines() if "role=user:" in line)
        message_id = UUID(first_line.split()[0].split("=", 1)[1])
        return MemoryExtraction(
            facts=[
                MemoryCandidate(
                    text="I planted tomato in June",
                    source_message_id=message_id,
                    evidence_quote="I planted tomato in June",
                ),
                MemoryCandidate(
                    text="  i   planted TOMATO in June  ",
                    source_message_id=message_id,
                    evidence_quote="I planted tomato in June",
                ),
            ]
        )

    async def close(self) -> None:
        return None

    async def generate_title(
        self, *, content: str, language: str, farmer_id: UUID, model: str
    ) -> GeneratedTitle:
        del content, language, farmer_id, model
        return GeneratedTitle(title="Memory chat")

    async def classify_chat_risk(
        self, *, content: str, farmer_id: UUID, model: str
    ) -> ChatRiskClassification:
        del content, farmer_id, model
        return ChatRiskClassification(
            requires_primary_model=True,
            reason_code=ChatRiskReason.UNCERTAIN,
        )


class RecordingMemory(MemoryProvider):
    def __init__(self) -> None:
        self.adds: list[tuple[MemoryScope, tuple[str, ...]]] = []
        self.deletes: list[tuple[MemoryScope, str]] = []

    async def search(self, *, scope: MemoryScope, query: str, limit: int) -> tuple[MemoryFact, ...]:
        del scope, query, limit
        return ()

    async def list(self, *, scope: MemoryScope, limit: int) -> tuple[MemoryFact, ...]:
        del scope, limit
        return ()

    async def index_fact(
        self, *, scope: MemoryScope, canonical_fact_id: UUID, text: str
    ) -> MemoryFact:
        del canonical_fact_id
        self.adds.append((scope, (text,)))
        return MemoryFact("remote-0", text)

    async def delete_fact(self, *, memory_id: str) -> None:
        self.deletes.append((self.adds[0][0], memory_id))

    async def close(self) -> None:
        return None


class FailingExtractionLLM(ExtractingLLM):
    async def extract_memories(
        self, request: MemoryExtractionRequest, *, model: str
    ) -> MemoryExtraction:
        del request, model
        raise ApplicationError(code="LLM_EXTRACTION_UNAVAILABLE", status_code=503)


class FailingMemory(RecordingMemory):
    async def index_fact(
        self, *, scope: MemoryScope, canonical_fact_id: UUID, text: str
    ) -> MemoryFact:
        del scope, canonical_fact_id, text
        raise ApplicationError(code="MEMORY_PROVIDER_UNAVAILABLE", status_code=503)


class DeletingMemory(RecordingMemory):
    def __init__(self) -> None:
        super().__init__()
        self.deleted_ids: list[str] = []

    async def delete_fact(self, *, memory_id: str) -> None:
        self.deleted_ids.append(memory_id)


class FlakyDeletingMemory(DeletingMemory):
    def __init__(self, failing_id: str) -> None:
        super().__init__()
        self.failing_id = failing_id
        self.failed_once = False

    async def delete_fact(self, *, memory_id: str) -> None:
        if memory_id == self.failing_id and not self.failed_once:
            self.failed_once = True
            raise ApplicationError(code="MEMORY_PROVIDER_UNAVAILABLE", status_code=503)
        await super().delete_fact(memory_id=memory_id)


@pytest.mark.asyncio
async def test_general_chat_is_filtered_deduplicated_and_connected_once() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    llm = ExtractingLLM()
    provider = RecordingMemory()

    try:
        async with sessions() as session:
            session.add_all([_farmer(FARMER, "owner"), _farmer(OTHER, "other")])
            farm = Farm(farmer_id=FARMER, name="Home Farm")
            session.add(farm)
            await session.flush()
            plot = Plot(
                farmer_id=FARMER,
                farm_id=farm.id,
                name="Tomato Plot",
                latitude=18,
                longitude=74,
            )
            chat = ChatSession(farmer_id=FARMER, scope_type="general")
            session.add_all([plot, chat])
            await session.flush()
            session.add_all(
                [
                    ChatMessage(
                        farmer_id=FARMER,
                        chat_id=chat.id,
                        sequence=1,
                        role="user",
                        content="I planted tomato in June. What should I do?",
                    ),
                    ChatMessage(
                        farmer_id=FARMER,
                        chat_id=chat.id,
                        sequence=2,
                        role="assistant",
                        content="You could inspect it tomorrow.",
                    ),
                ]
            )
            await session.commit()
            service = MemoryService(
                repository=MemoryRepository(session),
                chats=ChatRepository(session),
                farms=FarmRepository(session),
                provider=provider,
                llm=LLMRouter(llm, Settings(_env_file=None)),
            )

            connected = await service.connect_chat(
                FARMER,
                chat.id,
                ChatMemoryConnectionCreate(
                    target_type=MemoryConnectionTarget.PLOT,
                    plot_id=plot.id,
                ),
            )
            repeated = await service.connect_chat(
                FARMER,
                chat.id,
                ChatMemoryConnectionCreate(
                    target_type=MemoryConnectionTarget.FARM,
                    farm_id=farm.id,
                ),
            )
            with pytest.raises(ApplicationError) as hidden:
                await service.list_plot(OTHER, plot.id)
            with pytest.raises(ApplicationError) as not_retryable:
                await service.retry_fact(FARMER, connected.memories[0].id)
            await service.delete_fact(FARMER, connected.memories[0].id)
    finally:
        await engine.dispose()

    assert len(connected.memories) == 1
    assert connected.memories[0].index_status == "indexed"
    assert llm.requests[0][1] == "global.anthropic.claude-sonnet-4-6"
    assert "assistant: You could inspect it tomorrow." in llm.requests[0][0].transcript
    assert provider.adds[0][0].scope_id == plot.id
    assert provider.adds[0][1] == ("I planted tomato in June",)
    assert provider.deletes[0][0].scope_id == plot.id
    assert provider.deletes[0][1] == "remote-0"
    assert repeated.id == connected.id
    assert repeated.plot_id == plot.id
    assert hidden.value.code == "PLOT_NOT_FOUND"
    assert not_retryable.value.code == "MEMORY_FACT_NOT_RETRYABLE"


@pytest.mark.asyncio
async def test_scoped_chat_message_is_filtered_and_written_automatically() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    llm = ExtractingLLM()
    provider = RecordingMemory()

    try:
        async with sessions() as session:
            session.add(_farmer(FARMER, "scoped"))
            farm = Farm(farmer_id=FARMER, name="Scoped Farm")
            session.add(farm)
            await session.flush()
            plot = Plot(
                farmer_id=FARMER,
                farm_id=farm.id,
                name="Scoped Plot",
                latitude=18,
                longitude=74,
            )
            session.add(plot)
            await session.flush()
            chat = ChatSession(farmer_id=FARMER, scope_type="plot", plot_id=plot.id)
            session.add(chat)
            await session.flush()
            message = ChatMessage(
                farmer_id=FARMER,
                chat_id=chat.id,
                sequence=1,
                role="user",
                content="I planted tomato in June",
            )
            session.add(message)
            await session.commit()
            service = MemoryService(
                repository=MemoryRepository(session),
                chats=ChatRepository(session),
                farms=FarmRepository(session),
                provider=provider,
                llm=LLMRouter(llm, Settings(_env_file=None)),
            )

            service.schedule_scoped_message(
                FARMER,
                chat.id,
                message,
                farm_id=None,
                plot_id=plot.id,
            )
            await session.commit()
            await service.process_due_capture_jobs(limit=1)
            facts = await service.list_plot(FARMER, plot.id)
    finally:
        await engine.dispose()

    assert len(facts) == 1
    assert facts[0].text == "I planted tomato in June"
    assert facts[0].source_message_id == message.id
    assert facts[0].index_status == "indexed"


@pytest.mark.asyncio
async def test_farmer_can_page_through_more_than_one_hundred_memory_facts() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with sessions() as session:
            session.add(_farmer(FARMER, "pagination"))
            farm = Farm(farmer_id=FARMER, name="Memory Farm")
            session.add(farm)
            await session.flush()
            plot = Plot(
                farmer_id=FARMER,
                farm_id=farm.id,
                name="Memory Plot",
                latitude=18,
                longitude=74,
            )
            session.add(plot)
            await session.flush()
            stored_facts = [
                ScopedMemoryFact(
                    farmer_id=FARMER,
                    plot_id=plot.id,
                    text=f"Memory fact {index:03d}",
                    normalized_text=f"memory fact {index:03d}",
                    index_status="indexed",
                )
                for index in range(105)
            ]
            session.add_all(stored_facts)
            await session.commit()
            expected_ids = {fact.id for fact in stored_facts}
            service = MemoryService(
                repository=MemoryRepository(session),
                chats=ChatRepository(session),
                farms=FarmRepository(session),
                provider=RecordingMemory(),
                llm=LLMRouter(ExtractingLLM(), Settings(_env_file=None)),
            )

            first_page = await service.list_plot(FARMER, plot.id, limit=100, offset=0)
            second_page = await service.list_plot(FARMER, plot.id, limit=100, offset=100)
    finally:
        await engine.dispose()

    assert len(first_page) == 100
    assert len(second_page) == 5
    assert {fact.id for fact in first_page}.isdisjoint(fact.id for fact in second_page)
    assert {fact.id for fact in first_page + second_page} == expected_ids


@pytest.mark.asyncio
async def test_failed_automatic_capture_remains_durable_for_retry() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            session.add(_farmer(FARMER, "retry"))
            plot = Plot(farmer_id=FARMER, name="Retry Plot", latitude=18, longitude=74)
            session.add(plot)
            await session.flush()
            chat = ChatSession(farmer_id=FARMER, scope_type="plot", plot_id=plot.id)
            session.add(chat)
            await session.flush()
            message = ChatMessage(
                farmer_id=FARMER,
                chat_id=chat.id,
                sequence=1,
                role="user",
                content="I planted tomato in June",
            )
            session.add(message)
            await session.flush()
            service = MemoryService(
                repository=MemoryRepository(session),
                chats=ChatRepository(session),
                farms=FarmRepository(session),
                provider=RecordingMemory(),
                llm=LLMRouter(FailingExtractionLLM(), Settings(_env_file=None)),
            )
            job = service.schedule_scoped_message(
                FARMER, chat.id, message, farm_id=None, plot_id=plot.id
            )
            await session.commit()
            await service.process_due_capture_jobs(limit=1)
            retained = await session.get(MemoryCaptureJob, job.id)
    finally:
        await engine.dispose()

    assert retained is not None
    assert retained.attempts == 1
    assert retained.last_error_type == "ApplicationError"


@pytest.mark.asyncio
async def test_index_failure_dead_letters_job_without_losing_canonical_fact() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            session.add(_farmer(FARMER, "dead"))
            plot = Plot(farmer_id=FARMER, name="Dead Letter Plot", latitude=18, longitude=74)
            session.add(plot)
            await session.flush()
            chat = ChatSession(farmer_id=FARMER, scope_type="plot", plot_id=plot.id)
            session.add(chat)
            await session.flush()
            message = ChatMessage(
                farmer_id=FARMER,
                chat_id=chat.id,
                sequence=1,
                role="user",
                content="I planted tomato in June",
            )
            session.add(message)
            await session.flush()
            service = MemoryService(
                repository=MemoryRepository(session),
                chats=ChatRepository(session),
                farms=FarmRepository(session),
                provider=FailingMemory(),
                llm=LLMRouter(ExtractingLLM(), Settings(_env_file=None)),
                max_capture_attempts=1,
            )
            job = service.schedule_scoped_message(
                FARMER, chat.id, message, farm_id=None, plot_id=plot.id
            )
            await session.commit()

            await service.process_due_capture_jobs(limit=1)
            retained = await session.get(MemoryCaptureJob, job.id)
            facts = await service.list_plot(FARMER, plot.id)
    finally:
        await engine.dispose()

    assert retained is not None
    assert retained.status == "dead"
    assert retained.attempts == 1
    assert retained.dead_at is not None
    assert len(facts) == 1
    assert facts[0].index_status == "failed"


@pytest.mark.asyncio
async def test_scan_context_cleaner_deletes_remote_and_canonical_facts() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    provider = DeletingMemory()
    try:
        async with sessions() as session:
            session.add(_farmer(FARMER, "scan-delete"))
            plot = Plot(farmer_id=FARMER, name="Scan Plot", latitude=18, longitude=74)
            session.add(plot)
            await session.flush()
            case = DiagnosisCase(
                farmer_id=FARMER,
                plot_id=plot.id,
                title="Scan case",
                status="completed",
            )
            session.add(case)
            await session.flush()
            chat = ChatSession(
                farmer_id=FARMER,
                scope_type="scan",
                plot_id=plot.id,
                diagnosis_case_id=case.id,
            )
            session.add(chat)
            await session.flush()
            fact = ScopedMemoryFact(
                farmer_id=FARMER,
                plot_id=plot.id,
                source_chat_id=chat.id,
                text="Spots appeared after irrigation",
                normalized_text="spots appeared after irrigation",
                provider_memory_id="remote-scan-fact",
                index_status="indexed",
            )
            session.add(fact)
            await session.commit()

            cleaner = MemoryDiagnosisContextCleaner(MemoryRepository(session), provider)
            await cleaner.delete_scan_context(FARMER, case.id)
            await session.commit()
            retained = await session.get(ScopedMemoryFact, fact.id)
    finally:
        await engine.dispose()

    assert provider.deleted_ids == ["remote-scan-fact"]
    assert retained is None


@pytest.mark.asyncio
async def test_scan_context_cleanup_resumes_after_partial_provider_failure() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    provider = FlakyDeletingMemory("remote-2")
    try:
        async with sessions() as session:
            session.add(_farmer(FARMER, "scan-retry"))
            plot = Plot(farmer_id=FARMER, name="Retry Plot", latitude=18, longitude=74)
            session.add(plot)
            await session.flush()
            case = DiagnosisCase(
                farmer_id=FARMER,
                plot_id=plot.id,
                title="Retry case",
                status="completed",
            )
            session.add(case)
            await session.flush()
            chat = ChatSession(
                farmer_id=FARMER,
                scope_type="scan",
                plot_id=plot.id,
                diagnosis_case_id=case.id,
            )
            session.add(chat)
            await session.flush()
            facts = [
                ScopedMemoryFact(
                    farmer_id=FARMER,
                    plot_id=plot.id,
                    source_chat_id=chat.id,
                    text=f"Fact {number}",
                    normalized_text=f"fact {number}",
                    provider_memory_id=f"remote-{number}",
                    index_status="indexed",
                )
                for number in (1, 2)
            ]
            session.add_all(facts)
            await session.commit()
            cleaner = MemoryDiagnosisContextCleaner(MemoryRepository(session), provider)

            with pytest.raises(ApplicationError) as failed:
                await cleaner.delete_scan_context(FARMER, case.id)
            await cleaner.delete_scan_context(FARMER, case.id)
            await session.commit()
            retained = await session.scalars(
                select(ScopedMemoryFact).where(ScopedMemoryFact.source_chat_id == chat.id)
            )
    finally:
        await engine.dispose()

    assert failed.value.code == "MEMORY_PROVIDER_UNAVAILABLE"
    assert provider.deleted_ids == ["remote-1", "remote-2"]
    assert list(retained) == []


def _farmer(farmer_id: UUID, suffix: str) -> FarmerProfile:
    return FarmerProfile(
        id=farmer_id,
        cognito_sub=suffix,
        cognito_username=f"{suffix}@example.com",
        email=f"{suffix}@example.com",
    )
