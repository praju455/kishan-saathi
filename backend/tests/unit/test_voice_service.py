"""Tests for the owner-scoped voice I/O boundary."""

from dataclasses import dataclass
from typing import cast
from uuid import UUID

import pytest

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.integrations.audio.provider import (
    AudioProvider,
    AudioTranscriptionRequest,
    AudioTranscriptionResult,
    SpeechSynthesisRequest,
    SpeechSynthesisResult,
)
from app.modules.chats.repository import ChatRepository
from app.modules.voice.service import IncomingAudio, VoiceService

FARMER = UUID("00000000-0000-0000-0000-000000000001")
CHAT = UUID("00000000-0000-0000-0000-000000000010")
MESSAGE = UUID("00000000-0000-0000-0000-000000000020")


@dataclass(slots=True)
class StoredMessage:
    role: str
    content: str


class FakeChats:
    def __init__(self) -> None:
        self.owned = True
        self.stored_message: StoredMessage | None = StoredMessage(
            role="assistant",
            content="Inspect the lower leaves tomorrow morning.",
        )

    async def get_chat(self, farmer_id: UUID, chat_id: UUID) -> object | None:
        del farmer_id, chat_id
        return object() if self.owned else None

    async def message(
        self, farmer_id: UUID, chat_id: UUID, message_id: UUID
    ) -> StoredMessage | None:
        del farmer_id, chat_id, message_id
        return self.stored_message


class FakeAudioProvider(AudioProvider):
    def __init__(self) -> None:
        self.transcription_requests: list[AudioTranscriptionRequest] = []
        self.speech_requests: list[SpeechSynthesisRequest] = []

    async def transcribe(self, request: AudioTranscriptionRequest) -> AudioTranscriptionResult:
        self.transcription_requests.append(request)
        return AudioTranscriptionResult(
            text="मेरे टमाटर के पत्ते पीले क्यों हैं?",
            model="test-transcription-model",
        )

    async def synthesize(self, request: SpeechSynthesisRequest) -> SpeechSynthesisResult:
        self.speech_requests.append(request)
        return SpeechSynthesisResult(
            content=b"mp3",
            media_type="audio/mpeg",
            model="tts-1",
            voice="coral",
        )

    async def close(self) -> None:
        return None


def service(chats: FakeChats, provider: FakeAudioProvider) -> VoiceService:
    return VoiceService(
        settings=Settings(_env_file=None),
        chats=cast(ChatRepository, chats),
        provider=provider,
    )


@pytest.mark.asyncio
async def test_transcription_preserves_native_script_without_creating_chat_output() -> None:
    chats = FakeChats()
    provider = FakeAudioProvider()

    response = await service(chats, provider).transcribe(
        FARMER,
        CHAT,
        IncomingAudio(
            filename="recording.m4a",
            content_type="audio/mp4",
            content=b"\x00\x00\x00\x18ftypM4A bounded-recording",
        ),
        language_hint=None,
    )

    assert response.transcript == "मेरे टमाटर के पत्ते पीले क्यों हैं?"
    assert provider.transcription_requests[0].language_hint is None


@pytest.mark.asyncio
async def test_transcription_rejects_other_farmer_chat_before_provider_call() -> None:
    chats = FakeChats()
    chats.owned = False
    provider = FakeAudioProvider()

    with pytest.raises(ApplicationError) as raised:
        await service(chats, provider).transcribe(
            FARMER,
            CHAT,
            IncomingAudio("recording.m4a", "audio/mp4", b"\x00\x00\x00\x18ftypM4A audio"),
            language_hint=None,
        )

    assert raised.value.code == "CHAT_NOT_FOUND"
    assert provider.transcription_requests == []


@pytest.mark.asyncio
async def test_transcription_rejects_renamed_non_audio_before_provider_call() -> None:
    chats = FakeChats()
    provider = FakeAudioProvider()

    with pytest.raises(ApplicationError) as raised:
        await service(chats, provider).transcribe(
            FARMER,
            CHAT,
            IncomingAudio("recording.m4a", "audio/mp4", b"not really audio"),
            language_hint=None,
        )

    assert raised.value.code == "VOICE_AUDIO_CONTENT_INVALID"
    assert provider.transcription_requests == []


@pytest.mark.asyncio
async def test_speech_reads_only_owned_assistant_message() -> None:
    chats = FakeChats()
    provider = FakeAudioProvider()

    result = await service(chats, provider).speech(FARMER, CHAT, MESSAGE)

    assert result.content == b"mp3"
    assert provider.speech_requests == [
        SpeechSynthesisRequest(text="Inspect the lower leaves tomorrow morning.")
    ]

    chats.stored_message = StoredMessage(role="user", content="arbitrary text")
    with pytest.raises(ApplicationError) as raised:
        await service(chats, provider).speech(FARMER, CHAT, MESSAGE)
    assert raised.value.code == "CHAT_MESSAGE_NOT_FOUND"


@pytest.mark.asyncio
async def test_long_speech_is_split_without_storing_audio() -> None:
    chats = FakeChats()
    chats.stored_message = StoredMessage(
        role="assistant",
        content=("Inspect the lower leaves carefully. " * 140).strip(),
    )
    provider = FakeAudioProvider()

    result = await service(chats, provider).speech(FARMER, CHAT, MESSAGE)

    assert len(provider.speech_requests) == 2
    assert all(len(request.text) <= 3800 for request in provider.speech_requests)
    assert result.content == b"mp3mp3"
    assert result.media_type == "audio/mpeg"


def test_speech_chunks_preserve_native_script_and_all_text() -> None:
    text = ("पत्तियों को ध्यान से देखें। " * 220).strip()

    chunks = VoiceService._speech_chunks(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 3800 for chunk in chunks)
    assert " ".join(chunks) == text
