"""Startup dependency container and provider/plugin bindings."""

from app.core.config import Settings
from app.integrations.audio.provider import AudioProvider, UnavailableAudioProvider
from app.integrations.auth.accounts import AccountAuthProvider, UnavailableAccountAuthProvider
from app.integrations.auth.cognito import CognitoAuthProvider
from app.integrations.auth.cognito_accounts import CognitoAccountAuthProvider
from app.integrations.auth.provider import AuthProvider, UnavailableAuthProvider
from app.integrations.geocoding.open_meteo import OpenMeteoGeocodingProvider
from app.integrations.geocoding.provider import GeocodingProvider
from app.integrations.inference.provider import (
    LeafInferenceProvider,
    UnavailableLeafInferenceProvider,
)
from app.integrations.llm.bedrock_converse import BedrockConverseProvider
from app.integrations.llm.provider import LLMProvider, UnavailableLLMProvider
from app.integrations.memory.mem0 import Mem0MemoryProvider
from app.integrations.memory.provider import MemoryProvider, UnavailableMemoryProvider
from app.integrations.progression.bedrock_converse import BedrockProgressionProvider
from app.integrations.progression.provider import (
    ProgressionProvider,
    UnavailableProgressionProvider,
)
from app.integrations.storage.cloudinary import CloudinaryObjectStorage
from app.integrations.storage.local import LocalObjectStorage
from app.integrations.storage.provider import ObjectStorageProvider
from app.integrations.usage.provider import AIUsageSink
from app.integrations.weather.open_meteo import OpenMeteoForecastProvider
from app.integrations.weather.openweather import (
    OpenWeatherCurrentProvider,
    UnavailableCurrentWeatherProvider,
)
from app.integrations.weather.provider import CurrentWeatherProvider, ForecastWeatherProvider


def build_auth_provider(settings: Settings) -> AuthProvider:
    """Select the configured authentication adapter once during startup."""

    if not settings.cognito_configured:
        return UnavailableAuthProvider()
    return CognitoAuthProvider(settings)


def build_account_auth_provider(settings: Settings) -> AccountAuthProvider:
    if not settings.cognito_configured:
        return UnavailableAccountAuthProvider()
    return CognitoAccountAuthProvider(settings)


def build_object_storage(settings: Settings) -> ObjectStorageProvider:
    """Select local development storage or private production Cloudinary."""

    if settings.storage_backend == "cloudinary":
        return CloudinaryObjectStorage(settings)
    return LocalObjectStorage(settings.local_storage_path)


def build_leaf_inference_provider(settings: Settings) -> LeafInferenceProvider:
    """Fail honestly until an evaluated classifier checkpoint is supplied."""

    del settings
    return UnavailableLeafInferenceProvider()


def build_llm_provider(settings: Settings, usage_sink: AIUsageSink | None = None) -> LLMProvider:
    if not settings.bedrock_configured:
        return UnavailableLLMProvider()
    return BedrockConverseProvider(settings, usage_sink=usage_sink)


def build_memory_provider(settings: Settings) -> MemoryProvider:
    if not settings.mem0_api_key:
        return UnavailableMemoryProvider()
    return Mem0MemoryProvider(settings.mem0_api_key)


def build_progression_provider(settings: Settings) -> ProgressionProvider:
    """Bind visual comparison only when Bedrock is configured."""

    if not settings.bedrock_configured:
        return UnavailableProgressionProvider()
    return BedrockProgressionProvider(settings)


def build_audio_provider(settings: Settings) -> AudioProvider:
    """Voice remains unavailable until an approved speech provider is configured."""

    del settings
    return UnavailableAudioProvider()


def build_current_weather_provider(settings: Settings) -> CurrentWeatherProvider:
    if not settings.openweather_api_key:
        return UnavailableCurrentWeatherProvider()
    return OpenWeatherCurrentProvider(settings)


def build_forecast_weather_provider(settings: Settings) -> ForecastWeatherProvider:
    return OpenMeteoForecastProvider(settings)


def build_geocoding_provider(settings: Settings) -> GeocodingProvider:
    return OpenMeteoGeocodingProvider(settings)
