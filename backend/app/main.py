"""FastAPI application factory and process entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router, versioned_api_router
from app.core.config import Settings, get_settings
from app.core.container import (
    build_account_auth_provider,
    build_audio_provider,
    build_auth_provider,
    build_current_weather_provider,
    build_forecast_weather_provider,
    build_geocoding_provider,
    build_leaf_inference_provider,
    build_llm_provider,
    build_memory_provider,
    build_object_storage,
    build_progression_provider,
)
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging, request_context_middleware
from app.core.rate_limits import AuthRateLimiter, PaidOperationRateLimiter
from app.database.session import Database, DatabasePort
from app.integrations.audio.provider import AudioProvider
from app.integrations.auth.accounts import AccountAuthProvider
from app.integrations.auth.provider import AuthProvider
from app.integrations.geocoding.provider import GeocodingProvider
from app.integrations.inference.provider import LeafInferenceProvider
from app.integrations.llm.provider import LLMProvider
from app.integrations.memory.provider import MemoryProvider
from app.integrations.progression.provider import ProgressionProvider
from app.integrations.storage.provider import ObjectStorageProvider
from app.integrations.weather.provider import CurrentWeatherProvider, ForecastWeatherProvider
from app.modules.chats.worker import ChatTurnWorker
from app.modules.memories.worker import MemoryCaptureWorker
from app.modules.reports.worker import ReportRetentionWorker
from app.modules.storage_cleanup.worker import ObjectCleanupWorker
from app.modules.usage.sink import DatabaseAIUsageSink


def create_app(
    settings: Settings | None = None,
    database: DatabasePort | None = None,
    auth_provider: AuthProvider | None = None,
    object_storage: ObjectStorageProvider | None = None,
    leaf_inference_provider: LeafInferenceProvider | None = None,
    llm_provider: LLMProvider | None = None,
    memory_provider: MemoryProvider | None = None,
    current_weather_provider: CurrentWeatherProvider | None = None,
    forecast_weather_provider: ForecastWeatherProvider | None = None,
    geocoding_provider: GeocodingProvider | None = None,
    progression_provider: ProgressionProvider | None = None,
    audio_provider: AudioProvider | None = None,
    account_auth_provider: AccountAuthProvider | None = None,
) -> FastAPI:
    """Build an application with explicit, replaceable process dependencies."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    resolved_database = database or Database(resolved_settings)
    resolved_auth_provider = auth_provider or build_auth_provider(resolved_settings)
    resolved_account_auth_provider = account_auth_provider or build_account_auth_provider(
        resolved_settings
    )
    resolved_object_storage = object_storage or build_object_storage(resolved_settings)
    resolved_leaf_inference = leaf_inference_provider or build_leaf_inference_provider(
        resolved_settings
    )
    usage_sink = DatabaseAIUsageSink(resolved_database)
    resolved_llm_provider = llm_provider or build_llm_provider(
        resolved_settings, usage_sink=usage_sink
    )
    resolved_memory_provider = memory_provider or build_memory_provider(resolved_settings)
    resolved_progression_provider = progression_provider or build_progression_provider(
        resolved_settings
    )
    resolved_audio_provider = audio_provider or build_audio_provider(resolved_settings)
    paid_operation_rate_limiter = PaidOperationRateLimiter(resolved_settings, resolved_database)
    auth_rate_limiter = AuthRateLimiter(resolved_settings, resolved_database)
    resolved_current_weather = current_weather_provider or build_current_weather_provider(
        resolved_settings
    )
    resolved_forecast_weather = forecast_weather_provider or build_forecast_weather_provider(
        resolved_settings
    )
    resolved_geocoding = geocoding_provider or build_geocoding_provider(resolved_settings)
    cleanup_worker = ObjectCleanupWorker(
        settings=resolved_settings,
        database=resolved_database,
        storage=resolved_object_storage,
    )
    report_retention_worker = ReportRetentionWorker(
        settings=resolved_settings,
        database=resolved_database,
        storage=resolved_object_storage,
    )
    memory_capture_worker = MemoryCaptureWorker(
        settings=resolved_settings,
        database=resolved_database,
        llm_provider=resolved_llm_provider,
        memory_provider=resolved_memory_provider,
    )
    chat_turn_worker = ChatTurnWorker(
        settings=resolved_settings,
        database=resolved_database,
        llm_provider=resolved_llm_provider,
        memory_provider=resolved_memory_provider,
        current_weather_provider=resolved_current_weather,
        forecast_weather_provider=resolved_forecast_weather,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger = structlog.get_logger("lifecycle")
        logger.info("application.started", environment=resolved_settings.app_env)
        await cleanup_worker.start()
        await report_retention_worker.start()
        await memory_capture_worker.start()
        await chat_turn_worker.start()
        try:
            yield
        finally:
            await chat_turn_worker.stop()
            await memory_capture_worker.stop()
            await report_retention_worker.stop()
            await cleanup_worker.stop()
            await application.state.geocoding_provider.close()
            await application.state.forecast_weather_provider.close()
            await application.state.current_weather_provider.close()
            await application.state.memory_provider.close()
            await application.state.audio_provider.close()
            await application.state.progression_provider.close()
            await application.state.llm_provider.close()
            await application.state.leaf_inference_provider.close()
            await application.state.object_storage.close()
            await application.state.auth_provider.close()
            await application.state.account_auth_provider.close()
            await application.state.database.dispose()
            logger.info("application.stopped")

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        debug=resolved_settings.debug,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.database = resolved_database
    application.state.auth_provider = resolved_auth_provider
    application.state.account_auth_provider = resolved_account_auth_provider
    application.state.object_storage = resolved_object_storage
    application.state.leaf_inference_provider = resolved_leaf_inference
    application.state.llm_provider = resolved_llm_provider
    application.state.memory_provider = resolved_memory_provider
    application.state.progression_provider = resolved_progression_provider
    application.state.audio_provider = resolved_audio_provider
    application.state.paid_operation_rate_limiter = paid_operation_rate_limiter
    application.state.auth_rate_limiter = auth_rate_limiter
    application.state.current_weather_provider = resolved_current_weather
    application.state.forecast_weather_provider = resolved_forecast_weather
    application.state.geocoding_provider = resolved_geocoding
    if resolved_settings.cors_allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_allowed_origins,
            allow_credentials=False,
            allow_methods=["DELETE", "GET", "OPTIONS", "PATCH", "POST", "PUT"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
            expose_headers=["Retry-After", "X-Request-ID"],
        )
    application.middleware("http")(request_context_middleware)
    register_error_handlers(application)
    application.include_router(api_router)
    application.include_router(
        versioned_api_router,
        prefix=resolved_settings.api_v1_prefix,
    )
    return application


app = create_app()
