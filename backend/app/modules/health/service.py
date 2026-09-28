"""Readiness checks for required backend dependencies."""

import structlog

from app.core.config import Settings
from app.database.session import DatabasePort
from app.modules.health.schemas import (
    DependencyCheck,
    DependencyStatus,
    HealthResponse,
    HealthStatus,
)


class HealthService:
    """Build process health results while isolating dependency failures."""

    def __init__(
        self,
        settings: Settings,
        database: DatabasePort,
        *,
        leaf_inference_available: bool = False,
    ) -> None:
        self._settings = settings
        self._database = database
        self._leaf_inference_available = leaf_inference_available
        self._logger = structlog.get_logger(__name__)

    def liveness(self) -> HealthResponse:
        """Report that the API process can serve requests."""

        return HealthResponse(
            status=HealthStatus.OK,
            service=self._settings.app_name,
            version=self._settings.app_version,
        )

    async def readiness(self) -> HealthResponse:
        """Report whether all dependencies required for normal traffic work."""

        configured = {
            "authentication": self._settings.cognito_configured,
            "llm": self._settings.bedrock_configured,
            "memory": bool(self._settings.mem0_api_key),
            "current_weather": self._settings.openweather_configured,
            "leaf_inference": self._leaf_inference_available,
            "object_storage": (
                self._settings.storage_backend == "local" or self._settings.cloudinary_configured
            ),
        }
        required = {key: value for key, value in configured.items() if key != "leaf_inference"}
        checks = {
            name: DependencyCheck(
                status=DependencyStatus.UP if available else DependencyStatus.DOWN
            )
            for name, available in configured.items()
        }
        try:
            await self._database.ping()
        except Exception as exc:
            self._logger.warning(
                "health.database_unavailable",
                error_type=type(exc).__name__,
            )
            return HealthResponse(
                status=HealthStatus.NOT_READY,
                service=self._settings.app_name,
                version=self._settings.app_version,
                checks={
                    "database": DependencyCheck(status=DependencyStatus.DOWN),
                    **checks,
                },
            )

        checks = {
            "database": DependencyCheck(status=DependencyStatus.UP),
            **checks,
        }
        if self._settings.app_env in {"staging", "production"} and not all(required.values()):
            return HealthResponse(
                status=HealthStatus.NOT_READY,
                service=self._settings.app_name,
                version=self._settings.app_version,
                checks=checks,
            )
        return HealthResponse(
            status=HealthStatus.OK,
            service=self._settings.app_name,
            version=self._settings.app_version,
            checks=checks,
        )
