"""Production startup must fail before serving a partially configured app."""

import pytest

from app.core.config import Settings


def _production(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "_env_file": None,
        "app_env": "production",
        "database_url": (
            "postgresql+asyncpg://service:password@database.example/krishisathi?ssl=require"
        ),
        "storage_backend": "cloudinary",
        "cloudinary_cloud_name": "private-images",
        "cloudinary_api_key": "cloudinary-key",
        "cloudinary_api_secret": "cloudinary-secret",
        "cognito_user_pool_id": "ap-northeast-1_pool",
        "cognito_app_client_id": "client-id",
        "auth_rate_limit_hmac_key": "a-production-secret-with-at-least-32-characters",
        "bedrock_api_key": "configured-in-secret-manager",
        "openweather_api_key": "configured-in-secret-manager",
        "mem0_api_key": "configured-in-secret-manager",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_fully_configured_production_settings_are_accepted() -> None:
    assert _production().app_env == "production"


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"cognito_user_pool_id": ""}, "PRODUCTION_REQUIRES_COGNITO"),
        (
            {"auth_rate_limit_hmac_key": ""},
            "PRODUCTION_REQUIRES_AUTH_RATE_LIMIT_HMAC_KEY",
        ),
        ({"bedrock_api_key": ""}, "PRODUCTION_REQUIRES_BEDROCK"),
        (
            {"cloudinary_api_secret": ""},
            "CLOUDINARY_CONFIGURATION_REQUIRED",
        ),
        ({"openweather_api_key": ""}, "PRODUCTION_REQUIRES_OPENWEATHER"),
        ({"mem0_api_key": ""}, "PRODUCTION_REQUIRES_MEM0"),
    ],
)
def test_core_providers_are_required_in_production(
    override: dict[str, object], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        _production(**override)
