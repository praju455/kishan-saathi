"""Unit tests for validated application configuration."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_settings_reject_unsupported_environment() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="unknown")


def test_settings_reject_non_positive_database_timeout() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_connect_timeout_seconds=0)


def test_chat_turn_timeout_must_be_shorter_than_lease() -> None:
    with pytest.raises(ValidationError, match="CHAT_TURN_TIMEOUT_MUST_BE_SHORTER_THAN_LEASE"):
        Settings(
            _env_file=None,
            chat_turn_processing_timeout_seconds=180,
            chat_turn_lease_seconds=180,
        )


def test_supabase_url_is_normalized_for_asyncpg() -> None:
    settings = Settings(
        _env_file=None,
        database_url=(
            "postgresql://postgres.project:password@aws-0-region.pooler.supabase.com:6543/"
            "postgres?sslmode=require"
        ),
    )

    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.database_url.endswith("?ssl=require")


def test_cognito_region_is_derived_from_pool_id() -> None:
    settings = Settings(
        _env_file=None,
        cognito_user_pool_id="ap-northeast-1_example",
        cognito_app_client_id="client",
    )

    assert settings.cognito_region == "ap-northeast-1"
    assert settings.cognito_configured is True


def test_cors_wildcard_is_rejected() -> None:
    with pytest.raises(ValidationError, match="CORS_WILDCARD_FORBIDDEN"):
        Settings(_env_file=None, cors_allowed_origins=["*"])
