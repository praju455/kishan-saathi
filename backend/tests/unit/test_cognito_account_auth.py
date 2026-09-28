"""Cognito account lifecycle adapter tests."""

import json

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.integrations.auth.cognito_accounts import CognitoAccountAuthProvider


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        cognito_region="ap-south-1",
        cognito_user_pool_id="ap-south-1_test",
        cognito_app_client_id="client-id",
    )


@pytest.mark.asyncio
async def test_sign_up_sends_the_backend_validated_farmer_name() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert {"Name": "name", "Value": "Nikhil Kumar"} in payload["UserAttributes"]
        return httpx.Response(200, json={"UserConfirmed": False})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CognitoAccountAuthProvider(_settings(), http_client=client)
        confirmed = await provider.sign_up("Nikhil Kumar", "farmer@example.com", "valid-password")

    assert confirmed is False


@pytest.mark.asyncio
async def test_sign_in_maps_tokens_without_leaking_provider_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-amz-target"].endswith(".InitiateAuth")
        return httpx.Response(
            200,
            json={
                "AuthenticationResult": {
                    "AccessToken": "access",
                    "RefreshToken": "refresh",
                    "IdToken": "id",
                    "ExpiresIn": 3600,
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CognitoAccountAuthProvider(_settings(), http_client=client)
        tokens = await provider.sign_in("farmer@example.com", "valid-password")

    assert tokens.access_token == "access"
    assert tokens.refresh_token == "refresh"
    assert tokens.expires_in == 3600


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_code", "expected"),
    [
        ("NotAuthorizedException", "AUTH_INCORRECT_CREDENTIALS"),
        ("UserNotConfirmedException", "AUTH_EMAIL_UNCONFIRMED"),
        ("CodeMismatchException", "AUTH_CONFIRMATION_CODE_INVALID"),
        ("ExpiredCodeException", "AUTH_CONFIRMATION_CODE_EXPIRED"),
        ("LimitExceededException", "AUTH_RATE_LIMITED"),
    ],
)
async def test_provider_failures_are_normalized(provider_code: str, expected: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"__type": provider_code, "message": "secret"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CognitoAccountAuthProvider(_settings(), http_client=client)
        with pytest.raises(ApplicationError) as raised:
            await provider.sign_in("farmer@example.com", "valid-password")

    assert raised.value.code == expected


@pytest.mark.asyncio
async def test_protocol_error_is_provider_unreachable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("connection closed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CognitoAccountAuthProvider(_settings(), http_client=client)
        with pytest.raises(ApplicationError) as raised:
            await provider.sign_up("Nikhil Kumar", "farmer@example.com", "valid-password")

    assert raised.value.code == "AUTH_PROVIDER_UNREACHABLE"


@pytest.mark.asyncio
async def test_rejected_refresh_is_reported_as_expired_session() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"__type": "NotAuthorizedException", "message": "secret"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CognitoAccountAuthProvider(_settings(), http_client=client)
        with pytest.raises(ApplicationError) as raised:
            await provider.refresh("expired-refresh-token")

    assert raised.value.code == "AUTH_SESSION_EXPIRED"
    assert raised.value.status_code == 401


@pytest.mark.asyncio
async def test_password_reset_does_not_reveal_an_unknown_account() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"__type": "UserNotFoundException", "message": "secret"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = CognitoAccountAuthProvider(_settings(), http_client=client)
        await provider.request_password_reset("unknown@example.com")
