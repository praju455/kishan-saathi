"""Amazon Cognito account-lifecycle adapter behind the FastAPI boundary."""

from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.integrations.auth.accounts import AuthTokens


class CognitoAccountAuthProvider:
    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client_id = settings.cognito_app_client_id
        self._endpoint = f"https://cognito-idp.{settings.cognito_region}.amazonaws.com/"
        self._http = http_client or httpx.AsyncClient(
            timeout=settings.external_request_timeout_seconds
        )
        self._owns_http_client = http_client is None

    async def sign_up(self, name: str, email: str, password: str) -> bool:
        payload = await self._call(
            "SignUp",
            {
                "ClientId": self._client_id,
                "Username": email,
                "Password": password,
                "UserAttributes": [
                    {"Name": "email", "Value": email},
                    {"Name": "name", "Value": name},
                ],
            },
        )
        return payload.get("UserConfirmed") is True

    async def confirm_sign_up(self, email: str, code: str) -> None:
        await self._call(
            "ConfirmSignUp",
            {
                "ClientId": self._client_id,
                "Username": email,
                "ConfirmationCode": code,
            },
        )

    async def resend_confirmation(self, email: str) -> None:
        await self._call(
            "ResendConfirmationCode",
            {"ClientId": self._client_id, "Username": email},
        )

    async def sign_in(self, email: str, password: str) -> AuthTokens:
        payload = await self._call(
            "InitiateAuth",
            {
                "AuthFlow": "USER_PASSWORD_AUTH",
                "ClientId": self._client_id,
                "AuthParameters": {"USERNAME": email, "PASSWORD": password},
            },
        )
        return self._tokens(payload.get("AuthenticationResult"), refresh_token=None)

    async def refresh(self, refresh_token: str) -> AuthTokens:
        try:
            payload = await self._call(
                "InitiateAuth",
                {
                    "AuthFlow": "REFRESH_TOKEN_AUTH",
                    "ClientId": self._client_id,
                    "AuthParameters": {"REFRESH_TOKEN": refresh_token},
                },
            )
        except ApplicationError as exc:
            if exc.code == "AUTH_INCORRECT_CREDENTIALS":
                raise ApplicationError(code="AUTH_SESSION_EXPIRED", status_code=401) from exc
            raise
        return self._tokens(payload.get("AuthenticationResult"), refresh_token=refresh_token)

    async def request_password_reset(self, email: str) -> None:
        await self._call(
            "ForgotPassword",
            {"ClientId": self._client_id, "Username": email},
            ignored_provider_codes=frozenset({"UserNotFoundException"}),
        )

    async def confirm_password_reset(self, email: str, code: str, new_password: str) -> None:
        await self._call(
            "ConfirmForgotPassword",
            {
                "ClientId": self._client_id,
                "Username": email,
                "ConfirmationCode": code,
                "Password": new_password,
            },
        )

    async def _call(
        self,
        operation: str,
        body: dict[str, Any],
        *,
        ignored_provider_codes: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        try:
            response = await self._http.post(
                self._endpoint,
                headers={
                    "Content-Type": "application/x-amz-json-1.1",
                    "X-Amz-Target": f"AWSCognitoIdentityProviderService.{operation}",
                },
                json=body,
            )
        except httpx.TimeoutException as exc:
            raise ApplicationError(code="AUTH_PROVIDER_TIMEOUT", status_code=503) from exc
        except httpx.RequestError as exc:
            raise ApplicationError(code="AUTH_PROVIDER_UNREACHABLE", status_code=503) from exc

        try:
            payload = response.json() if response.content else {}
        except ValueError as exc:
            raise ApplicationError(code="AUTH_INVALID_RESPONSE", status_code=502) from exc
        if not isinstance(payload, dict):
            raise ApplicationError(code="AUTH_INVALID_RESPONSE", status_code=502)
        if response.is_success:
            return payload

        provider_code = str(payload.get("__type", "AUTH_FAILED")).split("#")[-1]
        if provider_code in ignored_provider_codes:
            return {}
        raise ApplicationError(
            code=self._error_code(operation, provider_code),
            status_code=429
            if provider_code in {"LimitExceededException", "TooManyRequestsException"}
            else 400,
        )

    @staticmethod
    def _error_code(operation: str, provider_code: str) -> str:
        if provider_code in {"LimitExceededException", "TooManyRequestsException"}:
            return "AUTH_RATE_LIMITED"
        if provider_code == "UserNotConfirmedException":
            return "AUTH_EMAIL_UNCONFIRMED"
        if provider_code == "UsernameExistsException":
            return "AUTH_ACCOUNT_EXISTS"
        if provider_code == "CodeMismatchException":
            return "AUTH_CONFIRMATION_CODE_INVALID"
        if provider_code == "ExpiredCodeException":
            return "AUTH_CONFIRMATION_CODE_EXPIRED"
        if provider_code == "InvalidPasswordException":
            return "AUTH_PASSWORD_INVALID"
        if provider_code == "ResourceNotFoundException":
            return "AUTH_CONFIGURATION_INVALID"
        if provider_code in {"NotAuthorizedException", "UserNotFoundException"}:
            return (
                "AUTH_INCORRECT_CREDENTIALS"
                if operation == "InitiateAuth"
                else "AUTH_PROVIDER_REJECTED"
            )
        return "AUTH_PROVIDER_REJECTED"

    @staticmethod
    def _tokens(value: object, *, refresh_token: str | None) -> AuthTokens:
        if not isinstance(value, dict):
            raise ApplicationError(code="AUTH_INVALID_RESPONSE", status_code=502)
        access = value.get("AccessToken")
        resolved_refresh = value.get("RefreshToken") or refresh_token
        expires_in = value.get("ExpiresIn", 3600)
        id_token = value.get("IdToken")
        if (
            not isinstance(access, str)
            or not isinstance(resolved_refresh, str)
            or not isinstance(expires_in, int)
            or (id_token is not None and not isinstance(id_token, str))
        ):
            raise ApplicationError(code="AUTH_INVALID_RESPONSE", status_code=502)
        return AuthTokens(
            access_token=access,
            refresh_token=resolved_refresh,
            id_token=id_token,
            expires_in=expires_in,
        )

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()
