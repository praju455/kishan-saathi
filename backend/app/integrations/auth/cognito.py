"""Amazon Cognito authentication provider adapter."""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import jwt
from jwt import PyJWK
from jwt.exceptions import InvalidTokenError

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.core.security import AuthenticatedPrincipal, ExternalIdentity


class CognitoAuthProvider:
    """Validate Cognito access tokens and resolve canonical user attributes."""

    def __init__(
        self,
        settings: Settings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._region = settings.cognito_region
        self._pool_id = settings.cognito_user_pool_id
        self._client_id = settings.cognito_app_client_id
        self._issuer = f"https://cognito-idp.{self._region}.amazonaws.com/{self._pool_id}"
        self._jwks_url = f"{self._issuer}/.well-known/jwks.json"
        self._identity_url = f"https://cognito-idp.{self._region}.amazonaws.com/"
        self._jwks_ttl = settings.cognito_jwks_cache_seconds
        self._jwks_min_refresh = settings.cognito_jwks_min_refresh_seconds
        self._jwks_max_bytes = settings.cognito_jwks_max_bytes
        self._http = http_client or httpx.AsyncClient(
            timeout=settings.external_request_timeout_seconds
        )
        self._owns_http_client = http_client is None
        self._jwks_by_kid: dict[str, dict[str, Any]] = {}
        self._jwks_expires_at = 0.0
        self._jwks_last_refresh_at = 0.0
        self._unknown_kids: dict[str, float] = {}
        self._jwks_lock = asyncio.Lock()

    async def verify_access_token(self, access_token: str) -> AuthenticatedPrincipal:
        """Verify signature, issuer, expiry, token use, and app-client binding."""

        try:
            header = jwt.get_unverified_header(access_token)
            kid = header.get("kid")
            algorithm = header.get("alg")
            if not isinstance(kid, str) or algorithm != "RS256":
                raise InvalidTokenError

            jwk_data = await self._jwk_for_kid(kid)
            signing_key = PyJWK.from_dict(jwk_data, algorithm="RS256").key
            claims = jwt.decode(
                access_token,
                signing_key,
                algorithms=["RS256"],
                issuer=self._issuer,
                options={
                    "verify_aud": False,
                    "require": ["sub", "iss", "exp", "token_use", "client_id"],
                },
            )
        except ApplicationError:
            raise
        except (InvalidTokenError, TypeError, ValueError, KeyError) as exc:
            raise ApplicationError(code="AUTH_TOKEN_INVALID", status_code=401) from exc

        if claims.get("token_use") != "access":
            raise ApplicationError(code="AUTH_TOKEN_INVALID", status_code=401)
        if claims.get("client_id") != self._client_id:
            raise ApplicationError(code="AUTH_TOKEN_INVALID", status_code=401)

        subject = claims.get("sub")
        username = claims.get("username") or claims.get("cognito:username")
        expires_at = claims.get("exp")
        if not isinstance(subject, str) or not isinstance(username, str):
            raise ApplicationError(code="AUTH_TOKEN_INVALID", status_code=401)
        if not isinstance(expires_at, int):
            raise ApplicationError(code="AUTH_TOKEN_INVALID", status_code=401)

        return AuthenticatedPrincipal(
            subject=subject,
            username=username,
            expires_at=datetime.fromtimestamp(expires_at, tz=UTC),
        )

    async def get_identity(self, access_token: str) -> ExternalIdentity:
        """Retrieve readable Cognito attributes for initial local provisioning."""

        try:
            response = await self._http.post(
                self._identity_url,
                headers={
                    "Content-Type": "application/x-amz-json-1.1",
                    "X-Amz-Target": "AWSCognitoIdentityProviderService.GetUser",
                },
                json={"AccessToken": access_token},
            )
        except httpx.HTTPError as exc:
            raise ApplicationError(code="AUTH_PROVIDER_UNAVAILABLE", status_code=503) from exc

        if response.status_code in {400, 401}:
            raise ApplicationError(code="AUTH_TOKEN_INVALID", status_code=401)
        if response.status_code >= 500:
            raise ApplicationError(code="AUTH_PROVIDER_UNAVAILABLE", status_code=503)
        if response.status_code != 200:
            raise ApplicationError(code="AUTH_IDENTITY_UNAVAILABLE", status_code=502)

        try:
            payload = response.json()
            username = payload["Username"]
            raw_attributes = payload["UserAttributes"]
            attributes = {
                item["Name"]: item["Value"]
                for item in raw_attributes
                if isinstance(item, dict) and "Name" in item and "Value" in item
            }
            subject = attributes["sub"]
            email = attributes["email"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ApplicationError(code="AUTH_IDENTITY_INVALID", status_code=502) from exc

        if not all(isinstance(value, str) and value for value in (username, subject, email)):
            raise ApplicationError(code="AUTH_IDENTITY_INVALID", status_code=502)

        raw_name = attributes.get("name")
        if raw_name is not None and not isinstance(raw_name, str):
            raise ApplicationError(code="AUTH_IDENTITY_INVALID", status_code=502)
        name = " ".join(raw_name.split()) if raw_name else None
        if name is not None and len(name) > 100:
            raise ApplicationError(code="AUTH_IDENTITY_INVALID", status_code=502)

        return ExternalIdentity(
            subject=subject,
            username=username,
            email=email,
            email_verified=attributes.get("email_verified", "false").lower() == "true",
            name=name or None,
        )

    async def close(self) -> None:
        """Close an internally owned HTTP connection pool."""

        if self._owns_http_client:
            await self._http.aclose()

    async def _jwk_for_kid(self, kid: str) -> dict[str, Any]:
        now = time.monotonic()
        unknown_until = self._unknown_kids.get(kid, 0)
        if unknown_until > now:
            raise ApplicationError(code="AUTH_TOKEN_INVALID", status_code=401)
        keys = await self._get_jwks(force_refresh=False)
        key = keys.get(kid)
        if key is None and now - self._jwks_last_refresh_at >= self._jwks_min_refresh:
            keys = await self._get_jwks(force_refresh=True)
            key = keys.get(kid)
        if key is None:
            self._unknown_kids[kid] = time.monotonic() + self._jwks_min_refresh
            raise ApplicationError(code="AUTH_TOKEN_INVALID", status_code=401)
        return key

    async def _get_jwks(self, *, force_refresh: bool) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        if not force_refresh and self._jwks_by_kid and now < self._jwks_expires_at:
            return self._jwks_by_kid

        async with self._jwks_lock:
            now = time.monotonic()
            if not force_refresh and self._jwks_by_kid and now < self._jwks_expires_at:
                return self._jwks_by_kid
            try:
                response = await self._http.get(self._jwks_url)
                response.raise_for_status()
                if len(response.content) > self._jwks_max_bytes:
                    raise ValueError("JWKS_RESPONSE_TOO_LARGE")
                payload = response.json()
                keys = payload["keys"]
                parsed = {
                    key["kid"]: key
                    for key in keys
                    if isinstance(key, dict) and isinstance(key.get("kid"), str)
                }
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                raise ApplicationError(
                    code="AUTH_PROVIDER_UNAVAILABLE",
                    status_code=503,
                ) from exc
            if not parsed:
                raise ApplicationError(code="AUTH_PROVIDER_UNAVAILABLE", status_code=503)
            self._jwks_by_kid = parsed
            refreshed_at = time.monotonic()
            self._jwks_last_refresh_at = refreshed_at
            self._jwks_expires_at = refreshed_at + self._jwks_ttl
            self._unknown_kids.clear()
            return self._jwks_by_kid
