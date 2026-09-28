"""Private Cloudinary object storage with owner-bound opaque public IDs."""

import asyncio
from io import BytesIO
from time import time
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

import cloudinary.uploader  # type: ignore[import-untyped]
import cloudinary.utils  # type: ignore[import-untyped]
import httpx
from cloudinary.exceptions import Error as CloudinaryError  # type: ignore[import-untyped]
from cloudinary.exceptions import NotFound as CloudinaryNotFound

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.integrations.storage.provider import ObjectStorageProvider, StoredObject


class CloudinaryClient(Protocol):
    def upload(self, content: bytes, **options: Any) -> dict[str, Any]: ...

    def destroy(self, public_id: str, **options: Any) -> dict[str, Any]: ...

    def private_download_url(self, public_id: str, format: str, **options: Any) -> str: ...


class CloudinarySDKClient:
    """Thin injectable wrapper around the official synchronous Python SDK."""

    def upload(self, content: bytes, **options: Any) -> dict[str, Any]:
        return cast(dict[str, Any], cloudinary.uploader.upload(BytesIO(content), **options))

    def destroy(self, public_id: str, **options: Any) -> dict[str, Any]:
        return cast(dict[str, Any], cloudinary.uploader.destroy(public_id, **options))

    def private_download_url(self, public_id: str, format: str, **options: Any) -> str:
        return cast(str, cloudinary.utils.private_download_url(public_id, format, **options))


class CloudinaryObjectStorage(ObjectStorageProvider):
    """Store authenticated images and stream bytes only through owner-authorized APIs."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: CloudinaryClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or CloudinarySDKClient()
        self._http = http_client or httpx.AsyncClient(
            timeout=settings.external_request_timeout_seconds
        )
        self._owns_http_client = http_client is None
        self._folder = settings.cloudinary_folder
        self._max_object_bytes = settings.max_image_bytes
        self._configuration: dict[str, Any] = {
            "cloud_name": settings.cloudinary_cloud_name,
            "api_key": settings.cloudinary_api_key.get_secret_value(),
            "api_secret": settings.cloudinary_api_secret.get_secret_value(),
            "secure": True,
        }

    async def put_private_image(
        self,
        *,
        owner_id: UUID,
        category: str,
        content: bytes,
    ) -> StoredObject:
        safe_category = self._safe_segment(category)
        key = f"{self._folder}/{owner_id}/{safe_category}/{uuid4()}"
        options = {
            **self._configuration,
            "public_id": key,
            "resource_type": "image",
            "type": "authenticated",
            "format": "jpg",
            "overwrite": False,
            "unique_filename": False,
            "invalidate": True,
            "context": {"owner_id": str(owner_id), "category": safe_category},
        }
        try:
            response = await asyncio.to_thread(self._client.upload, content, **options)
        except CloudinaryError as exc:
            raise ApplicationError(code="STORAGE_WRITE_FAILED", status_code=503) from exc
        if (
            response.get("public_id") != key
            or response.get("resource_type") != "image"
            or response.get("type") != "authenticated"
        ):
            raise ApplicationError(code="STORAGE_WRITE_FAILED", status_code=502)
        return StoredObject(key=key, size_bytes=len(content), media_type="image/jpeg")

    async def read_private(self, *, owner_id: UUID, key: str) -> bytes:
        self._validate_owned_key(owner_id, key)
        try:
            url = self._client.private_download_url(
                key,
                "jpg",
                **self._configuration,
                resource_type="image",
                type="authenticated",
                expires_at=int(time()) + 60,
            )
            response = await self._http.get(url)
        except CloudinaryNotFound as exc:
            raise ApplicationError(code="STORAGE_OBJECT_NOT_FOUND", status_code=404) from exc
        except (CloudinaryError, httpx.TimeoutException, httpx.NetworkError) as exc:
            raise ApplicationError(code="STORAGE_READ_FAILED", status_code=503) from exc
        if response.status_code == 404:
            raise ApplicationError(code="STORAGE_OBJECT_NOT_FOUND", status_code=404)
        if response.status_code != 200:
            raise ApplicationError(code="STORAGE_READ_FAILED", status_code=503)
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_object_bytes:
                    raise ApplicationError(code="STORAGE_OBJECT_TOO_LARGE", status_code=422)
            except ValueError as exc:
                raise ApplicationError(code="STORAGE_READ_FAILED", status_code=502) from exc
        content = response.content
        if len(content) > self._max_object_bytes:
            raise ApplicationError(code="STORAGE_OBJECT_TOO_LARGE", status_code=422)
        return content

    async def delete_private(self, *, owner_id: UUID, key: str) -> None:
        self._validate_owned_key(owner_id, key)
        try:
            response = await asyncio.to_thread(
                self._client.destroy,
                key,
                **self._configuration,
                resource_type="image",
                type="authenticated",
                invalidate=True,
            )
        except CloudinaryNotFound:
            return
        except CloudinaryError as exc:
            raise ApplicationError(code="STORAGE_DELETE_FAILED", status_code=503) from exc
        if response.get("result") not in {"ok", "not found"}:
            raise ApplicationError(code="STORAGE_DELETE_FAILED", status_code=503)

    async def close(self) -> None:
        if self._owns_http_client:
            await self._http.aclose()

    def _validate_owned_key(self, owner_id: UUID, key: str) -> None:
        expected_prefix = f"{self._folder}/{owner_id}/"
        if not key.startswith(expected_prefix) or ".." in key.split("/"):
            raise ApplicationError(code="STORAGE_OBJECT_NOT_FOUND", status_code=404)

    @staticmethod
    def _safe_segment(value: str) -> str:
        if not value or not value.replace("-", "").replace("_", "").isalnum():
            raise ApplicationError(code="STORAGE_CATEGORY_INVALID", status_code=422)
        return value
