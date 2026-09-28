"""Tests for authenticated, owner-bound Cloudinary storage."""

from typing import Any, cast
from uuid import UUID

import httpx
import pytest

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.integrations.storage.cloudinary import CloudinaryClient, CloudinaryObjectStorage

OWNER = UUID("00000000-0000-0000-0000-000000000011")
OTHER = UUID("00000000-0000-0000-0000-000000000012")


class FakeCloudinaryClient:
    def __init__(self) -> None:
        self.upload_options: dict[str, Any] = {}
        self.destroy_options: dict[str, Any] = {}
        self.destroyed: str | None = None

    def upload(self, content: bytes, **options: Any) -> dict[str, Any]:
        self.upload_options = options
        return {
            "public_id": options["public_id"],
            "resource_type": "image",
            "type": "authenticated",
            "bytes": len(content),
        }

    def destroy(self, public_id: str, **options: Any) -> dict[str, Any]:
        self.destroyed = public_id
        self.destroy_options = options
        return {"result": "ok"}

    def private_download_url(self, public_id: str, format: str, **options: Any) -> str:
        assert public_id.startswith(f"kishan-saathi-private/{OWNER}/")
        assert format == "jpg"
        assert options["type"] == "authenticated"
        return "https://download.example/private"


@pytest.mark.asyncio
async def test_cloudinary_storage_uses_authenticated_owner_scoped_assets() -> None:
    client = FakeCloudinaryClient()
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "image/jpeg", "content-length": "5"},
                content=b"image",
            )
        )
    )
    storage = _storage(client, http)

    stored = await storage.put_private_image(
        owner_id=OWNER,
        category="diagnosis",
        content=b"image",
    )
    content = await storage.read_private(owner_id=OWNER, key=stored.key)
    await storage.delete_private(owner_id=OWNER, key=stored.key)

    assert stored.key.startswith(f"kishan-saathi-private/{OWNER}/diagnosis/")
    assert client.upload_options["type"] == "authenticated"
    assert client.upload_options["overwrite"] is False
    assert content == b"image"
    assert client.destroyed == stored.key
    assert client.destroy_options["invalidate"] is True
    await http.aclose()


@pytest.mark.asyncio
async def test_cloudinary_storage_rejects_cross_owner_access_before_network() -> None:
    client = FakeCloudinaryClient()
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(500)))
    storage = _storage(client, http)
    key = f"kishan-saathi-private/{OWNER}/diagnosis/object"

    with pytest.raises(ApplicationError) as read_error:
        await storage.read_private(owner_id=OTHER, key=key)
    with pytest.raises(ApplicationError) as delete_error:
        await storage.delete_private(owner_id=OTHER, key=key)

    assert read_error.value.code == "STORAGE_OBJECT_NOT_FOUND"
    assert delete_error.value.code == "STORAGE_OBJECT_NOT_FOUND"
    assert client.destroyed is None
    await http.aclose()


def _storage(
    client: FakeCloudinaryClient,
    http: httpx.AsyncClient,
) -> CloudinaryObjectStorage:
    return CloudinaryObjectStorage(
        Settings(
            _env_file=None,
            storage_backend="cloudinary",
            cloudinary_cloud_name="cloud",
            cloudinary_api_key="key",
            cloudinary_api_secret="secret",
        ),
        client=cast(CloudinaryClient, client),
        http_client=http,
    )
