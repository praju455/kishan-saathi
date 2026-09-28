"""Private object-storage provider contract."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Opaque private object reference persisted by product modules."""

    key: str
    size_bytes: int
    media_type: str


class ObjectStorageProvider(Protocol):
    """Storage operations that avoid exposing filesystem or provider details."""

    async def put_private_image(
        self,
        *,
        owner_id: UUID,
        category: str,
        content: bytes,
    ) -> StoredObject: ...

    async def read_private(self, *, owner_id: UUID, key: str) -> bytes: ...

    async def delete_private(self, *, owner_id: UUID, key: str) -> None: ...

    async def close(self) -> None: ...
