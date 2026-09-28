"""Best-effort immediate deletion backed by durable retry records."""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.storage.provider import ObjectStorageProvider
from app.modules.storage_cleanup.models import ObjectDeletionJob


class ObjectCleanupService:
    def __init__(
        self,
        session: AsyncSession,
        storage: ObjectStorageProvider,
        *,
        backoff_base_seconds: int = 60,
        backoff_max_seconds: int = 21600,
    ) -> None:
        self._session = session
        self._storage = storage
        self._backoff_base_seconds = backoff_base_seconds
        self._backoff_max_seconds = backoff_max_seconds
        self._logger = structlog.get_logger(__name__)

    def enqueue(self, owner_id: UUID, object_key: str, reason: str) -> ObjectDeletionJob:
        job = ObjectDeletionJob(
            owner_id=owner_id,
            object_key=object_key,
            reason=reason,
        )
        self._session.add(job)
        return job

    async def process(self, job_ids: list[UUID]) -> None:
        """Attempt queued work; failures remain durable for later retry."""

        for job_id in job_ids:
            job = await self._session.scalar(
                select(ObjectDeletionJob).where(ObjectDeletionJob.id == job_id)
            )
            if job is None:
                continue
            owner_id = job.owner_id
            object_key = job.object_key
            reason = job.reason
            # Release the database transaction before remote object deletion.
            # Cloudinary/local deletion is idempotent; the durable row remains the
            # source of retry state until a short final transaction removes it.
            await self._session.commit()
            try:
                await self._storage.delete_private(owner_id=owner_id, key=object_key)
            except Exception as exc:
                job = await self._session.scalar(
                    select(ObjectDeletionJob)
                    .where(ObjectDeletionJob.id == job_id)
                    .with_for_update()
                )
                if job is None:
                    continue
                job.attempts += 1
                job.last_attempted_at = datetime.now(tz=UTC)
                job.last_error_type = type(exc).__name__
                delay = min(
                    self._backoff_base_seconds * (2 ** min(job.attempts - 1, 20)),
                    self._backoff_max_seconds,
                )
                job.next_attempt_at = datetime.now(tz=UTC) + timedelta(seconds=delay)
                await self._session.commit()
                self._logger.warning(
                    "object_cleanup.deletion_failed",
                    job_id=str(job.id),
                    reason=reason,
                    attempts=job.attempts,
                    error_type=type(exc).__name__,
                )
            else:
                job = await self._session.scalar(
                    select(ObjectDeletionJob)
                    .where(ObjectDeletionJob.id == job_id)
                    .with_for_update()
                )
                if job is None:
                    continue
                await self._session.delete(job)
                await self._session.commit()

    async def retry_owner(self, owner_id: UUID, *, limit: int = 100) -> int:
        job_ids = list(
            await self._session.scalars(
                select(ObjectDeletionJob.id)
                .where(ObjectDeletionJob.owner_id == owner_id)
                .order_by(ObjectDeletionJob.created_at)
                .limit(limit)
            )
        )
        await self.process(job_ids)
        return len(job_ids)

    async def process_due(self, *, limit: int = 50) -> int:
        """Claim due jobs without blocking other replicas, then process them."""

        now = datetime.now(tz=UTC)
        jobs = list(
            await self._session.scalars(
                select(ObjectDeletionJob)
                .where(ObjectDeletionJob.next_attempt_at <= now)
                .order_by(ObjectDeletionJob.next_attempt_at, ObjectDeletionJob.created_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            )
        )
        job_ids = [job.id for job in jobs]
        if jobs:
            lease_until = now + timedelta(minutes=5)
            for job in jobs:
                job.next_attempt_at = lease_until
            await self._session.commit()
        await self.process(job_ids)
        return len(job_ids)
