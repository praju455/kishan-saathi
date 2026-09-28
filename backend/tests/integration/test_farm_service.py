"""Integration tests for owner-scoped farm organization."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.errors import ApplicationError
from app.database.base import Base
from app.modules.farms.repository import FarmRepository
from app.modules.farms.schemas import (
    ActivityCreate,
    CropCreate,
    CropCycleClose,
    CropStageUpdate,
    CropUpdate,
    FarmCreate,
    PlotCreate,
)
from app.modules.farms.service import FarmService
from app.modules.users.models import FarmerProfile

FARMER_A = UUID("00000000-0000-0000-0000-00000000000a")
FARMER_B = UUID("00000000-0000-0000-0000-00000000000b")


@pytest.mark.asyncio
async def test_farm_plot_crop_and_activity_lifecycle_is_owner_scoped() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with sessions() as session:
            session.add_all(
                [
                    FarmerProfile(
                        id=FARMER_A,
                        cognito_sub="a",
                        cognito_username="a@example.com",
                        email="a@example.com",
                    ),
                    FarmerProfile(
                        id=FARMER_B,
                        cognito_sub="b",
                        cognito_username="b@example.com",
                        email="b@example.com",
                    ),
                ]
            )
            await session.commit()
            service = FarmService(FarmRepository(session))

            farm = await service.create_farm(FARMER_A, FarmCreate(name="Green Farm"))
            plot = await service.create_plot(
                FARMER_A,
                PlotCreate(
                    name="North Plot",
                    farm_id=farm.id,
                    latitude=Decimal("19.076000"),
                    longitude=Decimal("72.877700"),
                    location_label="Mumbai",
                    crops=[CropCreate(name="Tomato", stage="vegetative")],
                ),
            )
            crop = plot.crops[0]
            updated_crop = await service.update_crop_stage(
                FARMER_A, crop.id, CropStageUpdate(stage="flowering")
            )
            renamed_crop = await service.update_crop(
                FARMER_A,
                crop.id,
                CropUpdate(name="Cherry Tomato", variety="Local Red"),
            )
            activity = await service.create_activity(
                FARMER_A,
                plot.id,
                ActivityCreate(
                    crop_id=crop.id,
                    title="Started irrigation",
                    occurred_at=datetime.now(tz=UTC),
                ),
            )

            with pytest.raises(ApplicationError) as hidden:
                await service.get_plot(FARMER_B, plot.id)
            with pytest.raises(ApplicationError) as blocked:
                await service.delete_farm(FARMER_A, farm.id)
            with pytest.raises(ApplicationError) as crop_blocked:
                await service.delete_crop(FARMER_A, crop.id, confirm_history_loss=False)

            await service.delete_activity(FARMER_A, activity.id)
            await service.delete_crop(FARMER_A, crop.id, confirm_history_loss=True)
            await service.delete_plot(FARMER_A, plot.id)
            await service.delete_farm(FARMER_A, farm.id)

            assert await service.list_farms(FARMER_A) == []
    finally:
        await engine.dispose()

    assert updated_crop.stage == "flowering"
    assert renamed_crop.name == "Cherry Tomato"
    assert hidden.value.code == "PLOT_NOT_FOUND"
    assert blocked.value.code == "FARM_HAS_LINKED_DATA"
    assert crop_blocked.value.code == "CROP_HAS_LINKED_DATA"


@pytest.mark.asyncio
async def test_crop_cycle_can_close_but_not_move_or_reopen() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessions() as session:
            session.add(
                FarmerProfile(
                    id=FARMER_A,
                    cognito_sub="cycle",
                    cognito_username="cycle@example.com",
                    email="cycle@example.com",
                )
            )
            await session.commit()
            service = FarmService(FarmRepository(session))
            plot = await service.create_plot(
                FARMER_A,
                PlotCreate(
                    name="Cycle Plot",
                    latitude=Decimal("18"),
                    longitude=Decimal("74"),
                    crops=[CropCreate(name="Rice", stage="harvest")],
                ),
            )
            today = datetime.now(tz=UTC).date()
            closed = await service.close_crop_cycle(
                FARMER_A,
                plot.crops[0].id,
                CropCycleClose(ended_on=today),
            )
            renamed = await service.update_crop(
                FARMER_A, closed.id, CropUpdate(name="Archived Rice")
            )
            with pytest.raises(ApplicationError) as immutable:
                await service.update_crop(FARMER_A, closed.id, CropUpdate(stage="post-harvest"))
            with pytest.raises(ApplicationError) as stage_immutable:
                await service.update_crop_stage(
                    FARMER_A, closed.id, CropStageUpdate(stage="post-harvest")
                )
    finally:
        await engine.dispose()

    assert closed.cycle_ended_on == today
    assert renamed.name == "Archived Rice"
    assert immutable.value.code == "CROP_CYCLE_CLOSED"
    assert stage_immutable.value.code == "CROP_CYCLE_CLOSED"


@pytest.mark.asyncio
async def test_plot_requires_owning_farmer_farm() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with sessions() as session:
            session.add_all(
                [
                    FarmerProfile(
                        id=FARMER_A,
                        cognito_sub="a",
                        cognito_username="a@example.com",
                        email="a@example.com",
                    ),
                    FarmerProfile(
                        id=FARMER_B,
                        cognito_sub="b",
                        cognito_username="b@example.com",
                        email="b@example.com",
                    ),
                ]
            )
            await session.commit()
            service = FarmService(FarmRepository(session))
            foreign_farm = await service.create_farm(FARMER_B, FarmCreate(name="Other Farm"))

            with pytest.raises(ApplicationError) as raised:
                await service.create_plot(
                    FARMER_A,
                    PlotCreate(
                        name="Invalid Plot",
                        farm_id=foreign_farm.id,
                        latitude=Decimal("20"),
                        longitude=Decimal("75"),
                        crops=[CropCreate(name="Rice", stage="seedling")],
                    ),
                )
    finally:
        await engine.dispose()

    assert raised.value.code == "FARM_NOT_FOUND"
