"""Narrow feature ports that keep chat independent from concrete product modules."""

from typing import Protocol
from uuid import UUID


class JsonResponse(Protocol):
    def model_dump_json(self) -> str: ...


class PlotWeatherPort(Protocol):
    """Optional owner-authorized weather context available to a plot chat."""

    async def current(self, farmer_id: UUID, plot_id: UUID) -> JsonResponse: ...

    async def forecast(self, farmer_id: UUID, plot_id: UUID) -> JsonResponse: ...
