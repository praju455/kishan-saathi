"""Transaction-isolated, backend-owned plot forecast tool."""

from uuid import UUID

from app.core.config import Settings
from app.database.session import DatabasePort
from app.integrations.weather.provider import CurrentWeatherProvider, ForecastWeatherProvider
from app.modules.farms.repository import FarmRepository
from app.modules.weather.repository import WeatherRepository
from app.modules.weather.schemas import CurrentWeatherResponse, PlotForecastResponse
from app.modules.weather.service import WeatherService


class DatabasePlotWeatherTool:
    """Open a dedicated unit of work so tool caching cannot commit chat writes."""

    def __init__(
        self,
        *,
        settings: Settings,
        database: DatabasePort,
        current_provider: CurrentWeatherProvider,
        forecast_provider: ForecastWeatherProvider,
    ) -> None:
        self._settings = settings
        self._database = database
        self._current_provider = current_provider
        self._forecast_provider = forecast_provider

    async def current(self, farmer_id: UUID, plot_id: UUID) -> CurrentWeatherResponse:
        async with self._database.session() as session:
            service = WeatherService(
                settings=self._settings,
                repository=WeatherRepository(session),
                farms=FarmRepository(session),
                current_provider=self._current_provider,
                forecast_provider=self._forecast_provider,
            )
            return await service.current_for_plot(farmer_id, plot_id)

    async def forecast(self, farmer_id: UUID, plot_id: UUID) -> PlotForecastResponse:
        async with self._database.session() as session:
            service = WeatherService(
                settings=self._settings,
                repository=WeatherRepository(session),
                farms=FarmRepository(session),
                current_provider=self._current_provider,
                forecast_provider=self._forecast_provider,
            )
            return await service.forecast_for_plot(farmer_id, plot_id)
