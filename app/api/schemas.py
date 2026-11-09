from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str


class WelcomeResponse(BaseModel):
    message: str


class ColumnInfo(BaseModel):
    name: str
    type: str
    postgres_type: str


class TableInfo(BaseModel):
    schema_name: str = Field(alias="schema")
    columns: list[ColumnInfo]


class CountInfo(BaseModel):
    libros_fisicos: int
    reservas: int
    reservas_sinteticas: int
    libros_demo: int


class StateCount(BaseModel):
    estado: str | None
    cantidad: int


class MonthCount(BaseModel):
    mes: str | None
    cantidad: int


class DatabaseStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    schema_info: dict[str, TableInfo] = Field(alias="schema")
    counts: CountInfo
    reservas_por_estado: list[StateCount]
    reservas_por_mes: list[MonthCount]


class ModelStatusResponse(BaseModel):
    trained: bool
    trained_at: str | None = None
    metrics_available: bool
    training_samples: int | None = None
    data_type: str | None = None
    model_type: str | None = None
    metrics: dict[str, Any] | None = None


class ModelUpdateStatusResponse(BaseModel):
    scheduler_active: bool
    timezone: str
    schedule: dict[str, Any]
    last_execution_at: str | None
    next_execution_at: str | None
    last_result: dict[str, Any] | None
    last_training_metrics: dict[str, float] | None
    previous_model_preserved: bool
    successful_runs: int
    failed_runs: int


class WeeklyPrediction(BaseModel):
    period_start: str
    days_covered: int
    predicted_demand: float


class BookPrediction(BaseModel):
    book_id: str
    titulo: str | None
    category: str
    status: Literal["PREDICCION_DISPONIBLE", "DATOS_INSUFICIENTES"]
    forecast_demand_30_days: float | None
    weekly_forecast: list[WeeklyPrediction]
    historical_demand: int


class PredictionsResponse(BaseModel):
    data_type: str
    horizon_days: int
    forecast_start: str
    forecast_end_exclusive: str
    books: list[BookPrediction]


class Recommendation(BaseModel):
    book_id: str
    titulo: str | None
    category: str
    data_type: str
    historical_demand: int
    forecast_demand_30_days: float | None
    stock_available: int
    reserved_units: int
    total_inventory: int
    risk_level: Literal["ALTO", "MEDIO", "BAJO", "DATOS_INSUFICIENTES"]
    recommended_replenishment: int
    safety_margin: float
    formula: str


class RecommendationsResponse(BaseModel):
    data_type: str
    horizon_days: int
    safety_margin: float
    recommendations: list[Recommendation]


class MetricsResponse(BaseModel):
    data_type: str
    trained_at: str
    evaluation: dict[str, Any]


class AnalyticsResponse(BaseModel):
    data_type: str
    analytics: dict[str, Any]
