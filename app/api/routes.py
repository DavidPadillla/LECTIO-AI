from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.api.schemas import (
    AnalyticsResponse,
    BookPrediction,
    DatabaseStatusResponse,
    HealthResponse,
    MetricsResponse,
    ModelStatusResponse,
    ModelUpdateStatusResponse,
    PredictionsResponse,
    Recommendation,
    RecommendationsResponse,
    WelcomeResponse,
)
from app.database import get_database_diagnostics, get_db
from app.database.diagnostics import DatabaseSchemaError
from app.paths import DATA_DIR
from app.services.prediction import (
    ModelNotTrainedError,
    forecast_all,
    get_model_bundle,
)
from app.services.recommendation import build_recommendations
from app.services.weekly_update import WeeklyModelUpdater

REPORT_DIR = DATA_DIR / "reports"
router = APIRouter()


def _load_bundle() -> dict[str, Any]:
    try:
        return get_model_bundle()
    except ModelNotTrainedError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "El modelo todavía no está entrenado. Ejecuta el pipeline de entrenamiento.",
            },
        ) from None


@lru_cache(maxsize=2)
def _read_report(name: str, modified_ns: int) -> dict[str, Any]:
    path = REPORT_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def _load_report(name: str) -> dict[str, Any] | None:
    path = REPORT_DIR / name
    try:
        modified_ns = path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    return _read_report(name, modified_ns)


@router.get("/", response_model=WelcomeResponse, tags=["general"])
def read_root() -> dict[str, str]:
    return {"message": "LECTIO-AI: sistema de análisis y pronóstico de demanda bibliotecaria."}


@router.get("/health", response_model=HealthResponse, tags=["general"])
def read_health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/database/status",
    response_model=DatabaseStatusResponse,
    tags=["database"],
)
def read_database_status(session: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        return get_database_diagnostics(session)
    except DatabaseSchemaError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "No se pudo completar el diagnóstico por diferencias en el esquema.",
            },
        ) from None


@router.get("/model/status", response_model=ModelStatusResponse, tags=["model"])
def read_model_status() -> dict[str, Any]:
    try:
        bundle = get_model_bundle()
    except ModelNotTrainedError:
        return {
            "trained": False,
            "trained_at": None,
            "metrics_available": False,
            "training_samples": None,
            "data_type": None,
            "model_type": None,
            "metrics": None,
        }
    return {
        "trained": True,
        "trained_at": bundle["trained_at"],
        "metrics_available": True,
        "training_samples": bundle["training_samples"],
        "data_type": bundle["data_type"],
        "model_type": bundle["model_config"]["type"],
        "metrics": bundle["evaluation"],
    }


@router.get(
    "/model/update-status",
    response_model=ModelUpdateStatusResponse,
    tags=["model"],
)
def read_model_update_status(
    request: Request,
) -> dict[str, Any]:
    updater: WeeklyModelUpdater | None = getattr(
        request.app.state,
        "weekly_updater",
        None,
    )
    if updater is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "El estado del programador no está disponible.",
            },
        )
    return updater.get_status()


@router.get("/predictions", response_model=PredictionsResponse, tags=["predictions"])
def read_predictions(
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    result = forecast_all(_load_bundle())
    result["books"] = result["books"][:limit]
    return result


@router.get(
    "/predictions/{book_id}",
    response_model=BookPrediction,
    tags=["predictions"],
)
def read_book_prediction(book_id: str) -> dict[str, Any]:
    result = forecast_all(_load_bundle())
    for prediction in result["books"]:
        if prediction["book_id"] == book_id:
            return prediction
    raise HTTPException(status_code=404, detail="No se encontró el libro.")


def _recommendations() -> dict[str, Any]:
    bundle = _load_bundle()
    forecasts = forecast_all(bundle)
    return build_recommendations(forecasts, bundle["book_history"])


@router.get(
    "/recommendations",
    response_model=RecommendationsResponse,
    tags=["recommendations"],
)
def read_recommendations(
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    result = _recommendations()
    result["recommendations"] = result["recommendations"][:limit]
    return result


@router.get(
    "/recommendations/{book_id}",
    response_model=Recommendation,
    tags=["recommendations"],
)
def read_book_recommendation(book_id: str) -> dict[str, Any]:
    result = _recommendations()
    for recommendation in result["recommendations"]:
        if recommendation["book_id"] == book_id:
            return recommendation
    raise HTTPException(status_code=404, detail="No se encontró el libro.")


@router.get("/metrics", response_model=MetricsResponse, tags=["model"])
def read_metrics() -> dict[str, Any]:
    bundle = _load_bundle()
    return {
        "data_type": bundle["data_type"],
        "trained_at": bundle["trained_at"],
        "evaluation": bundle["evaluation"],
    }


@router.get("/analytics", response_model=AnalyticsResponse, tags=["analytics"])
def read_analytics() -> dict[str, Any]:
    analytics = _load_report("analytics.json")
    if analytics is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "El análisis todavía no está disponible. Ejecuta el pipeline de datos.",
            },
        )
    return {"data_type": analytics["data_type"], "analytics": analytics}
