from __future__ import annotations

import os
import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.paths import DATA_DIR, PROJECT_ROOT
from app.services.data_preparation import FEATURE_COLUMNS
from app.services.process_lock import ProcessFileLock
from app.services.training import MODEL_FILENAME

BUNDLED_MODEL_PATH = PROJECT_ROOT / "data" / "models" / MODEL_FILENAME
MODEL_PATH = DATA_DIR / "models" / MODEL_FILENAME
FORECAST_DAYS = 30


class ModelNotTrainedError(FileNotFoundError):
    pass


def ensure_model_available() -> bool:
    if MODEL_PATH.is_file():
        return True
    if not BUNDLED_MODEL_PATH.is_file():
        return False

    seed_lock = ProcessFileLock(DATA_DIR / "state" / "model-seed.lock")
    if not seed_lock.acquire():
        for _ in range(50):
            if MODEL_PATH.is_file():
                return True
            time.sleep(0.1)
        raise RuntimeError("No se pudo completar la inicialización del modelo.")

    temp_path: Path | None = None
    try:
        if MODEL_PATH.is_file():
            return True
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=MODEL_PATH.parent,
            prefix=f".{MODEL_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as target:
            temp_path = Path(target.name)
            with BUNDLED_MODEL_PATH.open("rb") as source:
                shutil.copyfileobj(source, target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_path, MODEL_PATH)
        return True
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        seed_lock.release()


@lru_cache(maxsize=4)
def _load_model_bundle(modified_ns: int, file_size: int) -> dict[str, Any]:
    del modified_ns, file_size
    if not MODEL_PATH.is_file():
        raise ModelNotTrainedError("El modelo todavía no está entrenado.")
    try:
        bundle = joblib.load(MODEL_PATH)
    except (OSError, EOFError, ValueError):
        raise ModelNotTrainedError("El artefacto del modelo no es válido.") from None
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelNotTrainedError("El artefacto del modelo no es válido.")
    return bundle


def get_model_bundle() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise ModelNotTrainedError("El modelo todavía no está entrenado.")
    try:
        stat = MODEL_PATH.stat()
    except OSError:
        raise ModelNotTrainedError("El artefacto del modelo no está disponible.") from None
    return _load_model_bundle(stat.st_mtime_ns, stat.st_size)


def clear_model_cache() -> None:
    _load_model_bundle.cache_clear()


def _week_features(
    book_id: str,
    category: str,
    period_start: pd.Timestamp,
    demand_history: list[float],
) -> dict[str, Any]:
    history = np.asarray(demand_history, dtype=float)
    week_number = int(period_start.isocalendar().week)
    return {
        "book_id": book_id,
        "category": category,
        "month_sin": float(np.sin(2 * np.pi * period_start.month / 12)),
        "month_cos": float(np.cos(2 * np.pi * period_start.month / 12)),
        "week_sin": float(np.sin(2 * np.pi * week_number / 52.18)),
        "week_cos": float(np.cos(2 * np.pi * week_number / 52.18)),
        "lag_1": float(history[-1]),
        "lag_2": float(history[-2]),
        "lag_4": float(history[-4]),
        "rolling_4_mean": float(history[-4:].mean()),
        "rolling_8_mean": float(history[-8:].mean()),
        "trend_4_vs_8": float(history[-4:].mean() - history[-8:].mean()),
    }


def forecast_all(bundle: dict[str, Any], days: int = FORECAST_DAYS) -> dict[str, Any]:
    if days < 1:
        raise ValueError("El horizonte debe ser positivo.")
    history = pd.DataFrame(bundle["history"])
    books = pd.DataFrame(bundle["book_history"])
    if history.empty:
        raise ModelNotTrainedError("El modelo no contiene historial temporal.")

    last_period = pd.Timestamp(history["period_start"].max())
    horizon_start = last_period + pd.Timedelta(days=7)
    horizon_end = horizon_start + pd.Timedelta(days=days)
    periods = []
    period_start = horizon_start
    while period_start < horizon_end:
        overlap_days = min(7, (horizon_end - period_start).days)
        periods.append((period_start, overlap_days))
        period_start += pd.Timedelta(days=7)

    forecasts = []
    for book in books.to_dict(orient="records"):
        book_id = str(book["book_id"])
        if not bool(book["sufficient_history"]):
            forecasts.append(
                {
                    "book_id": book_id,
                    "category": str(book["category"]),
                    "status": "DATOS_INSUFICIENTES",
                    "forecast_demand_30_days": None,
                    "weekly_forecast": [],
                    "historical_demand": int(book["total_reservations"]),
                }
            )
            continue

        book_history = history.loc[history["book_id"].astype(str) == book_id].sort_values(
            "period_start"
        )
        demand_history = book_history["demand"].astype(float).tolist()
        predictions = []
        total = 0.0
        for future_period, covered_days in periods:
            row = _week_features(
                book_id,
                str(book["category"]),
                future_period,
                demand_history,
            )
            raw_prediction = float(
                bundle["pipeline"].predict(
                    pd.DataFrame([row], columns=FEATURE_COLUMNS)
                )[0]
            )
            prediction = max(0.0, raw_prediction)
            weighted = prediction * covered_days / 7
            predictions.append(
                {
                    "period_start": future_period.date().isoformat(),
                    "days_covered": covered_days,
                    "predicted_demand": prediction,
                }
            )
            total += weighted
            demand_history.append(prediction)
        forecasts.append(
            {
                "book_id": book_id,
                "category": str(book["category"]),
                "status": "PREDICCION_DISPONIBLE",
                "forecast_demand_30_days": round(total, 2),
                "weekly_forecast": predictions,
                "historical_demand": int(book["total_reservations"]),
            }
        )

    return {
        "data_type": bundle["data_type"],
        "horizon_days": days,
        "forecast_start": horizon_start.date().isoformat(),
        "forecast_end_exclusive": horizon_end.date().isoformat(),
        "books": forecasts,
    }
