from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.data_preparation import (
    FEATURE_COLUMNS,
    InsufficientDataError,
    PreparedData,
)

CATEGORICAL_FEATURES = ["book_id", "category"]
NUMERIC_FEATURES = [
    feature for feature in FEATURE_COLUMNS if feature not in CATEGORICAL_FEATURES
]
MODEL_FILENAME = "demand_model.joblib"
MIN_TRAIN_PERIODS = 8
TEST_FRACTION = 0.2
MODEL_CANDIDATES = (
    {"hidden_layer_sizes": (8,), "alpha": 0.1},
    {"hidden_layer_sizes": (16,), "alpha": 0.1},
    {"hidden_layer_sizes": (16,), "alpha": 1.0},
    {"hidden_layer_sizes": (32,), "alpha": 1.0},
)


def build_model(config: dict[str, Any] | None = None) -> Pipeline:
    selected = config or MODEL_CANDIDATES[2]
    preprocessing = ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numbers", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    estimator = MLPRegressor(
        hidden_layer_sizes=selected["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=selected["alpha"],
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=False,
        random_state=42,
        tol=1e-3,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def _select_model_config(train: pd.DataFrame) -> dict[str, Any]:
    """Select MLP settings using expanding chronological validation folds."""
    periods = np.sort(train["period_start"].unique())
    fold_count = min(3, max(1, (len(periods) - MIN_TRAIN_PERIODS) // 3))
    validation_size = max(1, (len(periods) - MIN_TRAIN_PERIODS) // (fold_count + 1))
    fold_starts = [
        len(periods) - validation_size * (fold_count - index)
        for index in range(fold_count)
    ]
    folds = [
        (set(periods[:start]), set(periods[start : start + validation_size]))
        for start in fold_starts
        if start >= MIN_TRAIN_PERIODS
    ]
    if not folds:
        return {**MODEL_CANDIDATES[2], "validation_mae": None,
                "validation_rmse": None, "validation_folds": 0}

    scores = []
    for config in MODEL_CANDIDATES:
        actual_parts = []
        predicted_parts = []
        for train_periods, validation_periods in folds:
            fold_train = train.loc[train["period_start"].isin(train_periods)]
            fold_validation = train.loc[
                train["period_start"].isin(validation_periods)
            ]
            evaluator = build_model(config)
            evaluator.fit(fold_train[list(FEATURE_COLUMNS)], fold_train["target"])
            actual_parts.append(fold_validation["target"].to_numpy(dtype=float))
            predicted_parts.append(
                evaluator.predict(fold_validation[list(FEATURE_COLUMNS)])
            )
        actual = np.concatenate(actual_parts)
        predicted = np.concatenate(predicted_parts)
        metrics = _metrics(actual, predicted)
        scores.append((metrics["mae"] + metrics["rmse"], dict(config), metrics))

    _, selected, metrics = min(scores, key=lambda item: item[0])
    return {
        **selected,
        "validation_mae": metrics["mae"],
        "validation_rmse": metrics["rmse"],
        "validation_folds": len(folds),
    }


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
    }


def _evaluate(
    features: pd.DataFrame,
    incumbent_pipeline: Pipeline | None = None,
) -> dict[str, Any]:
    periods = np.sort(features["period_start"].unique())
    if len(periods) < MIN_TRAIN_PERIODS + 2:
        raise InsufficientDataError(
            "Hay muy pocos periodos semanales para una evaluación cronológica."
        )
    test_period_count = max(1, math.ceil(len(periods) * TEST_FRACTION))
    cutoff = periods[-test_period_count]
    train = features.loc[features["period_start"] < cutoff]
    test = features.loc[features["period_start"] >= cutoff].copy()
    if train["period_start"].nunique() < MIN_TRAIN_PERIODS or test.empty:
        raise InsufficientDataError(
            "La división cronológica no deja suficientes periodos de entrenamiento y prueba."
        )

    selected_config = _select_model_config(train)
    model_config = {
        "hidden_layer_sizes": selected_config["hidden_layer_sizes"],
        "alpha": selected_config["alpha"],
    }
    evaluator = build_model(model_config)
    evaluator.fit(train[list(FEATURE_COLUMNS)], train["target"])
    mlp_raw = evaluator.predict(test[list(FEATURE_COLUMNS)])
    baseline_raw = test["rolling_4_mean"].to_numpy(dtype=float)
    actual = test["target"].to_numpy(dtype=float)
    incumbent_metrics = None
    if incumbent_pipeline is not None:
        incumbent_prediction = incumbent_pipeline.predict(test[list(FEATURE_COLUMNS)])
        incumbent_metrics = _metrics(actual, incumbent_prediction)
    per_book = []
    comparison = test[["book_id", "target"]].copy()
    comparison["mlp_prediction_raw"] = mlp_raw
    comparison["baseline_prediction"] = baseline_raw
    train_book_totals = train.groupby("book_id")["target"].sum()
    active_book_totals = train_book_totals[train_book_totals > 0]
    low_demand_threshold = (
        float(active_book_totals.quantile(0.25))
        if not active_book_totals.empty
        else 0.0
    )
    low_demand_books = set(
        active_book_totals[active_book_totals <= low_demand_threshold].index
    )
    low_demand_test = test.loc[test["book_id"].isin(low_demand_books)]
    low_demand_metrics = None
    if not low_demand_test.empty:
        low_demand_actual = low_demand_test["target"].to_numpy(dtype=float)
        low_demand_metrics = {
            "books": int(low_demand_test["book_id"].nunique()),
            "test_samples": int(len(low_demand_test)),
            "mlp_regressor": _metrics(
                low_demand_actual,
                evaluator.predict(low_demand_test[list(FEATURE_COLUMNS)]),
            ),
            "baseline_rolling_4_mean": _metrics(
                low_demand_actual,
                low_demand_test["rolling_4_mean"].to_numpy(dtype=float),
            ),
        }
    for book_id, rows in comparison.groupby("book_id"):
        if len(rows) >= 4:
            per_book.append(
                {
                    "book_id": str(book_id),
                    "test_periods": int(len(rows)),
                    "mlp_mae": float(
                        mean_absolute_error(rows["target"], rows["mlp_prediction_raw"])
                    ),
                    "baseline_mae": float(
                        mean_absolute_error(rows["target"], rows["baseline_prediction"])
                    ),
                }
            )
    mlp_metrics = _metrics(actual, mlp_raw)
    baseline_metrics = _metrics(actual, baseline_raw)
    mlp_better_mae = mlp_metrics["mae"] < baseline_metrics["mae"]
    mlp_better_rmse = mlp_metrics["rmse"] < baseline_metrics["rmse"]
    return {
        "evaluation_type": "cronologica",
        "train_start": pd.Timestamp(train["period_start"].min()).date().isoformat(),
        "train_end": pd.Timestamp(train["period_start"].max()).date().isoformat(),
        "test_start": pd.Timestamp(test["period_start"].min()).date().isoformat(),
        "test_end": pd.Timestamp(test["period_start"].max()).date().isoformat(),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "mlp_regressor": mlp_metrics,
        "baseline_rolling_4_mean": baseline_metrics,
        "incumbent_model": incumbent_metrics,
        "mlp_raw_negative_predictions": int((mlp_raw < 0).sum()),
        "selected_model_config": {
            "hidden_layer_sizes": list(model_config["hidden_layer_sizes"]),
            "alpha": model_config["alpha"],
            "validation_mae": selected_config["validation_mae"],
            "validation_rmse": selected_config["validation_rmse"],
            "validation_folds": selected_config["validation_folds"],
        },
        "mlp_better_than_baseline_mae": mlp_better_mae,
        "mlp_better_than_baseline_rmse": mlp_better_rmse,
        "low_demand_evaluation": low_demand_metrics,
        "per_book_errors": per_book,
        "conclusion": (
            "El MLPRegressor supera al modelo de referencia en MAE y RMSE."
            if mlp_better_mae and mlp_better_rmse
            else "El MLPRegressor mejora MAE, pero no RMSE frente a la referencia."
            if mlp_better_mae
            else "El MLPRegressor no supera al modelo de referencia en MAE."
        ),
    }


def train_and_save(
    prepared: PreparedData,
    model_dir: Path,
    processed_dir: Path,
    report_dir: Path,
    incumbent_bundle: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    incumbent_pipeline = (
        incumbent_bundle.get("pipeline") if incumbent_bundle is not None else None
    )
    evaluation = _evaluate(prepared.features, incumbent_pipeline)
    selected_config = evaluation["selected_model_config"]
    final_model = build_model(
        {
            "hidden_layer_sizes": tuple(selected_config["hidden_layer_sizes"]),
            "alpha": selected_config["alpha"],
        }
    )
    final_model.fit(
        prepared.features[list(FEATURE_COLUMNS)],
        prepared.features["target"],
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "pipeline": final_model,
        "trained_at": trained_at,
        "data_type": prepared.report["data_type"],
        "training_samples": int(len(prepared.features)),
        "book_history": prepared.catalog.to_dict(orient="records"),
        "history": prepared.panel.to_dict(orient="records"),
        "feature_columns": list(FEATURE_COLUMNS),
        "evaluation": evaluation,
        "preparation": prepared.report,
        "model_config": {
            "type": "MLPRegressor",
            "hidden_layer_sizes": selected_config["hidden_layer_sizes"],
            "activation": "relu",
            "alpha": selected_config["alpha"],
            "max_iter": 1000,
            "early_stopping": False,
            "random_state": 42,
            "validation": "cronologica (último 20% de periodos)",
        },
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / MODEL_FILENAME)
    prepared.panel.to_csv(
        processed_dir / "weekly_demand.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    (report_dir / "preparation.json").write_text(
        json.dumps(prepared.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return bundle, evaluation
