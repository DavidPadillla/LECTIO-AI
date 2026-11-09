from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_analytics(
    panel: pd.DataFrame,
    catalog: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, Any]:
    totals = panel.groupby("book_id")["demand"].sum().sort_values(ascending=False)
    by_category = (
        panel.groupby("category", as_index=False)["demand"]
        .sum()
        .sort_values("demand", ascending=False)
    )
    monthly = (
        panel.assign(month=panel["period_start"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)["demand"]
        .sum()
        .sort_values("month")
    )
    by_weekday = (
        events.assign(weekday=events["created_at"].dt.day_name())
        .groupby("weekday", as_index=False)
        .size()
        .rename(columns={"size": "demand"})
    )
    mean = float(panel["demand"].mean())
    std = float(panel["demand"].std(ddof=0))
    nonzero_books = totals[totals > 0]
    high_threshold = float(nonzero_books.quantile(0.75)) if len(nonzero_books) else 0.0
    low_threshold = float(nonzero_books.quantile(0.25)) if len(nonzero_books) else 0.0

    book_summary = [
        {
            "book_id": str(book_id),
            "total_demand": int(total),
            "activity_level": (
                "ALTA"
                if total >= high_threshold and total > 0
                else "BAJA"
                if total <= low_threshold
                else "MEDIA"
            ),
        }
        for book_id, total in totals.items()
    ]
    first_half = panel.loc[panel["period_start"] <= panel["period_start"].median(), "demand"]
    second_half = panel.loc[panel["period_start"] > panel["period_start"].median(), "demand"]
    first_mean = float(first_half.mean()) if not first_half.empty else 0.0
    second_mean = float(second_half.mean()) if not second_half.empty else 0.0
    change_pct = (
        ((second_mean - first_mean) / first_mean) * 100
        if first_mean > 0
        else None
    )
    trend = (
        "crecimiento"
        if second_mean > first_mean
        else "disminucion"
        if second_mean < first_mean
        else "estable"
    )
    history_span_days = (
        panel["period_start"].max() - panel["period_start"].min()
    ).days
    enough_for_seasonality = history_span_days >= 730
    return {
        "data_type": "datos_reales",
        "data_source": "Neon PostgreSQL",
        "total_demand": int(panel["demand"].sum()),
        "books_analyzed": int(panel["book_id"].nunique()),
        "periods_analyzed": int(panel["period_start"].nunique()),
        "period_start": panel["period_start"].min().date().isoformat(),
        "period_end": panel["period_start"].max().date().isoformat(),
        "history_span_days": int(history_span_days),
        "descriptive_statistics": {
            "mean_weekly_book_demand": mean,
            "std_weekly_book_demand": std,
            "median_weekly_book_demand": float(panel["demand"].median()),
            "zero_demand_book_periods": int((panel["demand"] == 0).sum()),
            "nonzero_demand_book_periods": int((panel["demand"] > 0).sum()),
        },
        "demand_by_book": book_summary,
        "demand_by_category": [
            {"category": str(row.category), "demand": int(row.demand)}
            for row in by_category.itertuples(index=False)
        ],
        "monthly_demand": [
            {"month": str(row.month), "demand": int(row.demand)}
            for row in monthly.itertuples(index=False)
        ],
        "weekday_demand": [
            {"weekday": str(row.weekday), "demand": int(row.demand)}
            for row in by_weekday.itertuples(index=False)
        ],
        "trend": {
            "direction": trend,
            "change_percent": change_pct,
            "method": "comparing mean weekly demand across chronological halves",
        },
        "seasonality": {
            "assessed": enough_for_seasonality,
            "finding": (
                "Se requieren al menos dos años para evaluar estacionalidad anual."
                if not enough_for_seasonality
                else "La estacionalidad se describe, pero no se afirma como patrón probado."
            ),
        },
        "insufficient_history_books": int(
            (~catalog["sufficient_history"]).sum()
        ),
    }


def save_reports(
    panel: pd.DataFrame,
    catalog: pd.DataFrame,
    events: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analytics = build_analytics(panel, catalog, events)
    (output_dir / "analytics.json").write_text(
        json.dumps(analytics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    monthly = pd.DataFrame(analytics["monthly_demand"])
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(monthly["month"], monthly["demand"], marker="o")
    axis.set(title="Reservas históricas por mes", xlabel="Mes", ylabel="Reservas")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "demand_over_time.png", dpi=140)
    plt.close(fig)

    top_books = pd.DataFrame(analytics["demand_by_book"]).head(15).sort_values("total_demand")
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(top_books["book_id"], top_books["total_demand"])
    axis.set(title="Libros con mayor demanda", xlabel="Reservas")
    fig.tight_layout()
    fig.savefig(output_dir / "top_books.png", dpi=140)
    plt.close(fig)

    categories = pd.DataFrame(analytics["demand_by_category"])
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(categories["category"], categories["demand"])
    axis.set(title="Demanda por categoría", xlabel="Reservas")
    fig.tight_layout()
    fig.savefig(output_dir / "demand_by_category.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(panel["demand"], bins=min(25, max(5, int(panel["demand"].max()) + 1)))
    axis.set(title="Distribución de solicitudes por libro y semana", xlabel="Solicitudes", ylabel="Periodos")
    fig.tight_layout()
    fig.savefig(output_dir / "demand_distribution.png", dpi=140)
    plt.close(fig)

    midpoint = panel["period_start"].median()
    comparison = panel.assign(
        period=np.where(panel["period_start"] <= midpoint, "Primer periodo", "Segundo periodo")
    )
    comparison = comparison.groupby(["period", "category"], as_index=False)["demand"].mean()
    pivot = comparison.pivot(index="category", columns="period", values="demand").fillna(0)
    fig, axis = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=axis)
    axis.set(title="Demanda semanal media: primera y segunda mitad", ylabel="Solicitudes promedio")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "period_comparison.png", dpi=140)
    plt.close(fig)
    return analytics
