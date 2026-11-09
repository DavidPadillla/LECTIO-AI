from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database.diagnostics import (
    _inspect_tables,
    _qualified_table,
    _quote_identifier,
)

ALLOWED_STATES = {"ACTIVA", "DEVUELTA", "CANCELADA"}
WEEKLY_FREQUENCY = "W-MON"
MIN_BOOK_RESERVATIONS = 8
MIN_BOOK_ACTIVE_WEEKS = 3
MIN_HISTORY_WEEKS = 12
FEATURE_COLUMNS = (
    "book_id",
    "category",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_4_mean",
    "rolling_8_mean",
    "trend_4_vs_8",
)


class InsufficientDataError(ValueError):
    pass


@dataclass
class PreparedData:
    panel: pd.DataFrame
    features: pd.DataFrame
    catalog: pd.DataFrame
    events: pd.DataFrame
    report: dict[str, Any]


def load_source_data(session: Session) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tables = _inspect_tables(session)
    books = _qualified_table(tables["libros_fisicos"], "libros_fisicos")
    reservations = _qualified_table(tables["reservas"], "reservas")
    users = _qualified_table(tables["usuarios"], "usuarios")

    book_rows = session.execute(
        text(
            f"""
            SELECT {_quote_identifier('id')} AS book_id,
                   {_quote_identifier('categoria')} AS category,
                   {_quote_identifier('stock')} AS stock,
                   {_quote_identifier('reservado')} AS reserved,
                   ({_quote_identifier('descripcion')} ILIKE :demo_marker) AS is_demo
            FROM {books}
            """
        ),
        {"demo_marker": "%[DEMO-LECTIO]%"},
    ).mappings()

    reservation_rows = session.execute(
        text(
            f"""
            SELECT r.{_quote_identifier('id')} AS reservation_id,
                   r.{_quote_identifier('libro_fisico_id')} AS book_id,
                   r.{_quote_identifier('fecha_creacion')} AS created_at,
                   upper(trim(r.{_quote_identifier('estado')})) AS state,
                   (
                       coalesce(u.{_quote_identifier('username')} LIKE :username_prefix, false)
                       OR lower(trim(coalesce(r.{_quote_identifier('correo')}, '')))
                           LIKE :email_suffix
                   ) AS is_synthetic
            FROM {reservations} AS r
            LEFT JOIN {users} AS u
                ON u.{_quote_identifier('id')} = r.{_quote_identifier('usuario_id')}
            """
        ),
        {
            "username_prefix": "demo_user_%",
            "email_suffix": "%@demo.lectio.local",
        },
    ).mappings()

    return [dict(row) for row in book_rows], [dict(row) for row in reservation_rows]


def _week_start(values: pd.Series) -> pd.Series:
    return values.dt.to_period("W-SUN").dt.start_time


def _build_panel(events: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        raise InsufficientDataError("No hay reservas demostrativas válidas para preparar.")

    first_week = events["created_at"].min().to_period("W-SUN").start_time
    last_week = events["created_at"].max().to_period("W-SUN").start_time
    periods = pd.date_range(first_week, last_week, freq=WEEKLY_FREQUENCY)
    demo_catalog = catalog.loc[catalog["is_demo"]].copy()
    if demo_catalog.empty:
        raise InsufficientDataError("No se encontraron libros demostrativos.")

    observed = (
        events.assign(period_start=_week_start(events["created_at"]))
        .groupby(["book_id", "period_start"], as_index=False)
        .size()
        .rename(columns={"size": "demand"})
    )
    index = pd.MultiIndex.from_product(
        [demo_catalog["book_id"].tolist(), periods],
        names=["book_id", "period_start"],
    )
    panel = (
        observed.set_index(["book_id", "period_start"])
        .reindex(index, fill_value=0)
        .reset_index()
    )
    panel["demand"] = panel["demand"].astype("int64")
    panel = panel.merge(
        demo_catalog[["book_id", "category"]],
        on="book_id",
        how="left",
        validate="many_to_one",
    )
    panel["category"] = panel["category"].fillna("SIN_CATEGORIA").astype(str)
    return panel.sort_values(["book_id", "period_start"]).reset_index(drop=True)


def _build_features(panel: pd.DataFrame) -> pd.DataFrame:
    features = panel.copy()
    grouped = features.groupby("book_id", sort=False)["demand"]
    for lag in (1, 2, 4):
        features[f"lag_{lag}"] = grouped.shift(lag)
    features["rolling_4_mean"] = grouped.transform(
        lambda values: values.shift(1).rolling(4, min_periods=4).mean()
    )
    features["rolling_8_mean"] = grouped.transform(
        lambda values: values.shift(1).rolling(8, min_periods=8).mean()
    )
    features["trend_4_vs_8"] = features["rolling_4_mean"] - features["rolling_8_mean"]
    features["month"] = features["period_start"].dt.month
    features["week_of_year"] = features["period_start"].dt.isocalendar().week.astype(int)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    features["week_sin"] = np.sin(2 * np.pi * features["week_of_year"] / 52.18)
    features["week_cos"] = np.cos(2 * np.pi * features["week_of_year"] / 52.18)
    features["target"] = features["demand"].astype(float)
    return features.dropna(
        subset=["lag_1", "lag_2", "lag_4", "rolling_4_mean", "rolling_8_mean"]
    ).reset_index(drop=True)


def prepare_data(
    book_records: list[dict[str, Any]],
    reservation_records: list[dict[str, Any]],
) -> PreparedData:
    books = pd.DataFrame(book_records)
    reservations = pd.DataFrame(reservation_records)
    required_book_fields = {"book_id", "category", "stock", "reserved", "is_demo"}
    required_reservation_fields = {
        "reservation_id",
        "book_id",
        "created_at",
        "state",
        "is_synthetic",
    }
    if not required_book_fields.issubset(books.columns):
        raise ValueError("El catálogo no contiene las columnas requeridas.")
    if not required_reservation_fields.issubset(reservations.columns):
        raise ValueError("Las reservas no contienen las columnas requeridas.")

    books = books.drop_duplicates("book_id", keep="first").copy()
    books["book_id"] = books["book_id"].astype("string").str.strip()
    books = books.dropna(subset=["book_id"])
    books = books.loc[books["book_id"].ne("")].copy()
    books["category"] = books["category"].astype("string").str.strip()
    books["category"] = books["category"].replace("", pd.NA).fillna("SIN_CATEGORIA")
    books["stock"] = pd.to_numeric(books["stock"], errors="coerce")
    books["reserved"] = pd.to_numeric(books["reserved"], errors="coerce")
    books = books.dropna(subset=["stock", "reserved"])
    books = books.loc[(books["stock"] >= 0) & (books["reserved"] >= 0)].copy()
    books["stock"] = books["stock"].astype(int)
    books["reserved"] = books["reserved"].astype(int)
    books["is_demo"] = books["is_demo"].fillna(False).astype(bool)
    books["category"] = books["category"].astype(str)

    original_rows = len(reservations)
    reservations["reservation_id"] = reservations["reservation_id"].astype("string").str.strip()
    reservations["book_id"] = reservations["book_id"].astype("string").str.strip()
    reservations["created_at"] = pd.to_datetime(reservations["created_at"], errors="coerce")
    reservations["state"] = reservations["state"].astype("string").str.strip().str.upper()
    invalid_id_mask = (
        reservations["reservation_id"].isna()
        | reservations["reservation_id"].eq("")
        | reservations["book_id"].isna()
        | reservations["book_id"].eq("")
    )
    invalid_id_count = int(invalid_id_mask.sum())
    reservations = reservations.loc[~invalid_id_mask].copy()
    before_deduplication = len(reservations)
    reservations = reservations.sort_values("created_at").drop_duplicates(
        "reservation_id", keep="first"
    )
    duplicate_count = before_deduplication - len(reservations)
    invalid_date_mask = reservations["created_at"].isna()
    invalid_date_count = int(invalid_date_mask.sum())
    reservations = reservations.loc[~invalid_date_mask].copy()
    invalid_state_mask = ~reservations["state"].isin(ALLOWED_STATES)
    invalid_state_count = int(invalid_state_mask.sum())
    reservations = reservations.loc[~invalid_state_mask].copy()
    reservations["is_synthetic"] = reservations["is_synthetic"].fillna(False).astype(bool)

    catalog_ids = set(books["book_id"])
    unknown_book_mask = ~reservations["book_id"].isin(catalog_ids)
    unknown_book_count = int(unknown_book_mask.sum())
    reservations = reservations.loc[~unknown_book_mask].copy()

    demo_ids = set(books.loc[books["is_demo"], "book_id"])
    non_demo_reservations = int((~reservations["is_synthetic"]).sum())
    non_demo_book_events = int(
        (reservations["is_synthetic"] & ~reservations["book_id"].isin(demo_ids)).sum()
    )
    events = reservations.loc[
        reservations["is_synthetic"] & reservations["book_id"].isin(demo_ids)
    ].copy()
    if events.empty:
        raise InsufficientDataError(
            "No se encontraron reservas demostrativas asociadas a libros demostrativos."
        )

    demo_catalog = books.loc[books["is_demo"]].copy()
    panel = _build_panel(events, demo_catalog)
    event_details = events[["book_id", "created_at"]].merge(
        demo_catalog[["book_id", "category"]],
        on="book_id",
        how="left",
        validate="many_to_one",
    )
    features = _build_features(panel)
    activity = (
        events.groupby("book_id")
        .agg(
            total_reservations=("reservation_id", "nunique"),
            active_weeks=("created_at", lambda values: _week_start(values).nunique()),
        )
        .reset_index()
    )
    all_books = books[
        ["book_id", "category", "stock", "reserved", "is_demo"]
    ].merge(
        activity, on="book_id", how="left", validate="one_to_one"
    )
    all_books[["total_reservations", "active_weeks"]] = all_books[
        ["total_reservations", "active_weeks"]
    ].fillna(0).astype(int)
    all_books["total_inventory"] = (
        all_books["stock"] + all_books["reserved"]
    )
    all_books["sufficient_history"] = (
        all_books["is_demo"]
        & (all_books["total_reservations"] >= MIN_BOOK_RESERVATIONS)
        & (all_books["active_weeks"] >= MIN_BOOK_ACTIVE_WEEKS)
        & (len(panel["period_start"].unique()) >= MIN_HISTORY_WEEKS)
    )
    eligible_ids = set(
        all_books.loc[all_books["sufficient_history"], "book_id"]
    )
    features = features.loc[features["book_id"].isin(eligible_ids)].copy()
    if features.empty:
        raise InsufficientDataError(
            "No hay libros con historial suficiente para construir un conjunto de entrenamiento."
        )

    report = {
        "data_type": "sintetico_demostrativo",
        "frequency": "semanal",
        "frequency_reason": (
            f"{len(events)} solicitudes demostrativas en "
            f"{panel['period_start'].nunique()} semanas para "
            f"{len(demo_catalog)} libros; se usa frecuencia semanal para reducir la "
            "dispersión diaria. El horizonte de 30 días equivale a cuatro semanas "
            "completas y dos días de una quinta semana."
        ),
        "mean_events_per_week": round(
            len(events) / panel["period_start"].nunique(), 2
        ),
        "source_reservations": original_rows,
        "valid_demo_reservations": int(len(events)),
        "duplicate_reservations_removed": duplicate_count,
        "invalid_id_reservations_removed": invalid_id_count,
        "invalid_date_reservations_removed": invalid_date_count,
        "invalid_state_reservations_removed": invalid_state_count,
        "unknown_book_reservations_removed": unknown_book_count,
        "non_synthetic_reservations_excluded": non_demo_reservations,
        "demo_reservations_linked_to_non_demo_books_excluded": non_demo_book_events,
        "books_total": int(len(books)),
        "demo_books": int(len(demo_catalog)),
        "non_demo_books_excluded_from_training": int((~books["is_demo"]).sum()),
        "eligible_demo_books": int(len(eligible_ids)),
        "insufficient_history_demo_books": int(
            len(demo_catalog)
            - all_books.loc[all_books["is_demo"], "sufficient_history"].sum()
        ),
        "first_period": panel["period_start"].min().date().isoformat(),
        "last_period": panel["period_start"].max().date().isoformat(),
        "periods": int(panel["period_start"].nunique()),
        "nonzero_book_periods": int((panel["demand"] > 0).sum()),
        "zero_book_periods": int((panel["demand"] == 0).sum()),
        "predictors": list(FEATURE_COLUMNS),
        "lag_strategy": "lags 1, 2, 4 semanas; rolling means de 4 y 8 semanas.",
        "leakage_control": (
            "Todas las variables de rezago y ventanas se desplazan antes de calcularse; "
            "la división train/test es cronológica por periodo."
        ),
    }
    return PreparedData(panel, features, all_books, event_details, report)
