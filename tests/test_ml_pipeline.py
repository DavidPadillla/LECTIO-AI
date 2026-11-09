import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

from app.api import routes
from app.services.data_preparation import prepare_data
from app.services.exploratory_analysis import build_analytics
from app.services.prediction import ModelNotTrainedError, forecast_all
from app.services.recommendation import build_recommendations
from app.services.training import train_and_save


def build_fixture():
    books = [
        {
            "book_id": "demo-a",
            "category": "Historia",
            "stock": 5,
            "reserved": 2,
            "is_demo": True,
            "description": "[DEMO-LECTIO] Private fixture must not persist",
        },
        {
            "book_id": "demo-b",
            "category": "Ciencia",
            "stock": 1,
            "reserved": 1,
            "is_demo": True,
        },
        {
            "book_id": "real-c",
            "category": "Arte",
            "stock": 3,
            "reserved": 0,
            "is_demo": False,
        },
    ]
    events = []
    periods = pd.date_range("2025-01-06", periods=32, freq="W-MON")
    counter = 0
    for week_index, period in enumerate(periods):
        events.append(
            {
                "reservation_id": f"a-{counter}",
                "book_id": "demo-a",
                "created_at": period + pd.Timedelta(days=1),
                "state": ("ACTIVA", "DEVUELTA", "CANCELADA")[week_index % 3],
                "is_synthetic": True,
                "correo": "private@example.test",
            }
        )
        counter += 1
        if week_index % 2 == 0:
            events.append(
                {
                    "reservation_id": f"b-{counter}",
                    "book_id": "demo-b",
                    "created_at": period + pd.Timedelta(days=2),
                    "state": "DEVUELTA",
                    "is_synthetic": True,
                }
            )
            counter += 1

    events.extend(
        [
            {**events[0], "state": "CANCELADA"},
            {
                "reservation_id": "real-event",
                "book_id": "real-c",
                "created_at": periods[-1],
                "state": "ACTIVA",
                "is_synthetic": False,
            },
            {
                "reservation_id": "invalid-date",
                "book_id": "demo-a",
                "created_at": "not-a-date",
                "state": "ACTIVA",
                "is_synthetic": True,
            },
            {
                "reservation_id": "invalid-state",
                "book_id": "demo-a",
                "created_at": periods[-1],
                "state": "PENDING",
                "is_synthetic": True,
            },
        ]
    )
    return books, events


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        books, events = build_fixture()
        self.prepared = prepare_data(books, events)

    def test_duplicates_invalid_rows_and_non_demo_are_excluded(self):
        report = self.prepared.report
        self.assertEqual(report["duplicate_reservations_removed"], 1)
        self.assertEqual(report["invalid_date_reservations_removed"], 1)
        self.assertEqual(report["invalid_state_reservations_removed"], 1)
        self.assertEqual(report["non_synthetic_reservations_excluded"], 1)
        self.assertEqual(report["valid_demo_reservations"], 48)

    def test_all_statuses_count_once_and_zero_weeks_are_filled(self):
        self.assertEqual(int(self.prepared.panel["demand"].sum()), 48)
        self.assertEqual(self.prepared.report["periods"], 32)
        self.assertGreater(self.prepared.report["zero_book_periods"], 0)
        self.assertEqual(set(self.prepared.panel["book_id"]), {"demo-a", "demo-b"})
        weekday_total = sum(
            item["demand"]
            for item in build_analytics(
                self.prepared.panel,
                self.prepared.catalog,
                self.prepared.events,
            )["weekday_demand"]
        )
        self.assertEqual(weekday_total, 48)

    def test_features_use_only_previous_periods_and_contain_no_personal_fields(self):
        row = self.prepared.features.loc[
            (self.prepared.features["book_id"] == "demo-a")
            & (self.prepared.features["period_start"] == pd.Timestamp("2025-03-17"))
        ].iloc[0]
        previous = self.prepared.panel.loc[
            (self.prepared.panel["book_id"] == "demo-a")
            & (self.prepared.panel["period_start"] == pd.Timestamp("2025-03-10")),
            "demand",
        ].iloc[0]
        self.assertEqual(row["lag_1"], previous)
        self.assertNotIn("correo", self.prepared.panel.columns)
        self.assertNotIn("description", self.prepared.panel.columns)
        self.assertNotIn("username", self.prepared.panel.columns)
        self.assertNotIn("correo", self.prepared.events.columns)
        self.assertNotIn("reservation_id", self.prepared.events.columns)


class TrainingAndRecommendationTests(unittest.TestCase):
    def test_train_evaluate_forecast_and_recommend_without_negative_demand(self):
        books, events = build_fixture()
        prepared = prepare_data(books, events)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, evaluation = train_and_save(
                prepared,
                root / "models",
                root / "processed",
                root / "reports",
            )
            self.assertLess(evaluation["train_end"], evaluation["test_start"])
            self.assertEqual(evaluation["test_samples"] > 0, True)
            self.assertIn("mae", evaluation["mlp_regressor"])
            self.assertIn("rmse", evaluation["mlp_regressor"])
            exported = pd.read_csv(root / "processed" / "weekly_demand.csv")
            self.assertNotIn("correo", exported.columns)
            self.assertNotIn("username", exported.columns)

            forecasts = forecast_all(bundle)
            self.assertEqual(forecasts["horizon_days"], 30)
            self.assertTrue(
                all(
                    week["predicted_demand"] >= 0
                    for book in forecasts["books"]
                    for week in book["weekly_forecast"]
                )
            )
            recommendations = build_recommendations(
                forecasts, bundle["book_history"]
            )
            self.assertTrue(
                all(item["recommended_replenishment"] >= 0
                    for item in recommendations["recommendations"])
            )
            for item in recommendations["recommendations"]:
                self.assertEqual(
                    item["total_inventory"],
                    item["stock_available"] + item["reserved_units"],
                )

    def test_recommendation_risk_and_replenishment_thresholds(self):
        forecasts = {
            "data_type": "sintetico_demostrativo",
            "horizon_days": 30,
            "books": [
                {"book_id": "high", "category": "A", "historical_demand": 20,
                 "forecast_demand_30_days": 10.0},
                {"book_id": "medium", "category": "A", "historical_demand": 20,
                 "forecast_demand_30_days": 10.0},
                {"book_id": "low", "category": "A", "historical_demand": 20,
                 "forecast_demand_30_days": 10.0},
                {"book_id": "insufficient", "category": "A", "historical_demand": 0,
                 "forecast_demand_30_days": None},
            ],
        }
        catalog = [
            {"book_id": "high", "stock": 2, "reserved": 3, "total_inventory": 5},
            {"book_id": "medium", "stock": 11, "reserved": 2, "total_inventory": 13},
            {"book_id": "low", "stock": 20, "reserved": 0, "total_inventory": 20},
            {"book_id": "insufficient", "stock": 0, "reserved": 0, "total_inventory": 0},
        ]
        result = build_recommendations(forecasts, catalog)
        by_book = {item["book_id"]: item for item in result["recommendations"]}
        self.assertEqual(by_book["high"]["risk_level"], "ALTO")
        self.assertEqual(by_book["high"]["recommended_replenishment"], 10)
        self.assertEqual(by_book["medium"]["risk_level"], "MEDIO")
        self.assertEqual(by_book["medium"]["recommended_replenishment"], 1)
        self.assertEqual(by_book["low"]["risk_level"], "BAJO")
        self.assertEqual(by_book["low"]["recommended_replenishment"], 0)
        self.assertEqual(by_book["insufficient"]["risk_level"], "DATOS_INSUFICIENTES")
        self.assertEqual(by_book["insufficient"]["recommended_replenishment"], 0)


class ApiFallbackTests(unittest.TestCase):
    def test_model_not_trained_status_and_prediction_error(self):
        with patch.object(
            routes, "get_model_bundle", side_effect=ModelNotTrainedError()
        ):
            status = routes.read_model_status()
            self.assertFalse(status["trained"])
            self.assertFalse(status["metrics_available"])
            with self.assertRaises(HTTPException) as raised:
                routes.read_predictions()
            self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
