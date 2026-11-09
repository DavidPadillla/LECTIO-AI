import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from starlette.requests import Request

from app.api.routes import read_model_update_status
from app.services import prediction
from app.services.prediction import MODEL_PATH, clear_model_cache, get_model_bundle
from app.services.process_lock import ProcessFileLock
from app.services.weekly_update import (
    SchedulerConfig,
    WeeklyModelUpdater,
)
from app.services.training import MODEL_FILENAME


class ConstantPipeline:
    def __init__(self, value=0.5):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


def build_bundle(prediction=0.5):
    dates = pd.date_range("2025-01-06", periods=12, freq="W-MON")
    history = [
        {"book_id": "demo-book", "period_start": date, "demand": 1, "category": "Novelas"}
        for date in dates
    ]
    book_history = [
        {
            "book_id": "demo-book",
            "category": "Novelas",
            "stock": 2,
            "reserved": 1,
            "total_inventory": 3,
            "total_reservations": 12,
            "active_weeks": 12,
            "sufficient_history": True,
            "is_demo": True,
        }
    ]
    return {
        "pipeline": ConstantPipeline(prediction),
        "data_type": "sintetico_demostrativo",
        "history": history,
        "book_history": book_history,
        "training_samples": 20,
        "trained_at": "2026-01-01T00:00:00+00:00",
        "model_config": {"type": "MLPRegressor"},
    }


def build_pipeline_result(output_root, candidate=None, metrics=None):
    output_root = Path(output_root)
    (output_root / "models").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "processed").mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate or build_bundle(), output_root / "models" / MODEL_FILENAME)
    (output_root / "reports" / "analytics.json").write_text(
        json.dumps({"data_type": "sintetico_demostrativo"}),
        encoding="utf-8",
    )
    (output_root / "processed" / "weekly_demand.csv").write_text(
        "book_id,demand\ndemo-book,1\n",
        encoding="utf-8",
    )
    evaluation = metrics or {
        "train_samples": 20,
        "mlp_regressor": {"mae": 0.4, "rmse": 0.5},
        "baseline_rolling_4_mean": {"mae": 0.6, "rmse": 0.7},
        "incumbent_model": {"mae": 0.45, "rmse": 0.55},
    }
    return {
        "preparation": {
            "source_reservations": 32,
            "demo_books": 1,
        },
        "evaluation": evaluation,
    }


class SchedulerConfigurationTests(unittest.TestCase):
    def test_default_schedule_and_timezone(self):
        with patch.dict(os.environ, {}, clear=True):
            config = SchedulerConfig.from_env()
        self.assertEqual(config.day_of_week, "mon")
        self.assertEqual(config.hour, 2)
        self.assertEqual(config.minute, 0)
        self.assertEqual(config.timezone, "America/Bogota")
        self.assertEqual(ZoneInfo(config.timezone).key, "America/Bogota")

    def test_schedule_can_be_configured_and_rejects_invalid_values(self):
        with patch.dict(
            os.environ,
            {
                "LECTIO_UPDATE_DAY": "viernes",
                "LECTIO_UPDATE_HOUR": "3",
                "LECTIO_UPDATE_MINUTE": "15",
                "LECTIO_UPDATE_TIMEZONE": "America/Bogota",
                "LECTIO_UPDATE_MIN_BASELINE_IMPROVEMENT": "0.05",
                "LECTIO_UPDATE_MAX_INCUMBENT_REGRESSION": "0.01",
            },
            clear=True,
        ):
            config = SchedulerConfig.from_env()
        self.assertEqual((config.day_of_week, config.hour, config.minute), ("fri", 3, 15))
        self.assertEqual(config.min_baseline_improvement, 0.05)
        self.assertEqual(config.max_incumbent_regression, 0.01)
        with patch.dict(os.environ, {"LECTIO_UPDATE_HOUR": "30"}, clear=True):
            with self.assertRaises(ValueError):
                SchedulerConfig.from_env()

    def test_scheduler_starts_once_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as temporary:
            updater = WeeklyModelUpdater(data_dir=Path(temporary))
            second = WeeklyModelUpdater(data_dir=Path(temporary))
            self.assertTrue(updater.start())
            self.assertTrue(updater._scheduler.running)
            job = updater._scheduler.get_job("weekly-model-update")
            self.assertEqual(job.trigger.timezone.key, "America/Bogota")
            self.assertFalse(second.start())
            updater.stop()
            self.assertFalse(updater._scheduler_active)
            self.assertTrue(second.start())
            second.stop()

    def test_disabled_scheduler_does_not_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            updater = WeeklyModelUpdater(
                data_dir=Path(temporary),
                config=SchedulerConfig(enabled=False),
            )
            self.assertFalse(updater.start())
            self.assertFalse(updater._scheduler_active)


class WeeklyUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        self.old_bundle = build_bundle(0.7)
        self.model_file = self.data_dir / "models" / MODEL_FILENAME
        self.model_file.parent.mkdir(parents=True)
        self.model_file.write_bytes(b"previous-valid-model")
        self.config = SchedulerConfig()

    def tearDown(self):
        clear_model_cache()
        self.temporary.cleanup()

    def make_updater(self, runner):
        return WeeklyModelUpdater(
            data_dir=self.data_dir,
            config=self.config,
            pipeline_runner=runner,
        )

    def test_successful_update_is_atomic_and_writes_history_and_predictions(self):
        def runner(output_root, incumbent_bundle):
            self.assertIs(incumbent_bundle, self.old_bundle)
            return build_pipeline_result(output_root)

        updater = self.make_updater(runner)
        self.assertFalse(updater._scheduler_active)
        with patch("app.services.weekly_update.get_model_bundle", return_value=self.old_bundle), patch(
            "app.services.weekly_update.clear_model_cache"
        ):
            result = updater.run_update()
        self.assertFalse(updater._scheduler_active)
        self.assertEqual(result["status"], "EXITOSA")
        self.assertFalse(result["previous_model_preserved"])
        self.assertEqual(result["reservations_queried"], 32)
        self.assertEqual(result["books_analyzed"], 1)
        self.assertEqual(result["records_used"], 20)
        self.assertNotEqual(self.model_file.read_bytes(), b"previous-valid-model")
        reports = self.data_dir / "reports"
        self.assertTrue((reports / "predictions.json").is_file())
        self.assertTrue((reports / "recommendations.json").is_file())
        history = json.loads((reports / "model_update_history.json").read_text())
        self.assertEqual(history["runs"][-1]["status"], "EXITOSA")
        status = updater.get_status()
        self.assertEqual(status["successful_runs"], 1)
        self.assertEqual(status["failed_runs"], 0)
        app = FastAPI()
        app.state.weekly_updater = updater
        request = Request({"type": "http", "app": app})
        self.assertEqual(read_model_update_status(request)["last_result"]["status"], "EXITOSA")

    def test_existing_model_serves_predictions_during_and_after_atomic_swap(self):
        joblib.dump(self.old_bundle, self.model_file)
        entered_training = threading.Event()
        allow_training_to_finish = threading.Event()

        def runner(output_root, **_):
            entered_training.set()
            if not allow_training_to_finish.wait(timeout=10):
                raise TimeoutError("test update release timed out")
            return build_pipeline_result(output_root)

        updater = self.make_updater(runner)
        clear_model_cache()
        result_holder = []
        with patch.object(prediction, "MODEL_PATH", self.model_file), patch(
            "app.services.weekly_update.get_model_bundle",
            return_value=self.old_bundle,
        ):
            self.assertEqual(get_model_bundle()["pipeline"].value, 0.7)
            thread = threading.Thread(target=lambda: result_holder.append(updater.run_update()))
            thread.start()
            self.assertTrue(entered_training.wait(timeout=10))
            self.assertEqual(get_model_bundle()["pipeline"].value, 0.7)
            allow_training_to_finish.set()
            thread.join(timeout=20)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result_holder[0]["status"], "EXITOSA")
            self.assertEqual(get_model_bundle()["pipeline"].value, 0.5)

    def test_training_failure_preserves_incumbent_and_records_failure(self):
        def runner(**_):
            raise RuntimeError("error containing a secret URL must not be logged")

        updater = self.make_updater(runner)
        with patch("app.services.weekly_update.get_model_bundle", return_value=self.old_bundle):
            result = updater.run_update()
        self.assertEqual(result["status"], "MODELO_ANTERIOR_CONSERVADO")
        self.assertTrue(result["previous_model_preserved"])
        self.assertEqual(self.model_file.read_bytes(), b"previous-valid-model")
        self.assertTrue(result["reason"])
        history_text = updater.history_path.read_text(encoding="utf-8")
        self.assertNotIn("secret URL", history_text)
        self.assertNotIn("RuntimeError", history_text)

    def test_worse_candidate_is_rejected_and_old_model_remains(self):
        poor_metrics = {
            "train_samples": 20,
            "mlp_regressor": {"mae": 0.8, "rmse": 0.9},
            "baseline_rolling_4_mean": {"mae": 0.6, "rmse": 0.7},
            "incumbent_model": {"mae": 0.7, "rmse": 0.8},
        }

        def runner(output_root, **_):
            return build_pipeline_result(output_root, metrics=poor_metrics)

        updater = self.make_updater(runner)
        with patch("app.services.weekly_update.get_model_bundle", return_value=self.old_bundle):
            result = updater.run_update()
        self.assertEqual(result["status"], "MODELO_ANTERIOR_CONSERVADO")
        self.assertTrue(result["previous_model_preserved"])
        self.assertEqual(self.model_file.read_bytes(), b"previous-valid-model")
        self.assertIn("baseline", result["reason"])

    def test_busy_update_lock_prevents_overlapping_run(self):
        updater = self.make_updater(lambda **_: self.fail("pipeline must not run"))
        held_lock = ProcessFileLock(self.data_dir / "state" / "update.lock")
        self.assertTrue(held_lock.acquire())
        try:
            result = updater.run_update()
        finally:
            held_lock.release()
        self.assertEqual(result["status"], "MODELO_ANTERIOR_CONSERVADO")
        self.assertIn("otra actualización", result["reason"])

    def test_candidate_must_pass_baseline_and_incumbent_on_same_validation(self):
        evaluation = {
            "mlp_regressor": {"mae": 0.4, "rmse": 0.6},
            "baseline_rolling_4_mean": {"mae": 0.5, "rmse": 0.7},
            "incumbent_model": {"mae": 0.3, "rmse": 0.5},
        }
        accepted, reason, comparison = WeeklyModelUpdater._acceptance(
            evaluation,
            self.config,
        )
        self.assertFalse(accepted)
        self.assertIsNotNone(reason)
        self.assertTrue(comparison["baseline_passed"])
        self.assertFalse(comparison["incumbent_passed"])


if __name__ == "__main__":
    unittest.main()
