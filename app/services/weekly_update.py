from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import joblib
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.services.pipeline import DATA_DIR, run_pipeline
from app.services.prediction import (
    ModelNotTrainedError,
    clear_model_cache,
    forecast_all,
    get_model_bundle,
)
from app.services.process_lock import ProcessFileLock
from app.services.recommendation import build_recommendations
from app.services.training import MODEL_FILENAME

logger = logging.getLogger(__name__)
HISTORY_FILENAME = "model_update_history.json"
VALID_STATUSES = {"EXITOSA", "FALLIDA", "MODELO_ANTERIOR_CONSERVADO"}
WEEKDAYS = {
    "monday": "mon",
    "mon": "mon",
    "lunes": "mon",
    "tuesday": "tue",
    "tue": "tue",
    "martes": "tue",
    "wednesday": "wed",
    "wed": "wed",
    "miercoles": "wed",
    "miércoles": "wed",
    "thursday": "thu",
    "thu": "thu",
    "jueves": "thu",
    "friday": "fri",
    "fri": "fri",
    "viernes": "fri",
    "saturday": "sat",
    "sat": "sat",
    "sabado": "sat",
    "sábado": "sat",
    "sunday": "sun",
    "sun": "sun",
    "domingo": "sun",
}


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool = True
    day_of_week: str = "mon"
    hour: int = 2
    minute: int = 0
    timezone: str = "America/Bogota"
    min_baseline_improvement: float = 0.0
    max_incumbent_regression: float = 0.0

    @classmethod
    def from_env(cls) -> SchedulerConfig:
        day_input = os.getenv("LECTIO_UPDATE_DAY", "monday").strip().lower()
        day = WEEKDAYS.get(day_input)
        if day is None:
            raise ValueError("LECTIO_UPDATE_DAY debe ser un día válido.")

        try:
            hour = int(os.getenv("LECTIO_UPDATE_HOUR", "2"))
            minute = int(os.getenv("LECTIO_UPDATE_MINUTE", "0"))
            min_improvement = float(
                os.getenv("LECTIO_UPDATE_MIN_BASELINE_IMPROVEMENT", "0")
            )
            max_regression = float(
                os.getenv("LECTIO_UPDATE_MAX_INCUMBENT_REGRESSION", "0")
            )
        except ValueError:
            raise ValueError("La configuración de actualización debe ser numérica.") from None
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("La hora de actualización no es válida.")
        if not math.isfinite(min_improvement) or not 0 <= min_improvement < 1:
            raise ValueError(
                "LECTIO_UPDATE_MIN_BASELINE_IMPROVEMENT debe estar entre 0 y 1."
            )
        if not math.isfinite(max_regression) or not 0 <= max_regression <= 1:
            raise ValueError(
                "LECTIO_UPDATE_MAX_INCUMBENT_REGRESSION debe estar entre 0 y 1."
            )
        timezone = os.getenv(
            "LECTIO_UPDATE_TIMEZONE", "America/Bogota"
        ).strip()
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("LECTIO_UPDATE_TIMEZONE no es una zona horaria válida.") from None
        enabled = os.getenv("LECTIO_UPDATE_ENABLED", "true").strip().lower()
        if enabled not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("LECTIO_UPDATE_ENABLED debe ser un valor booleano.")
        return cls(
            enabled=enabled in {"true", "1", "yes"},
            day_of_week=day,
            hour=hour,
            minute=minute,
            timezone=timezone,
            min_baseline_improvement=min_improvement,
            max_incumbent_regression=max_regression,
        )


class WeeklyModelUpdater:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        config: SchedulerConfig | None = None,
        pipeline_runner: Callable[..., dict[str, Any]] = run_pipeline,
    ) -> None:
        self.data_dir = data_dir
        self.config = config or SchedulerConfig.from_env()
        self.pipeline_runner = pipeline_runner
        self._scheduler: BackgroundScheduler | None = None
        self._scheduler_lock: ProcessFileLock | None = None
        self._update_lock = ProcessFileLock(data_dir / "state" / "update.lock")
        self._thread_lock = threading.Lock()
        self._scheduler_active = False

    @property
    def history_path(self) -> Path:
        return self.data_dir / "reports" / HISTORY_FILENAME

    def start(self) -> bool:
        with self._thread_lock:
            if self._scheduler_active:
                return True
            if not self.config.enabled:
                return False
            scheduler_lock = ProcessFileLock(self.data_dir / "state" / "scheduler.lock")
            if not scheduler_lock.acquire():
                logger.info("Weekly model scheduler is already owned by another process.")
                return False
            scheduler: BackgroundScheduler | None = None
            try:
                scheduler = BackgroundScheduler(
                    timezone=ZoneInfo(self.config.timezone)
                )
                trigger = CronTrigger(
                    day_of_week=self.config.day_of_week,
                    hour=self.config.hour,
                    minute=self.config.minute,
                    timezone=ZoneInfo(self.config.timezone),
                )
                scheduler.add_job(
                    self.run_update,
                    trigger=trigger,
                    id="weekly-model-update",
                    replace_existing=True,
                    max_instances=1,
                    coalesce=True,
                    misfire_grace_time=6 * 60 * 60,
                )
                scheduler.start()
            except Exception as error:
                if scheduler is not None and scheduler.running:
                    scheduler.shutdown(wait=False)
                scheduler_lock.release()
                logger.error(
                    "Weekly model scheduler could not start (%s).",
                    type(error).__name__,
                )
                return False
            self._scheduler = scheduler
            self._scheduler_lock = scheduler_lock
            self._scheduler_active = True
            return True

    def stop(self) -> None:
        with self._thread_lock:
            scheduler = self._scheduler
            lock = self._scheduler_lock
            self._scheduler = None
            self._scheduler_lock = None
            self._scheduler_active = False
        if scheduler is not None and scheduler.running:
            scheduler.shutdown(wait=True)
        if lock is not None:
            lock.release()

    def _read_history(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.history_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, json.JSONDecodeError):
            logger.error("Weekly model update history could not be read.")
            return []
        runs = value.get("runs", []) if isinstance(value, dict) else []
        return [run for run in runs if isinstance(run, dict)]

    def _write_history(self, runs: list[dict[str, Any]]) -> None:
        path = self.history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump({"runs": runs}, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_promote(staged: Path, destination: Path) -> None:
        if not staged.is_file():
            raise FileNotFoundError("A staged update artifact is missing.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, destination)

    @staticmethod
    def _acceptance(
        evaluation: dict[str, Any],
        config: SchedulerConfig,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        candidate = evaluation["mlp_regressor"]
        baseline = evaluation["baseline_rolling_4_mean"]
        required_factor = 1 - config.min_baseline_improvement
        baseline_passed = (
            candidate["mae"] < baseline["mae"] * required_factor
            and candidate["rmse"] < baseline["rmse"] * required_factor
        )
        comparison: dict[str, Any] = {
            "candidate_mae": candidate["mae"],
            "candidate_rmse": candidate["rmse"],
            "baseline_mae": baseline["mae"],
            "baseline_rmse": baseline["rmse"],
            "baseline_passed": baseline_passed,
            "minimum_baseline_improvement": config.min_baseline_improvement,
            "incumbent_mae": None,
            "incumbent_rmse": None,
            "incumbent_passed": None,
            "maximum_incumbent_regression": config.max_incumbent_regression,
        }
        if not baseline_passed:
            return False, "El candidato no superó MAE y RMSE del baseline.", comparison
        incumbent = evaluation.get("incumbent_model")
        if incumbent is not None:
            tolerance_factor = 1 + config.max_incumbent_regression
            incumbent_passed = (
                candidate["mae"] <= incumbent["mae"] * tolerance_factor
                and candidate["rmse"] <= incumbent["rmse"] * tolerance_factor
            )
            comparison.update(
                {
                    "incumbent_mae": incumbent["mae"],
                    "incumbent_rmse": incumbent["rmse"],
                    "incumbent_passed": incumbent_passed,
                }
            )
            if not incumbent_passed:
                return (
                    False,
                    "El candidato excedió la regresión permitida frente al modelo actual.",
                    comparison,
                )
        return True, None, comparison

    def _new_run_record(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(ZoneInfo(self.config.timezone)).isoformat(),
            "status": "FALLIDA",
            "reservations_queried": None,
            "books_analyzed": None,
            "records_used": None,
            "metrics": None,
            "comparison": None,
            "previous_model_preserved": False,
            "reason": None,
        }

    def run_update(self) -> dict[str, Any]:
        if not self._update_lock.acquire():
            return {
                "status": "MODELO_ANTERIOR_CONSERVADO",
                "reason": "Ya hay otra actualización en curso.",
            }

        record = self._new_run_record()
        had_valid_model = False
        staging_dir: Path | None = None
        try:
            try:
                incumbent_bundle = get_model_bundle()
                had_valid_model = True
            except ModelNotTrainedError:
                incumbent_bundle = None

            self.data_dir.mkdir(parents=True, exist_ok=True)
            staging_dir = Path(
                tempfile.mkdtemp(prefix=".weekly-update-", dir=self.data_dir)
            )
            result = self.pipeline_runner(
                output_root=staging_dir,
                incumbent_bundle=incumbent_bundle,
            )
            preparation = result["preparation"]
            evaluation = result["evaluation"]
            record.update(
                {
                    "reservations_queried": preparation["source_reservations"],
                    "books_analyzed": preparation["demo_books"],
                    "records_used": evaluation["train_samples"],
                    "metrics": {
                        "mae": evaluation["mlp_regressor"]["mae"],
                        "rmse": evaluation["mlp_regressor"]["rmse"],
                    },
                }
            )
            accepted, rejection_reason, comparison = self._acceptance(
                evaluation,
                self.config,
            )
            record["comparison"] = comparison
            if not accepted:
                record["status"] = (
                    "MODELO_ANTERIOR_CONSERVADO"
                    if had_valid_model
                    else "FALLIDA"
                )
                record["previous_model_preserved"] = had_valid_model
                record["reason"] = rejection_reason
                return record

            staged_model = staging_dir / "models" / MODEL_FILENAME
            candidate_bundle = joblib.load(staged_model)
            predictions = forecast_all(candidate_bundle)
            recommendations = build_recommendations(
                predictions,
                candidate_bundle["book_history"],
            )
            reports_dir = staging_dir / "reports"
            for name, value in (
                ("predictions.json", predictions),
                ("recommendations.json", recommendations),
            ):
                (reports_dir / name).write_text(
                    json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            for subdir in ("processed", "reports"):
                staged_subdir = staging_dir / subdir
                if staged_subdir.exists():
                    destination_dir = self.data_dir / subdir
                    destination_dir.mkdir(parents=True, exist_ok=True)
                    for artifact in staged_subdir.iterdir():
                        self._atomic_promote(artifact, destination_dir / artifact.name)
            self._atomic_promote(
                staged_model,
                self.data_dir / "models" / MODEL_FILENAME,
            )
            clear_model_cache()
            record["status"] = "EXITOSA"
            record["reason"] = None
            return record
        except Exception as error:
            logger.error(
                "Weekly model update failed (%s).",
                type(error).__name__,
            )
            record["status"] = (
                "MODELO_ANTERIOR_CONSERVADO"
                if had_valid_model
                else "FALLIDA"
            )
            record["previous_model_preserved"] = had_valid_model
            record["reason"] = (
                "La actualización falló; se conservó el modelo válido anterior."
                if had_valid_model
                else "La actualización falló y no hay un modelo válido anterior."
            )
            return record
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            try:
                try:
                    runs = self._read_history()
                    runs.append(record)
                    self._write_history(runs)
                except OSError as error:
                    logger.error(
                        "Weekly model update history could not be written (%s).",
                        type(error).__name__,
                    )
            finally:
                self._update_lock.release()

    def get_status(self) -> dict[str, Any]:
        runs = self._read_history()
        last_run = runs[-1] if runs else None
        next_run = None
        if self._scheduler_active and self._scheduler is not None:
            job = self._scheduler.get_job("weekly-model-update")
            if job is not None and job.next_run_time is not None:
                next_run = job.next_run_time.isoformat()
        return {
            "scheduler_active": self._scheduler_active,
            "timezone": self.config.timezone,
            "schedule": {
                "day_of_week": self.config.day_of_week,
                "hour": self.config.hour,
                "minute": self.config.minute,
            },
            "last_execution_at": last_run.get("timestamp") if last_run else None,
            "next_execution_at": next_run,
            "last_result": last_run,
            "last_training_metrics": (
                last_run.get("metrics") if last_run else None
            ),
            "previous_model_preserved": (
                bool(last_run.get("previous_model_preserved")) if last_run else False
            ),
            "successful_runs": sum(run.get("status") == "EXITOSA" for run in runs),
            "failed_runs": sum(
                run.get("status") in {"FALLIDA", "MODELO_ANTERIOR_CONSERVADO"}
                for run in runs
            ),
        }


def preload_current_model() -> bool:
    try:
        get_model_bundle()
        return True
    except ModelNotTrainedError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar una actualización manual de LECTIO-AI.")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Ejecuta una actualización manual sin esperar al próximo horario.",
    )
    arguments = parser.parse_args()
    if not arguments.run_now:
        parser.error("Usa --run-now para iniciar manualmente una actualización.")
    result = WeeklyModelUpdater().run_update()
    print(json.dumps(result, ensure_ascii=False, indent=2))
