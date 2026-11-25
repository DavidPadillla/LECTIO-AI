from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.services.weekly_update import WeeklyModelUpdater

logger = logging.getLogger(__name__)


class ManualUpdateJobs:
    """Run a model refresh in the background and expose its current progress."""

    def __init__(self, updater: WeeklyModelUpdater) -> None:
        self.updater = updater
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Ya hay una actualización manual en curso.")

            job_id = str(uuid4())
            self._job = {
                "job_id": job_id,
                "state": "RUNNING",
                "stage": "iniciando",
                "message": "Preparando la actualización del modelo.",
                "started_at": self._now(),
                "updated_at": self._now(),
                "result": None,
            }
            self._thread = threading.Thread(
                target=self._run,
                args=(job_id,),
                name=f"lectio-manual-update-{job_id[:8]}",
                daemon=True,
            )
            self._thread.start()
            return dict(self._job)

    def _set_progress(self, job_id: str, stage: str, message: str) -> None:
        with self._lock:
            if self._job is None or self._job["job_id"] != job_id:
                return
            self._job.update(
                {"stage": stage, "message": message, "updated_at": self._now()}
            )

    def _run(self, job_id: str) -> None:
        try:
            result = self.updater.run_update(
                progress_callback=lambda stage, message: self._set_progress(
                    job_id, stage, message
                )
            )
            state = "COMPLETED"
            if result.get("status") == "FALLIDA":
                state = "FAILED"
            message = {
                "EXITOSA": "El nuevo modelo pasó la evaluación y quedó activo.",
                "MODELO_ANTERIOR_CONSERVADO": (
                    "La actualización terminó, pero se conservó el modelo anterior."
                ),
                "FALLIDA": "No se pudo actualizar el modelo.",
            }.get(result.get("status"), "La actualización terminó.")
            stage = (
                "terminado"
                if result.get("status") == "EXITOSA"
                else "evaluacion_finalizada"
                if state == "COMPLETED"
                else "error"
            )
            with self._lock:
                if self._job is not None and self._job["job_id"] == job_id:
                    self._job.update(
                        {
                            "state": state,
                            "stage": stage,
                            "message": message,
                            "updated_at": self._now(),
                            "result": result,
                        }
                    )
        except Exception as error:
            logger.error(
                "Manual model update job failed (%s).",
                type(error).__name__,
            )
            with self._lock:
                if self._job is not None and self._job["job_id"] == job_id:
                    self._job.update(
                        {
                            "state": "FAILED",
                            "stage": "error",
                            "message": "Ocurrió un error inesperado al actualizar el modelo.",
                            "updated_at": self._now(),
                        }
                    )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            if self._job is None or self._job["job_id"] != job_id:
                return None
            return dict(self._job)
