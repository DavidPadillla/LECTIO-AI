from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import router
from app.database import dispose_engine
from app.services.prediction import ensure_model_available
from app.services.weekly_update import WeeklyModelUpdater, preload_current_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    updater = WeeklyModelUpdater()
    app.state.weekly_updater = updater
    try:
        ensure_model_available()
        preload_current_model()
        try:
            updater.start()
        except Exception as error:
            logger.error(
                "Weekly model scheduler initialization failed (%s).",
                type(error).__name__,
            )
        yield
    finally:
        updater.stop()
        dispose_engine()


app = FastAPI(
    title="LECTIO-AI",
    description="Sistema predictivo de demanda de libros para una biblioteca.",
    lifespan=lifespan,
)
app.include_router(router)
