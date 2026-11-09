from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lectio-biblioteca.onrender.com",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Accept", "Authorization", "Content-Type"],
    allow_credentials=False,
)
app.include_router(router)
