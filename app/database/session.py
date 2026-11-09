from functools import lru_cache
import os
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(Exception):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    database_url = os.getenv("DB_URL", "").strip()
    if not database_url:
        raise DatabaseConfigurationError("DB_URL no está configurada.")

    try:
        url = make_url(database_url)
    except ArgumentError:
        raise DatabaseConfigurationError("DB_URL no tiene un formato válido.") from None

    if url.drivername not in {
        "postgresql",
        "postgresql+psycopg",
        "postgresql+psycopg2",
    }:
        raise DatabaseConfigurationError("DB_URL debe utilizar PostgreSQL con psycopg2.")

    url = url.set(drivername="postgresql+psycopg2")
    return create_engine(
        url,
        connect_args={"options": "-c default_transaction_read_only=on"},
        echo=False,
        hide_parameters=True,
        pool_pre_ping=True,
    )


def get_db() -> Generator[Session, None, None]:
    session: Session | None = None
    try:
        session = SessionLocal(bind=get_engine())
        session.execute(text("SET TRANSACTION READ ONLY"))
        yield session
    except DatabaseConfigurationError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "La conexión a la base de datos no está configurada correctamente.",
            },
        ) from None
    except SQLAlchemyError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "message": "No se pudo conectar o consultar la base de datos.",
            },
        ) from None
    finally:
        if session is not None:
            session.close()


def dispose_engine() -> None:
    if get_engine.cache_info().currsize:
        get_engine().dispose()
        get_engine.cache_clear()
