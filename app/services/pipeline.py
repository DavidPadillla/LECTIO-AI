from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.session import get_engine
from app.paths import DATA_DIR
from app.services.data_preparation import load_source_data, prepare_data
from app.services.exploratory_analysis import save_reports
from app.services.training import train_and_save


def run_pipeline(
    output_root: Path = DATA_DIR,
    incumbent_bundle: dict[str, Any] | None = None,
) -> dict:
    engine = get_engine()
    try:
        with Session(engine) as session:
            session.execute(text("SET TRANSACTION READ ONLY"))
            book_records, reservation_records = load_source_data(session)
    except SQLAlchemyError:
        raise RuntimeError(
            "No se pudieron leer los datos de la base de datos en modo de solo lectura."
        ) from None
    finally:
        engine.dispose()
        get_engine.cache_clear()

    prepared = prepare_data(book_records, reservation_records)
    analytics = save_reports(
        prepared.panel,
        prepared.catalog,
        prepared.events,
        output_root / "reports",
    )
    bundle, evaluation = train_and_save(
        prepared,
        output_root / "models",
        output_root / "processed",
        output_root / "reports",
        incumbent_bundle=incumbent_bundle,
    )
    return {
        "data_type": bundle["data_type"],
        "preparation": prepared.report,
        "analytics_summary": {
            "total_demand": analytics["total_demand"],
            "books_analyzed": analytics["books_analyzed"],
        },
        "evaluation": evaluation,
        "trained_at": bundle["trained_at"],
    }


if __name__ == "__main__":
    try:
        print(json.dumps(run_pipeline(), ensure_ascii=False, indent=2))
    except Exception as error:
        print(
            json.dumps(
                {"status": "error", "message": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
