from __future__ import annotations

import math
import os
from typing import Any


def _safety_margin() -> float:
    try:
        value = float(os.getenv("LECTIO_SAFETY_MARGIN", "0.20"))
    except ValueError:
        raise ValueError("LECTIO_SAFETY_MARGIN debe ser numérico.") from None
    if not 0 <= value <= 2:
        raise ValueError("LECTIO_SAFETY_MARGIN debe estar entre 0 y 2.")
    return value


def build_recommendations(
    forecasts: dict[str, Any],
    book_history: list[dict[str, Any]],
) -> dict[str, Any]:
    margin = _safety_margin()
    inventory = {str(book["book_id"]): book for book in book_history}
    recommendations = []
    for forecast in forecasts["books"]:
        book = inventory[forecast["book_id"]]
        predicted = forecast["forecast_demand_30_days"]
        stock = int(book["stock"])
        reserved = int(book["reserved"])
        total_inventory = int(book["total_inventory"])

        if predicted is None:
            risk = "DATOS_INSUFICIENTES"
            reorder = 0
        else:
            target_stock = predicted * (1 + margin)
            reorder = max(0, math.ceil(target_stock - stock))
            if predicted > stock:
                risk = "ALTO"
            elif predicted > 0 and target_stock > stock:
                risk = "MEDIO"
            else:
                risk = "BAJO"

        recommendations.append(
            {
                "book_id": forecast["book_id"],
                "category": forecast["category"],
                "data_type": forecasts["data_type"],
                "historical_demand": forecast["historical_demand"],
                "forecast_demand_30_days": predicted,
                "stock_available": stock,
                "reserved_units": reserved,
                "total_inventory": total_inventory,
                "risk_level": risk,
                "recommended_replenishment": reorder,
                "safety_margin": margin,
                "formula": (
                    "max(0, ceil(demanda_pronosticada_30_dias "
                    "* (1 + margen_seguridad) - stock_disponible))"
                ),
            }
        )
    return {
        "data_type": forecasts["data_type"],
        "horizon_days": forecasts["horizon_days"],
        "safety_margin": margin,
        "recommendations": recommendations,
    }
