import re
from datetime import date, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


TABLE_NAMES = ("usuarios", "libros_fisicos", "reservas")
REQUIRED_COLUMNS = {
    "usuarios": ("id", "username"),
    "libros_fisicos": ("id", "titulo", "descripcion", "categoria", "stock", "reservado"),
    "reservas": ("id", "usuario_id", "libro_fisico_id", "correo", "estado", "fecha_creacion"),
}
TEXT_DATA_TYPES = {"character", "character varying", "text"}
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatabaseSchemaError(Exception):
    pass


def _quote_identifier(identifier: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(identifier):
        raise DatabaseSchemaError("El esquema contiene un identificador no compatible.")
    return f'"{identifier}"'


def _inspect_tables(session: Session) -> dict[str, dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT table_schema, table_name, column_name,
                   data_type, udt_name, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = ANY (current_schemas(false))
              AND table_name IN ('usuarios', 'libros_fisicos', 'reservas')
            ORDER BY table_schema, table_name, ordinal_position
            """
        )
    ).mappings()

    tables: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["table_name"]
        table = tables.setdefault(
            name,
            {"schema": row["table_schema"], "columns": []},
        )
        if table["schema"] != row["table_schema"]:
            raise DatabaseSchemaError(
                f"Hay más de una tabla visible llamada {name} en el search_path."
            )
        table["columns"].append(
            {
                "name": row["column_name"],
                "type": row["data_type"],
                "postgres_type": row["udt_name"],
            }
        )

    missing_tables = [name for name in TABLE_NAMES if name not in tables]
    if missing_tables:
        raise DatabaseSchemaError("No se encontraron todas las tablas requeridas.")

    for table_name, required in REQUIRED_COLUMNS.items():
        present = {column["name"] for column in tables[table_name]["columns"]}
        if any(column not in present for column in required):
            raise DatabaseSchemaError(
                "Las tablas requeridas no tienen todas las columnas necesarias."
            )

    for table_name, column_name in (
        ("usuarios", "username"),
        ("libros_fisicos", "titulo"),
        ("libros_fisicos", "descripcion"),
        ("reservas", "correo"),
    ):
        column = next(
            item
            for item in tables[table_name]["columns"]
            if item["name"] == column_name
        )
        if column["type"] not in TEXT_DATA_TYPES:
            raise DatabaseSchemaError(
                "Una columna de texto requerida tiene un tipo incompatible."
            )

    fecha_creacion = next(
        column
        for column in tables["reservas"]["columns"]
        if column["name"] == "fecha_creacion"
    )
    if fecha_creacion["type"] not in {
        "date",
        "timestamp without time zone",
        "timestamp with time zone",
    }:
        raise DatabaseSchemaError(
            "La columna fecha_creacion no tiene un tipo de fecha compatible."
        )

    return tables


def _qualified_table(table: dict[str, Any], table_name: str) -> str:
    return (
        f'{_quote_identifier(table["schema"])}.'
        f"{_quote_identifier(table_name)}"
    )


def _count(session: Session, table: str) -> int:
    return int(session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one())


def get_database_diagnostics(session: Session) -> dict[str, Any]:
    tables = _inspect_tables(session)
    books_table = _qualified_table(tables["libros_fisicos"], "libros_fisicos")
    reservations_table = _qualified_table(tables["reservas"], "reservas")
    users_table = _qualified_table(tables["usuarios"], "usuarios")
    user_id = _quote_identifier("usuario_id")
    user_pk = _quote_identifier("id")
    username = _quote_identifier("username")
    email = _quote_identifier("correo")
    state = _quote_identifier("estado")
    created_at = _quote_identifier("fecha_creacion")
    description = _quote_identifier("descripcion")

    synthetic_reservations = session.execute(
        text(
            f"""
            SELECT COUNT(DISTINCT r.{_quote_identifier('id')})
            FROM {reservations_table} AS r
            LEFT JOIN {users_table} AS u ON u.{user_pk} = r.{user_id}
            WHERE u.{username} LIKE :username_prefix
               OR lower(trim(r.{email})) LIKE :email_suffix
            """
        ),
        {
            "username_prefix": "demo_user_%",
            "email_suffix": "%@demo.lectio.local",
        },
    ).scalar_one()

    states = session.execute(
        text(
            f"""
            SELECT {state} AS estado, COUNT(*) AS cantidad
            FROM {reservations_table}
            GROUP BY {state}
            ORDER BY {state} NULLS LAST
            """
        )
    ).mappings()

    monthly_reservations = session.execute(
        text(
            f"""
            SELECT date_trunc('month', {created_at}::timestamp) AS mes,
                   COUNT(*) AS cantidad
            FROM {reservations_table}
            GROUP BY date_trunc('month', {created_at}::timestamp)
            ORDER BY mes NULLS LAST
            """
        )
    ).mappings()

    demo_books = session.execute(
        text(f"SELECT COUNT(*) FROM {books_table} WHERE {description} ILIKE :marker"),
        {"marker": "%[DEMO-LECTIO]%"},
    ).scalar_one()

    schema = {
        name: {
            "schema": table["schema"],
            "columns": table["columns"],
        }
        for name, table in tables.items()
    }

    return {
        "status": "connected",
        "schema": schema,
        "counts": {
            "libros_fisicos": _count(session, books_table),
            "reservas": _count(session, reservations_table),
            "reservas_sinteticas": int(synthetic_reservations),
            "libros_demo": int(demo_books),
        },
        "reservas_por_estado": [
            {"estado": row["estado"], "cantidad": int(row["cantidad"])}
            for row in states
        ],
        "reservas_por_mes": [
            {
                "mes": (
                    row["mes"].strftime("%Y-%m")
                    if isinstance(row["mes"], (date, datetime))
                    else None
                ),
                "cantidad": int(row["cantidad"]),
            }
            for row in monthly_reservations
        ],
    }
