import os
import sqlite3
from collections import defaultdict

from sqlalchemy import create_engine, inspect, MetaData, Table, text


SQLITE_PATH = "/mlflow/db/mlflow.db"

POSTGRES_URI = os.environ.get("MLFLOW_BACKEND_STORE_URI")

BATCH_SIZE = 1000

SKIP_TABLES = {
    "alembic_version",
    "workspaces",
}


def get_sqlite_connection():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_postgres_engine():
    if not POSTGRES_URI:
        raise RuntimeError(
            "MLFLOW_BACKEND_STORE_URI environment variable is not set."
        )

    return create_engine(
        POSTGRES_URI,
        future=True,
        pool_pre_ping=True,
    )


def get_sqlite_tables(sqlite_conn):
    rows = sqlite_conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()

    return [row[0] for row in rows]


def get_postgres_tables(engine):
    inspector = inspect(engine)
    return sorted(inspector.get_table_names(schema="public"))


def validate_schema(sqlite_conn, engine):
    sqlite_tables = set(get_sqlite_tables(sqlite_conn))
    postgres_tables = set(get_postgres_tables(engine))

    if sqlite_tables != postgres_tables:
        only_sqlite = sorted(sqlite_tables - postgres_tables)
        only_postgres = sorted(postgres_tables - sqlite_tables)

        raise RuntimeError(
            "Schema mismatch detected.\n"
            f"Only in SQLite: {only_sqlite}\n"
            f"Only in PostgreSQL: {only_postgres}"
        )

    print(f"Schema validation passed: {len(sqlite_tables)} tables")


def validate_alembic_version(sqlite_conn, engine):
    sqlite_version = sqlite_conn.execute(
        "SELECT version_num FROM alembic_version"
    ).fetchone()[0]

    with engine.connect() as conn:
        postgres_version = conn.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one()

    print(f"SQLite Alembic version    : {sqlite_version}")
    print(f"PostgreSQL Alembic version: {postgres_version}")

    if sqlite_version != postgres_version:
        raise RuntimeError("Alembic versions do not match.")

    print("Alembic version validation passed.")


def ensure_postgres_is_empty(engine):
    tables = get_postgres_tables(engine)

    non_empty = []

    with engine.connect() as conn:
        for table_name in tables:
            if table_name in SKIP_TABLES:
                continue

            count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')
            ).scalar_one()

            if count != 0:
                non_empty.append((table_name, count))

    if non_empty:
        raise RuntimeError(
            "PostgreSQL destination is not empty:\n"
            + "\n".join(
                f"{table}: {count}"
                for table, count in non_empty
            )
        )

    print("PostgreSQL destination is empty.")


def build_dependency_order(engine):
    metadata = MetaData()
    metadata.reflect(bind=engine, schema="public")

    ordered_tables = [
        table.name
        for table in metadata.sorted_tables
        if table.name not in SKIP_TABLES
    ]

    return ordered_tables


def copy_table(sqlite_conn, pg_conn, pg_table, table_name):
    source_count = sqlite_conn.execute(
        f'SELECT COUNT(*) FROM "{table_name}"'
    ).fetchone()[0]

    if source_count == 0:
        print(f"{table_name:<35} 0 rows")
        return 0

    cursor = sqlite_conn.execute(
        f'SELECT * FROM "{table_name}"'
    )

    total_inserted = 0

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)

        if not rows:
            break

        records = [dict(row) for row in rows]

        pg_conn.execute(
            pg_table.insert(),
            records,
        )

        total_inserted += len(records)

    print(
        f"{table_name:<35} "
        f"{total_inserted:>8} rows"
    )

    return total_inserted


def migrate(sqlite_conn, engine):
    metadata = MetaData()
    metadata.reflect(bind=engine, schema="public")

    ordered_tables = build_dependency_order(engine)

    print()
    print("=" * 80)
    print("TABLE COPY ORDER")
    print("=" * 80)

    for index, table_name in enumerate(ordered_tables, start=1):
        print(f"{index:02d}. {table_name}")

    print()
    print("=" * 80)
    print("STARTING DATA MIGRATION")
    print("=" * 80)

    with engine.begin() as pg_conn:
        for table_name in ordered_tables:
            pg_table = metadata.tables[f"public.{table_name}"]

            copy_table(
                sqlite_conn,
                pg_conn,
                pg_table,
                table_name,
            )

    print()
    print("Data migration transaction committed.")


def compare_row_counts(sqlite_conn, engine):
    print()
    print("=" * 80)
    print("ROW COUNT VALIDATION")
    print("=" * 80)

    mismatches = []

    tables = get_sqlite_tables(sqlite_conn)

    with engine.connect() as pg_conn:
        for table_name in tables:
            if table_name in SKIP_TABLES:
                continue

            sqlite_count = sqlite_conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            postgres_count = pg_conn.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')
            ).scalar_one()

            status = "OK"

            if sqlite_count != postgres_count:
                status = "MISMATCH"
                mismatches.append(
                    (
                        table_name,
                        sqlite_count,
                        postgres_count,
                    )
                )

            print(
                f"{table_name:<35} "
                f"SQLite={sqlite_count:<8} "
                f"PostgreSQL={postgres_count:<8} "
                f"{status}"
            )

    if mismatches:
        raise RuntimeError(
            f"{len(mismatches)} table count mismatch(es) detected."
        )

    print()
    print("All table row counts match.")


def reset_postgres_sequences(engine):
    print()
    print("=" * 80)
    print("RESETTING POSTGRESQL SEQUENCES")
    print("=" * 80)

    inspector = inspect(engine)

    with engine.begin() as conn:
        for table_name in inspector.get_table_names(schema="public"):

            if table_name in SKIP_TABLES:
                continue

            columns = inspector.get_columns(
                table_name,
                schema="public",
            )

            for column in columns:
                column_name = column["name"]

                sequence_name = conn.execute(
                    text(
                        """
                        SELECT pg_get_serial_sequence(
                            :table_name,
                            :column_name
                        )
                        """
                    ),
                    {
                        "table_name": f"public.{table_name}",
                        "column_name": column_name,
                    },
                ).scalar()

                if not sequence_name:
                    continue

                maximum = conn.execute(
                    text(
                        f'SELECT MAX("{column_name}") '
                        f'FROM "{table_name}"'
                    )
                ).scalar()

                if maximum is None:
                    conn.execute(
                        text(
                            "SELECT setval("
                            ":sequence_name, 1, false)"
                        ),
                        {
                            "sequence_name": sequence_name,
                        },
                    )
                else:
                    conn.execute(
                        text(
                            "SELECT setval("
                            ":sequence_name, :maximum, true)"
                        ),
                        {
                            "sequence_name": sequence_name,
                            "maximum": maximum,
                        },
                    )

                print(
                    f"{table_name}.{column_name} "
                    f"-> {sequence_name} "
                    f"(max={maximum})"
                )


def validate_important_entities(sqlite_conn, engine):
    print()
    print("=" * 80)
    print("IMPORTANT ENTITY VALIDATION")
    print("=" * 80)

    checks = [
        "experiments",
        "runs",
        "registered_models",
        "model_versions",
        "registered_model_aliases",
        "logged_models",
    ]

    with engine.connect() as pg_conn:

        for table_name in checks:

            sqlite_count = sqlite_conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            postgres_count = pg_conn.execute(
                text(
                    f'SELECT COUNT(*) FROM "{table_name}"'
                )
            ).scalar_one()

            print(
                f"{table_name:<30} "
                f"SQLite={sqlite_count:<6} "
                f"PostgreSQL={postgres_count:<6}"
            )

def validate_bootstrap_tables(sqlite_conn, engine):
    print()
    print("=" * 80)
    print("BOOTSTRAP TABLE VALIDATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Validate workspaces
    # ---------------------------------------------------------

    sqlite_rows = sqlite_conn.execute(
        """
        SELECT
            name,
            description,
            default_artifact_root,
            trace_archival_location,
            trace_archival_retention
        FROM workspaces
        ORDER BY name
        """
    ).fetchall()

    sqlite_data = [
        tuple(row)
        for row in sqlite_rows
    ]

    with engine.connect() as conn:
        postgres_rows = conn.execute(
            text(
                """
                SELECT
                    name,
                    description,
                    default_artifact_root,
                    trace_archival_location,
                    trace_archival_retention
                FROM workspaces
                ORDER BY name
                """
            )
        ).fetchall()

    postgres_data = [
        tuple(row)
        for row in postgres_rows
    ]

    print(f"SQLite workspaces    : {len(sqlite_data)}")
    print(f"PostgreSQL workspaces: {len(postgres_data)}")

    if sqlite_data != postgres_data:
        raise RuntimeError(
            "Bootstrap workspace data does not match.\n"
            f"SQLite: {sqlite_data}\n"
            f"PostgreSQL: {postgres_data}"
        )

    print("Workspace bootstrap validation passed.")

def main():
    print("=" * 80)
    print("MLFLOW SQLITE -> POSTGRESQL MIGRATION")
    print("=" * 80)

    sqlite_conn = get_sqlite_connection()
    engine = get_postgres_engine()

    try:
        validate_schema(
            sqlite_conn,
            engine,
        )

        validate_alembic_version(
            sqlite_conn,
            engine,
        )

        validate_bootstrap_tables(
            sqlite_conn,
            engine,
        )
        
        ensure_postgres_is_empty(engine)

        migrate(
            sqlite_conn,
            engine,
        )

        compare_row_counts(
            sqlite_conn,
            engine,
        )

        reset_postgres_sequences(engine)

        validate_important_entities(
            sqlite_conn,
            engine,
        )

        print()
        print("=" * 80)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print(
            "SQLite source has NOT been modified."
        )

        print(
            "PostgreSQL now contains the migrated MLflow metadata."
        )

    finally:
        sqlite_conn.close()
        engine.dispose()


if __name__ == "__main__":
    main()