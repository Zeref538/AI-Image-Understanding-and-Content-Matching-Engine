from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from app import config

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def connect(url: str | None = None) -> psycopg.Connection:
    # autocommit, so every multi-step write is an explicit `with con.transaction():`
    return psycopg.connect(url or config.DATABASE_URL, autocommit=True, row_factory=dict_row)


def migrate(con: psycopg.Connection) -> list[str]:
    """Apply new migrations/*.sql in name order, once each."""
    con.execute("SELECT pg_advisory_lock(726002)")  # api and worker both start by migrating
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
                           name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())""")
        done = {r["name"] for r in con.execute("SELECT name FROM schema_migrations")}
        applied = []
        for f in sorted(MIGRATIONS.glob("*.sql")):
            if f.name not in done:
                with con.transaction():
                    con.execute(f.read_text(encoding="utf-8"))
                    con.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (f.name,))
                applied.append(f.name)
        return applied
    finally:
        con.execute("SELECT pg_advisory_unlock(726002)")


if __name__ == "__main__":
    with connect() as c:
        print("applied:", migrate(c) or "nothing new")
