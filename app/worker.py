"""Runs the batch jobs, one item at a time, so slow model calls never block a request.

    python -m app.worker
"""
import logging
import time

import psycopg

from app import db, jobs, repo

log = logging.getLogger("worker")


def run_once(con: psycopg.Connection) -> bool:
    item = repo.claim_item(con)
    if item is None:
        return False
    jobs.process(con, item)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    while True:
        try:
            with db.connect() as con:
                db.migrate(con)
                log.info("worker ready")
                while True:
                    if not run_once(con):
                        time.sleep(1)
        except psycopg.OperationalError as exc:
            log.warning("database unavailable (%s), reconnecting in 3s", exc)
            time.sleep(3)


if __name__ == "__main__":
    main()
