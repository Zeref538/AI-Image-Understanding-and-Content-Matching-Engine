"""Per-call cost and the budget guard. Integers only: micro-dollars."""
from psycopg import Connection

from app.config import PRICING

MTOK = 1_000_000


class BudgetExceeded(Exception):
    pass


def cost_micros(purpose: str, input_tokens: int, output_tokens: int) -> int:
    p = PRICING["reference"][purpose]
    numerator = input_tokens * p["input_per_mtok_micros"] + output_tokens * p["output_per_mtok_micros"]
    return (numerator + MTOK // 2) // MTOK      # round half up to a whole micro-dollar


def check_budget(con: Connection, job_id: int | None) -> None:
    """Called before every model call. Stops a runaway job before it spends, not after."""
    limits = PRICING["budget"]
    if job_id is not None:
        calls = con.execute("SELECT COUNT(*) AS n FROM ai_calls WHERE job_id = %s", (job_id,)).fetchone()["n"]
        if calls >= limits["max_calls_per_job"]:
            raise BudgetExceeded(f"job {job_id} reached its limit of {limits['max_calls_per_job']} model calls")
    spent = con.execute("SELECT COALESCE(SUM(cost_micros), 0)::bigint AS s FROM ai_calls "
                        "WHERE created_at >= date_trunc('day', now())").fetchone()["s"]
    if spent >= limits["max_cost_micros_per_day"]:
        raise BudgetExceeded(f"today's reference cost {spent} micro-dollars reached the daily budget "
                             f"of {limits['max_cost_micros_per_day']}")
