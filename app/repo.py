"""Data layer: every SQL statement lives here."""
from psycopg import Connection
from psycopg.types.json import Jsonb

# --- images ------------------------------------------------------------------------

IMAGE_COLUMNS = """i.id, i.file, i.title, i.license, i.source_url,
                   m.kind, m.subject, m.category, m.caption, m.confidence, m.kind_prob,
                   m.kind_alternatives, m.flagged, m.flag_reason, m.model, m.tagged_at,
                   COALESCE((SELECT array_agg(t.tag ORDER BY t.tag) FROM image_tags t WHERE t.image_id = i.id),
                            '{}') AS tags"""


def upsert_image(con: Connection, file: str, title: str, license: str, source_url: str, sha256: str) -> None:
    con.execute("""INSERT INTO images (file, title, license, source_url, sha256) VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (file) DO UPDATE SET title = EXCLUDED.title, license = EXCLUDED.license,
                       source_url = EXCLUDED.source_url, sha256 = EXCLUDED.sha256""",
                (file, title, license, source_url, sha256))


def list_images(con: Connection, flagged: bool | None = None) -> list[dict]:
    where = "" if flagged is None else ("WHERE m.flagged" if flagged else "WHERE m.flagged IS NOT TRUE")
    return con.execute(f"SELECT {IMAGE_COLUMNS} FROM images i LEFT JOIN image_metadata m ON m.image_id = i.id "
                       f"{where} ORDER BY i.id").fetchall()


def get_image(con: Connection, image_id: int) -> dict | None:
    return con.execute(f"SELECT {IMAGE_COLUMNS} FROM images i LEFT JOIN image_metadata m ON m.image_id = i.id "
                       "WHERE i.id = %s", (image_id,)).fetchone()


def save_image_metadata(con: Connection, image_id: int, model: str, r) -> None:
    t = r.tags
    con.execute(
        """INSERT INTO image_metadata (image_id, model, kind, subject, category, caption, confidence, kind_prob,
                                       kind_alternatives, flagged, flag_reason, raw)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (image_id) DO UPDATE SET model = EXCLUDED.model, kind = EXCLUDED.kind,
               subject = EXCLUDED.subject, category = EXCLUDED.category, caption = EXCLUDED.caption,
               confidence = EXCLUDED.confidence, kind_prob = EXCLUDED.kind_prob,
               kind_alternatives = EXCLUDED.kind_alternatives, flagged = EXCLUDED.flagged,
               flag_reason = EXCLUDED.flag_reason, raw = EXCLUDED.raw, tagged_at = now()""",
        (image_id, model, t.kind, t.subject, t.category, t.caption, t.confidence, r.kind_prob,
         Jsonb(r.alternatives), r.flagged, r.flag_reason, Jsonb(t.model_dump())))
    con.execute("DELETE FROM image_tags WHERE image_id = %s", (image_id,))
    for tag in [t.kind, *t.attributes]:
        con.execute("INSERT INTO image_tags (image_id, tag) VALUES (%s, %s) ON CONFLICT DO NOTHING", (image_id, tag))


# --- posts ---------------------------------------------------------------------------

def upsert_post(con: Connection, slug: str, title: str, body: str) -> None:
    con.execute("""INSERT INTO posts (slug, title, body) VALUES (%s, %s, %s)
                   ON CONFLICT (slug) DO UPDATE SET title = EXCLUDED.title, body = EXCLUDED.body""",
                (slug, title, body))


def list_posts(con: Connection) -> list[dict]:
    return con.execute("""SELECT p.id, p.slug, p.title, u.kind, u.subject, u.summary
                          FROM posts p LEFT JOIN post_understanding u ON u.post_id = p.id ORDER BY p.id""").fetchall()


def get_post(con: Connection, post_id: int) -> dict | None:
    return con.execute("""SELECT p.id, p.slug, p.title, p.body, u.kind, u.subject, u.category, u.summary, u.model
                          FROM posts p LEFT JOIN post_understanding u ON u.post_id = p.id
                          WHERE p.id = %s""", (post_id,)).fetchone()


def save_post_understanding(con: Connection, post_id: int, model: str, u) -> None:
    con.execute(
        """INSERT INTO post_understanding (post_id, model, kind, subject, category, summary, raw)
           VALUES (%s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (post_id) DO UPDATE SET model = EXCLUDED.model, kind = EXCLUDED.kind,
               subject = EXCLUDED.subject, category = EXCLUDED.category, summary = EXCLUDED.summary,
               raw = EXCLUDED.raw, understood_at = now()""",
        (post_id, model, u.kind, u.subject, u.category, u.summary, Jsonb(u.model_dump())))


# --- embeddings ------------------------------------------------------------------------

def save_embedding(con: Connection, owner_type: str, owner_id: int, model: str, text: str, vector: list[float]) -> None:
    con.execute("""INSERT INTO embeddings (owner_type, owner_id, model, text, vector) VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (owner_type, owner_id, model) DO UPDATE SET text = EXCLUDED.text,
                       vector = EXCLUDED.vector, created_at = now()""",
                (owner_type, owner_id, model, text, vector))


def get_embedding(con: Connection, owner_type: str, owner_id: int, model: str) -> list[float] | None:
    row = con.execute("SELECT vector FROM embeddings WHERE owner_type = %s AND owner_id = %s AND model = %s",
                      (owner_type, owner_id, model)).fetchone()
    return row["vector"] if row else None


def image_vectors(con: Connection, model: str) -> dict[int, list[float]]:
    rows = con.execute("SELECT owner_id, vector FROM embeddings WHERE owner_type = 'image' AND model = %s",
                       (model,)).fetchall()
    return {r["owner_id"]: r["vector"] for r in rows}


# --- suggestions and reviews -----------------------------------------------------------------

def upsert_suggestion(con: Connection, post_id: int, image_id: int, rank: int, similarity: float,
                      verdict: str, reasons: list[str]) -> int:
    return con.execute(
        """INSERT INTO suggestions (post_id, image_id, rank, similarity, verdict, reasons)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (post_id, image_id) DO UPDATE SET rank = EXCLUDED.rank, similarity = EXCLUDED.similarity,
               verdict = EXCLUDED.verdict, reasons = EXCLUDED.reasons, updated_at = now()
           RETURNING id""", (post_id, image_id, rank, similarity, verdict, Jsonb(reasons))).fetchone()["id"]


def get_suggestion(con: Connection, suggestion_id: int) -> dict | None:
    return con.execute(
        """SELECT s.*, i.file, p.slug, p.title AS post_title, r.decision, r.note, r.reviewed_at
           FROM suggestions s JOIN images i ON i.id = s.image_id JOIN posts p ON p.id = s.post_id
           LEFT JOIN reviews r ON r.suggestion_id = s.id WHERE s.id = %s""", (suggestion_id,)).fetchone()


def list_suggestions(con: Connection, top: int = 3) -> list[dict]:
    return con.execute(
        """SELECT s.id, s.image_id, s.rank, s.similarity, s.verdict, s.reasons, i.file, m.kind AS image_kind,
                  p.slug, u.kind AS post_kind, r.decision
           FROM suggestions s JOIN images i ON i.id = s.image_id JOIN posts p ON p.id = s.post_id
           LEFT JOIN image_metadata m ON m.image_id = i.id LEFT JOIN post_understanding u ON u.post_id = p.id
           LEFT JOIN reviews r ON r.suggestion_id = s.id
           WHERE s.rank <= %s ORDER BY p.id, s.rank""", (top,)).fetchall()


def save_review(con: Connection, suggestion_id: int, decision: str, note: str | None) -> dict:
    return con.execute(
        """INSERT INTO reviews (suggestion_id, decision, note) VALUES (%s, %s, %s)
           ON CONFLICT (suggestion_id) DO UPDATE SET decision = EXCLUDED.decision, note = EXCLUDED.note,
               reviewed_at = CASE WHEN reviews.decision = EXCLUDED.decision AND reviews.note IS NOT DISTINCT FROM
                                       EXCLUDED.note THEN reviews.reviewed_at ELSE now() END
           RETURNING *""", (suggestion_id, decision, note)).fetchone()


# --- jobs ------------------------------------------------------------------------------------

def job_by_key(con: Connection, key: str) -> dict | None:
    return con.execute("SELECT * FROM jobs WHERE idempotency_key = %s", (key,)).fetchone()


def create_job(con: Connection, kind: str, force: bool, key: str | None) -> dict:
    return con.execute("INSERT INTO jobs (kind, force, idempotency_key) VALUES (%s, %s, %s) RETURNING *",
                       (kind, force, key)).fetchone()


def add_job_items(con: Connection, job_id: int, target_type: str, ids: list[int]) -> None:
    for target_id in ids:
        con.execute("INSERT INTO job_items (job_id, target_type, target_id) VALUES (%s, %s, %s) "
                    "ON CONFLICT DO NOTHING", (job_id, target_type, target_id))


def job_progress(con: Connection, job_id: int) -> dict | None:
    job = con.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
    if job is None:
        return None
    counts = {r["status"]: r["n"] for r in con.execute(
        "SELECT status, COUNT(*) AS n FROM job_items WHERE job_id = %s GROUP BY status", (job_id,))}
    cost = con.execute("""SELECT COUNT(*) AS calls, COALESCE(SUM(cost_micros), 0)::bigint AS cost_micros,
                                 COALESCE(SUM(input_tokens), 0)::bigint AS input_tokens,
                                 COALESCE(SUM(output_tokens), 0)::bigint AS output_tokens,
                                 COUNT(*) FILTER (WHERE status <> 'ok') AS failed_calls
                          FROM ai_calls WHERE job_id = %s""", (job_id,)).fetchone()
    failures = con.execute("""SELECT target_type, target_id, attempts, last_error FROM job_items
                              WHERE job_id = %s AND status IN ('failed', 'skipped') ORDER BY id""",
                           (job_id,)).fetchall()
    return {"job": job, "items": counts, "total": sum(counts.values()), "cost": cost, "failures": failures}


def claim_item(con: Connection) -> dict | None:
    """Lease one due item for 10 minutes. If the worker dies mid-call, the lease runs out
    and another worker picks it up; a model call never runs inside an open transaction."""
    with con.transaction():
        item = con.execute(
            """SELECT ji.*, j.kind, j.force FROM job_items ji JOIN jobs j ON j.id = ji.job_id
               WHERE ji.status = 'pending' AND ji.next_attempt_at <= now() AND j.status IN ('queued', 'running')
               ORDER BY ji.id LIMIT 1 FOR UPDATE OF ji SKIP LOCKED""").fetchone()
        if item:
            con.execute("UPDATE job_items SET next_attempt_at = now() + interval '10 minutes' WHERE id = %s",
                        (item["id"],))
            con.execute("""UPDATE jobs SET status = 'running', started_at = COALESCE(started_at, now())
                           WHERE id = %s AND status = 'queued'""", (item["job_id"],))
        return item


def finish_item(con: Connection, item_id: int, status: str, error: str | None = None) -> None:
    con.execute("""UPDATE job_items SET status = %s, attempts = attempts + 1, last_error = %s, finished_at = now()
                   WHERE id = %s""", (status, error, item_id))


def retry_item(con: Connection, item_id: int, error: str, delay_s: int) -> None:
    con.execute("""UPDATE job_items SET attempts = attempts + 1, last_error = %s,
                       next_attempt_at = now() + make_interval(secs => %s) WHERE id = %s""",
                (error, delay_s, item_id))


def stop_job_for_budget(con: Connection, job_id: int, reason: str) -> None:
    with con.transaction():
        con.execute("""UPDATE job_items SET status = 'skipped', last_error = %s, finished_at = now()
                       WHERE job_id = %s AND status = 'pending'""", (reason, job_id))
        con.execute("UPDATE jobs SET status = 'stopped_budget', finished_at = now() WHERE id = %s", (job_id,))


def close_job_if_finished(con: Connection, job_id: int) -> str | None:
    row = con.execute("""SELECT COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                                COUNT(*) FILTER (WHERE status = 'failed') AS failed
                         FROM job_items WHERE job_id = %s""", (job_id,)).fetchone()
    if row["pending"]:
        return None
    status = "done_with_failures" if row["failed"] else "done"
    updated = con.execute("""UPDATE jobs SET status = %s, finished_at = now()
                             WHERE id = %s AND status IN ('queued', 'running') RETURNING id""",
                          (status, job_id)).fetchone()
    return status if updated else None


# --- cost log and alerts ------------------------------------------------------------------------

def log_call(con: Connection, *, job_id, purpose, target_type, target_id, model, input_tokens, output_tokens,
             duration_ms, cost_micros, status, error=None) -> None:
    con.execute("""INSERT INTO ai_calls (job_id, purpose, target_type, target_id, model, input_tokens,
                                         output_tokens, duration_ms, cost_micros, status, error)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (job_id, purpose, target_type, target_id, model, input_tokens, output_tokens, duration_ms,
                 cost_micros, status, error))


def cost_summary(con: Connection) -> dict:
    by_purpose = con.execute("""SELECT purpose, model, COUNT(*) AS calls, SUM(input_tokens)::bigint AS input_tokens,
                                       SUM(output_tokens)::bigint AS output_tokens,
                                       SUM(cost_micros)::bigint AS cost_micros,
                                       ROUND(AVG(duration_ms))::int AS avg_ms,
                                       COUNT(*) FILTER (WHERE status <> 'ok') AS not_ok
                                FROM ai_calls GROUP BY purpose, model ORDER BY purpose""").fetchall()
    recent = con.execute("""SELECT id, job_id, purpose, target_type, target_id, model, input_tokens, output_tokens,
                                   duration_ms, cost_micros, status, error, created_at
                            FROM ai_calls ORDER BY id DESC LIMIT 20""").fetchall()
    return {"by_purpose": by_purpose, "recent": recent}


def add_alert(con: Connection, kind: str, message: str) -> None:
    con.execute("INSERT INTO alerts (kind, message) VALUES (%s, %s)", (kind, message))
