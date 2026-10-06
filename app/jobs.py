"""Batch jobs: tag every image, or understand every post. Each item is retried, flagged or
failed, and every model call (good or bad) lands in the cost log."""
import logging

import psycopg
from psycopg import Connection

from app import config, costs, embeddings, ollama, posts, repo, vision

log = logging.getLogger("jobs")
MAX_ATTEMPTS = 3


def create(con: Connection, kind: str, force: bool, key: str | None) -> tuple[dict, bool]:
    """Returns (job, created). The same Idempotency-Key returns the first job, not a second one."""
    if key and (existing := repo.job_by_key(con, key)):
        return existing, False
    owner = "image" if kind == "tag_images" else "post"
    table = "images" if owner == "image" else "posts"
    # Without force, only items that have no embedding for the current model: a re-run after
    # a crash finishes the rest instead of paying for everything again.
    ids = [r["id"] for r in con.execute(
        f"""SELECT t.id FROM {table} t WHERE %s OR NOT EXISTS (
                SELECT 1 FROM embeddings e WHERE e.owner_type = %s AND e.owner_id = t.id AND e.model = %s)
            ORDER BY t.id""", (force, owner, config.EMBED_MODEL))]
    try:
        with con.transaction():
            job = repo.create_job(con, kind, force, key)
            repo.add_job_items(con, job["id"], owner, ids)
            if not ids:
                con.execute("UPDATE jobs SET status = 'done', started_at = now(), finished_at = now() WHERE id = %s",
                            (job["id"],))
    except psycopg.errors.UniqueViolation:
        return repo.job_by_key(con, key), False          # the same key arrived twice at once
    return con.execute("SELECT * FROM jobs WHERE id = %s", (job["id"],)).fetchone(), True


def _log(con, item, purpose, model, status, *, call=None, tokens_in=0, tokens_out=0, ms=0, error=None):
    if call is not None:
        tokens_in, tokens_out, ms = call.input_tokens, call.output_tokens, call.duration_ms
    repo.log_call(con, job_id=item["job_id"], purpose=purpose, target_type=item["target_type"],
                  target_id=item["target_id"], model=model, input_tokens=tokens_in, output_tokens=tokens_out,
                  duration_ms=ms, cost_micros=costs.cost_micros(purpose, tokens_in, tokens_out),
                  status=status, error=error)


def _embed(con, item, text: str, as_query: bool) -> list[float]:
    costs.check_budget(con, item["job_id"])
    try:
        vector, tokens, ms = embeddings.embed_one(text, as_query)
    except ollama.ModelError as exc:
        _log(con, item, "embedding", config.EMBED_MODEL, "error", error=str(exc))
        raise
    _log(con, item, "embedding", config.EMBED_MODEL, "ok", tokens_in=tokens, ms=ms)
    return vector


def _tag_image(con: Connection, item: dict, attempt: int) -> str:
    image = repo.get_image(con, item["target_id"])
    try:
        r = vision.tag_image(config.IMAGES_DIR / image["file"], config.load_thresholds(), attempt)
    except vision.InvalidOutput as exc:
        _log(con, item, "vision", config.VISION_MODEL, "invalid_output", call=exc.call, error=str(exc))
        raise
    except ollama.ModelError as exc:
        _log(con, item, "vision", config.VISION_MODEL, "error", error=str(exc))
        raise
    _log(con, item, "vision", config.VISION_MODEL, "ok", call=r.call)
    text = embeddings.image_text(r.tags.subject, r.tags.caption, r.tags.attributes)
    vector = _embed(con, item, text, as_query=False)
    with con.transaction():
        repo.save_image_metadata(con, image["id"], config.VISION_MODEL, r)
        repo.save_embedding(con, "image", image["id"], config.EMBED_MODEL, text, vector)
    return "flagged" if r.flagged else "done"


def _index_post(con: Connection, item: dict, attempt: int) -> str:
    post = repo.get_post(con, item["target_id"])
    try:
        u, call = posts.understand(post["title"], post["body"], attempt)
    except vision.InvalidOutput as exc:
        _log(con, item, "post_understanding", config.TEXT_MODEL, "invalid_output", call=exc.call, error=str(exc))
        raise
    except ollama.ModelError as exc:
        _log(con, item, "post_understanding", config.TEXT_MODEL, "error", error=str(exc))
        raise
    _log(con, item, "post_understanding", config.TEXT_MODEL, "ok", call=call)
    text = embeddings.post_text(u.subject, u.summary)
    vector = _embed(con, item, text, as_query=True)
    with con.transaction():
        repo.save_post_understanding(con, post["id"], config.TEXT_MODEL, u)
        repo.save_embedding(con, "post", post["id"], config.EMBED_MODEL, text, vector)
    return "done"


def process(con: Connection, item: dict) -> None:
    attempt = item["attempts"] + 1
    try:
        costs.check_budget(con, item["job_id"])
        work = _tag_image if item["target_type"] == "image" else _index_post
        repo.finish_item(con, item["id"], work(con, item, attempt))
    except costs.BudgetExceeded as exc:
        repo.stop_job_for_budget(con, item["job_id"], f"budget: {exc}")
        repo.add_alert(con, "budget_exceeded", f"job {item['job_id']} stopped: {exc}")
        log.error("ALERT job %s stopped by the budget guard: %s", item["job_id"], exc)
        return
    except (ollama.ModelError, vision.InvalidOutput) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if attempt >= MAX_ATTEMPTS:
            repo.finish_item(con, item["id"], "failed", error)
            repo.add_alert(con, "job_item_failed", f"job {item['job_id']} {item['target_type']} "
                                                   f"{item['target_id']} failed {attempt} times: {error}")
            log.error("ALERT %s %s failed %d times: %s", item["target_type"], item["target_id"], attempt, error)
        else:
            repo.retry_item(con, item["id"], error, 2 ** attempt)
            log.warning("%s %s attempt %d failed, retrying: %s", item["target_type"], item["target_id"], attempt, error)
    status = repo.close_job_if_finished(con, item["job_id"])
    if status:
        log.info("job %s finished: %s", item["job_id"], status)
        if status == "done_with_failures":
            repo.add_alert(con, "job_failures", f"job {item['job_id']} finished with failed items")
