"""HTTP layer: routes, validation and status codes. Decisions live in matching.py and guard.py."""
import html
import re
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import config, db, jobs, matching, repo

app = FastAPI(title="AI Image Understanding & Content Matching Engine", version="1.0.0")
KEY_PATTERN = re.compile(r"^[\x21-\x7e]{1,255}$")
Id = Annotated[int, Path(ge=1, le=2_147_483_647)]  # a fresh Path per use; one shared instance breaks


class ApiError(Exception):
    def __init__(self, status: int, error: str, message: str):
        self.status, self.error, self.message = status, error, message


@app.exception_handler(ApiError)
async def api_error(_: Request, exc: ApiError):
    return JSONResponse({"error": exc.error, "message": exc.message}, status_code=exc.status)


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]} for e in exc.errors()]
    return JSONResponse({"error": "invalid_request", "message": "The request is invalid.", "details": details},
                        status_code=422)


def get_con():
    con = db.connect()
    try:
        yield con
    finally:
        con.close()


def _found(row, what: str):
    if row is None:
        raise ApiError(404, "not_found", f"{what} not found")
    return row


@app.get("/health")
def health(con=Depends(get_con)):
    con.execute("SELECT 1")
    return {"status": "ok", "db": "ok", "models": {"vision": config.VISION_MODEL, "text": config.TEXT_MODEL,
                                                   "embedding": config.EMBED_MODEL}}


# --- jobs --------------------------------------------------------------------------------

class JobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tag_images", "index_posts"]
    force: bool = Field(False, description="Re-run items that are already done")


@app.post("/jobs", status_code=202, summary="Start a batch job; the worker runs it in the background")
def start_job(body: JobIn, con=Depends(get_con), idempotency_key: str | None = Header(None, alias="Idempotency-Key")):
    if idempotency_key is not None and not KEY_PATTERN.match(idempotency_key):
        raise ApiError(400, "invalid_idempotency_key", "Idempotency-Key must be 1 to 255 printable characters.")
    job, created = jobs.create(con, body.kind, body.force, idempotency_key)
    progress = repo.job_progress(con, job["id"])
    return JSONResponse(jsonable_encoder({"created": created, **progress}), status_code=202 if created else 200,
                        headers={"Location": f"/jobs/{job['id']}"})


@app.get("/jobs/{job_id}", summary="Progress, failures and cost of one job")
def get_job(job_id: Id, con=Depends(get_con)):
    return _found(repo.job_progress(con, job_id), "job")


# --- images and posts ------------------------------------------------------------------------

@app.get("/images")
def list_images(con=Depends(get_con), flagged: bool | None = Query(None)):
    return {"images": repo.list_images(con, flagged)}


@app.get("/images/{image_id}")
def get_image(image_id: Id, con=Depends(get_con)):
    return _found(repo.get_image(con, image_id), "image")


@app.get("/images/{image_id}/file", include_in_schema=False)
def image_file(image_id: Id, con=Depends(get_con)):
    img = _found(repo.get_image(con, image_id), "image")
    return FileResponse(config.IMAGES_DIR / img["file"], media_type="image/jpeg")


@app.get("/posts")
def list_posts(con=Depends(get_con)):
    return {"posts": repo.list_posts(con)}


@app.get("/posts/{post_id}")
def get_post(post_id: Id, con=Depends(get_con)):
    return _found(repo.get_post(con, post_id), "post")


@app.get("/posts/{post_id}/images", summary="Ranked images for a post, each through the mismatch guard")
def post_images(post_id: Id, top: int = Query(10, ge=1, le=100), con=Depends(get_con)):
    try:
        return matching.rank(con, post_id, top)
    except LookupError as exc:
        raise ApiError(404, "not_found", str(exc))
    except matching.NotReady as exc:
        raise ApiError(409, "not_indexed", str(exc))


@app.get("/posts/{post_id}/images/{image_id}/check", summary="Force one image through the guard for one post")
def check_pair(post_id: Id, image_id: Id, con=Depends(get_con)):
    try:
        return matching.check_pair(con, post_id, image_id)
    except LookupError as exc:
        raise ApiError(404, "not_found", str(exc))
    except matching.NotReady as exc:
        raise ApiError(409, "not_indexed", str(exc))


# --- review ------------------------------------------------------------------------------------

class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: str | None = Field(None, max_length=500)


@app.get("/suggestions/{suggestion_id}", summary="Why this image was suggested or refused")
def get_suggestion(suggestion_id: Id, con=Depends(get_con)):
    s = _found(repo.get_suggestion(con, suggestion_id), "suggestion")
    s["image"] = repo.get_image(con, s["image_id"])
    return s


def _review(con, suggestion_id: int, decision: str, body: ReviewIn | None):
    _found(repo.get_suggestion(con, suggestion_id), "suggestion")
    review = repo.save_review(con, suggestion_id, decision, body.note if body else None)
    return {"suggestion_id": suggestion_id, "decision": review["decision"], "note": review["note"],
            "reviewed_at": review["reviewed_at"]}


@app.post("/suggestions/{suggestion_id}/approve", summary="Approve a pairing (safe to repeat)")
def approve(suggestion_id: Id, body: ReviewIn | None = None, con=Depends(get_con)):
    return _review(con, suggestion_id, "approved", body)


@app.post("/suggestions/{suggestion_id}/reject", summary="Reject a pairing (safe to repeat)")
def reject(suggestion_id: Id, body: ReviewIn | None = None, con=Depends(get_con)):
    return _review(con, suggestion_id, "rejected", body)


@app.get("/review", response_class=HTMLResponse, summary="Every post's top suggestions as a plain table")
def review_table(top: int = Query(3, ge=1, le=20), con=Depends(get_con)):
    rows = repo.list_suggestions(con, top)
    e = html.escape
    body = "".join(
        f"<tr class='{r['verdict']}'><td>{e(r['slug'])}</td><td>{e(r['post_kind'] or '')}</td><td>{r['rank']}</td>"
        f"<td><img src='/images/{r['image_id']}/file' height='60'> {e(r['file'])}</td>"
        f"<td>{e(r['image_kind'] or '')}</td><td>{r['similarity']:.3f}</td><td>{r['verdict']}</td>"
        f"<td>{e('; '.join(r['reasons']))}</td><td>{e(r['decision'] or '')}</td><td>{r['id']}</td></tr>"
        for r in rows)
    return f"""<!doctype html><meta charset="utf-8"><title>Review</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;
padding:4px 6px;vertical-align:middle}}tr.accepted{{background:#e8f6ea}}tr.rejected{{background:#fbeaea}}</style>
<h1>Suggestions (top {top} per post)</h1>
<p>Approve or reject with <code>POST /suggestions/&lt;id&gt;/approve</code> or <code>/reject</code>.</p>
<table><tr><th>post</th><th>post kind</th><th>rank</th><th>image</th><th>image kind</th><th>similarity</th>
<th>guard</th><th>reasons</th><th>review</th><th>id</th></tr>{body}</table>"""


@app.get("/costs", summary="The per-call cost log and totals")
def get_costs(con=Depends(get_con)):
    return {"note": "Local models cost $0. cost_micros is the reference cloud price from config/pricing.json.",
            **repo.cost_summary(con)}
