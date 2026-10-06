"""Rank every image for a post, run each through the guard, and keep the result for review."""
from psycopg import Connection

from app import config, guard, repo
from app.embeddings import cosine


class NotReady(Exception):
    pass


def _post_with_vector(con: Connection, post_id: int) -> tuple[dict, list[float]]:
    post = repo.get_post(con, post_id)
    if post is None:
        raise LookupError(f"post {post_id} not found")
    vector = repo.get_embedding(con, "post", post_id, config.EMBED_MODEL)
    if post["kind"] is None or vector is None:
        raise NotReady(f"post {post_id} has not been indexed yet: start an index_posts job")
    return post, vector


def _image_view(img: dict) -> dict:
    return {"id": img["id"], "file": img["file"], "kind": img["kind"], "subject": img["subject"],
            "caption": img["caption"], "flagged": img["flagged"], "kind_prob": img["kind_prob"]}


def rank(con: Connection, post_id: int, top: int = 10) -> dict:
    post, post_vec = _post_with_vector(con, post_id)
    vectors = repo.image_vectors(con, config.EMBED_MODEL)
    if not vectors:
        raise NotReady("no images have been tagged yet: start a tag_images job")
    images = {i["id"]: i for i in repo.list_images(con)}
    t = config.load_thresholds()
    scored = sorted(((cosine(post_vec, v), images[i]) for i, v in vectors.items()), key=lambda x: -x[0])
    verdicts = []
    for position, (sim, img) in enumerate(scored, 1):
        v = guard.check(post, img, sim, t)
        sid = repo.upsert_suggestion(con, post_id, img["id"], position, sim, "accepted" if v.accepted else "rejected",
                                     v.reasons)
        verdicts.append((img, sim, v, sid))
    best = next(((img, sim, v, sid) for img, sim, v, sid in verdicts if v.accepted), None)
    body = {
        "post": {"id": post["id"], "slug": post["slug"], "title": post["title"]},
        "understood_as": {"kind": post["kind"], "subject": post["subject"], "category": post["category"],
                          "summary": post["summary"]},
        "thresholds": t.__dict__,
        "candidates": [{"rank": n, "suggestion_id": sid, "image": _image_view(img), "similarity": round(sim, 4),
                        "verdict": "accepted" if v.accepted else "rejected", "reasons": v.reasons}
                       for n, (img, sim, v, sid) in enumerate(verdicts[:top], 1)],
    }
    if best:
        img, sim, v, sid = best
        body.update(status="match", suggestion={"suggestion_id": sid, "image": _image_view(img),
                                                "similarity": round(sim, 4), "reasons": v.reasons})
    else:
        body.update(status="no_confident_match", suggestion=None,
                    reasons=guard.no_match_summary([(img, sim, v) for img, sim, v, _ in verdicts]))
    return body


def check_pair(con: Connection, post_id: int, image_id: int) -> dict:
    """Force one image through the guard for one post, whatever its rank."""
    post, post_vec = _post_with_vector(con, post_id)
    image = repo.get_image(con, image_id)
    if image is None:
        raise LookupError(f"image {image_id} not found")
    vector = repo.get_embedding(con, "image", image_id, config.EMBED_MODEL)
    sim = cosine(post_vec, vector) if vector else 0.0
    v = guard.check(post, image if vector else None, sim, config.load_thresholds())
    return {"post": {"id": post["id"], "slug": post["slug"], "kind": post["kind"]},
            "image": _image_view(image), "similarity": round(sim, 4),
            "result": "ACCEPTED" if v.accepted else "REJECTED", "reasons": v.reasons}
