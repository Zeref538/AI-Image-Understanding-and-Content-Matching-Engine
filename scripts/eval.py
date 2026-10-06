"""Measure the system against the labeled eval set, over the live API.

    python scripts/eval.py             # score the current thresholds, write eval/results.json + .md
    python scripts/eval.py --tune      # sweep thresholds on the TUNE split only, print the table

Top-1 precision, as the brief defines it: of the posts that have a correct image in the corpus,
the share whose first suggested image is one of them. A post where the guard refuses everything
scores 0 for that post: refusing is safe, but it is not a correct suggestion.
"""
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import Thresholds, load_thresholds  # noqa: E402
from app.guard import check  # noqa: E402

load_dotenv(ROOT / ".env")
BASE = os.environ.get("EVAL_BASE_URL", "http://localhost:8001")
image_labels = json.loads((ROOT / "eval" / "image_labels.json").read_text())
post_labels = json.loads((ROOT / "eval" / "post_labels.json").read_text())["posts"]


def fetch() -> list[dict]:
    http = httpx.Client(base_url=BASE, timeout=60)
    posts = {p["slug"]: p for p in http.get("/posts").json()["posts"]}
    images = {i["id"]: i for i in http.get("/images").json()["images"]}
    rows = []
    for slug, label in post_labels.items():
        r = http.get(f"/posts/{posts[slug]['id']}/images", params={"top": 100}).json()
        cands = [{"file": c["image"]["file"], "truth": image_labels[c["image"]["file"]]["subject"],
                  "similarity": c["similarity"], "image": images[c["image"]["id"]]} for c in r["candidates"]]
        rows.append({"slug": slug, "expected": label["expected"], "split": label["split"],
                     "post": {"kind": r["understood_as"]["kind"], "category": r["understood_as"]["category"]},
                     "status": r["status"], "candidates": cands})
    return rows


def decide(row: dict, t: Thresholds) -> dict | None:
    """Re-run the guard locally with thresholds t: the first accepted candidate, or None."""
    for c in row["candidates"]:
        img = dict(c["image"])
        if img["kind_prob"] is not None or img["confidence"] is not None:
            img["flagged"] = ((img["kind_prob"] is not None and img["kind_prob"] < t.min_kind_prob)
                              or img["confidence"] < t.min_confidence)
            img["flag_reason"] = "re-scored"
        if check(row["post"], img, c["similarity"], t).accepted:
            return c
    return None


def score(rows: list[dict], t: Thresholds) -> dict:
    has_answer = [r for r in rows if r["expected"]]
    no_answer = [r for r in rows if not r["expected"]]
    picks = {r["slug"]: decide(r, t) for r in rows}
    correct = sum(1 for r in has_answer if picks[r["slug"]] and picks[r["slug"]]["truth"] == r["expected"])
    wrong = [r["slug"] for r in rows if picks[r["slug"]] and picks[r["slug"]]["truth"] != r["expected"]]
    missed = [r["slug"] for r in has_answer if picks[r["slug"]] is None]
    refused_ok = sum(1 for r in no_answer if picks[r["slug"]] is None)
    ranking_top1 = sum(1 for r in has_answer if r["candidates"][0]["truth"] == r["expected"])
    suggested = sum(1 for p in picks.values() if p)
    return {"posts": len(rows), "posts_with_a_correct_image": len(has_answer),
            "top1_precision": round(correct / len(has_answer), 3) if has_answer else None,
            "top1_correct": correct,
            "ranking_only_top1": f"{ranking_top1}/{len(has_answer)}",
            "suggestions_made": suggested,
            "suggestion_precision": round(correct / suggested, 3) if suggested else None,
            "wrong_suggestions": wrong, "missed_posts": missed,
            "no_image_posts_refused": f"{refused_ok}/{len(no_answer)}"}


def fox_wolf_boundary(rows: list[dict]) -> dict:
    fox_posts = [r for r in rows if r["expected"] == "red fox"]
    fox = [c["similarity"] for r in fox_posts for c in r["candidates"] if c["truth"] == "red fox"]
    wolf = [c["similarity"] for r in fox_posts for c in r["candidates"] if c["truth"] in ("gray wolf", "coyote")]
    return {"fox_posts": len(fox_posts), "fox_image_similarity_min": min(fox), "fox_image_similarity_max": max(fox),
            "wolf_or_coyote_similarity_max": max(wolf),
            "wolf_or_coyote_pairs_above_fox_min": sum(1 for s in wolf if s >= min(fox)), "pairs": len(wolf)}


def main() -> None:
    rows = fetch()
    if "--tune" in sys.argv:
        tune = [r for r in rows if r["split"] == "tune"]
        print("min_sim  min_kind_p  top1   sugg.prec  wrong  missed  no-image refused   (TUNE split only)")
        for sim in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
            for kp in [0.4, 0.5, 0.6, 0.7, 0.8]:
                s = score(tune, Thresholds(sim, kp, 0.5))
                print(f"{sim:7.2f}  {kp:10.2f}  {s['top1_precision']:.3f}  {s['suggestion_precision'] or 0:9.3f}"
                      f"  {len(s['wrong_suggestions']):5d}  {len(s['missed_posts']):6d}  {s['no_image_posts_refused']:>16}")
        return
    t = load_thresholds()
    result = {"thresholds": t.__dict__, "all": score(rows, t),
              "tune": score([r for r in rows if r["split"] == "tune"], t),
              "test": score([r for r in rows if r["split"] == "test"], t),
              "fox_wolf_boundary": fox_wolf_boundary(rows),
              "per_post": [{"slug": r["slug"], "split": r["split"], "expected": r["expected"],
                            "post_kind": r["post"]["kind"],
                            "pick": (p := decide(r, t)) and {"file": p["file"], "truth": p["truth"],
                                                             "similarity": p["similarity"]},
                            "ranked_first": {k: r["candidates"][0][k] for k in ("file", "truth", "similarity")}}
                           for r in rows]}
    (ROOT / "eval" / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    a, te = result["all"], result["test"]
    print(f"top-1 precision: {a['top1_correct']}/{a['posts_with_a_correct_image']} = {a['top1_precision']}"
          f"   (held-out test split: {te['top1_correct']}/{te['posts_with_a_correct_image']} = {te['top1_precision']})")
    print(f"ranking alone, before the guard: {a['ranking_only_top1']} first-ranked images correct")
    print(f"suggestions made: {a['suggestions_made']}, precision {a['suggestion_precision']}, "
          f"wrong: {a['wrong_suggestions'] or 'none'}, missed: {a['missed_posts'] or 'none'}")
    print(f"posts with no suitable image correctly refused: {a['no_image_posts_refused']}")
    print(f"fox/wolf boundary: {result['fox_wolf_boundary']}")
    print("wrote eval/results.json")


if __name__ == "__main__":
    main()
