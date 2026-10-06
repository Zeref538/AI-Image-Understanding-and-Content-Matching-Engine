"""Load the 50 corpus images and 16 posts into the database. Safe to run twice.

    python -m app.seed
Then start the batch jobs (POST /jobs) so the worker tags images and indexes posts.
"""
import hashlib
import json

from app import config, db, repo


def main() -> None:
    corpus = json.loads((config.ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))
    posts = json.loads((config.ROOT / "data" / "posts.json").read_text(encoding="utf-8"))
    with db.connect() as con:
        db.migrate(con)
        with con.transaction():
            for item in corpus:
                path = config.IMAGES_DIR / item["file"]
                if not path.exists():
                    raise SystemExit(f"{path} is missing: run python scripts/fetch_corpus.py")
                repo.upsert_image(con, item["file"], item["title"], item["license"], item["landing_page"],
                                  hashlib.sha256(path.read_bytes()).hexdigest())
            for p in posts:
                repo.upsert_post(con, p["slug"], p["title"], p["body"])
    print(f"seeded {len(corpus)} images and {len(posts)} posts")


if __name__ == "__main__":
    main()
