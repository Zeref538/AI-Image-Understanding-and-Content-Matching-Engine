"""Re-apply the flag thresholds in config/guard.json to the stored tags. No model calls:
the measured kind_prob and the model's confidence are already saved, so changing a
threshold never means paying to tag the images again. Runs at every api start.

    python -m app.reflag
"""
from app import config, db
from app.vision import ImageTags, flag


def reflag(con) -> int:
    t = config.load_thresholds()
    rows = con.execute("SELECT image_id, raw, kind_prob, kind_alternatives FROM image_metadata").fetchall()
    flagged = 0
    with con.transaction():
        for r in rows:
            reason = flag(ImageTags.model_validate(r["raw"]), r["kind_prob"], r["kind_alternatives"], t)
            con.execute("UPDATE image_metadata SET flagged = %s, flag_reason = %s WHERE image_id = %s",
                        (reason is not None, reason, r["image_id"]))
            flagged += reason is not None
    return flagged


if __name__ == "__main__":
    with db.connect() as c:
        db.migrate(c)
        print(f"re-flagged with {config.load_thresholds()}: {reflag(c)} flagged")
