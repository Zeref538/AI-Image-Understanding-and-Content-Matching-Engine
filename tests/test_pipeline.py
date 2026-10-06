"""The batch jobs, the cost log, the budget guard, matching and the review API, through
the real database and HTTP layer with a fake model server."""
import json

from conftest import run_jobs

from app import config, costs, repo


def start(client, kind, key=None, force=False):
    headers = {"Idempotency-Key": key} if key else {}
    return client.post("/jobs", json={"kind": kind, "force": force}, headers=headers)


# --- batch jobs -------------------------------------------------------------------------------

def test_tag_images_job_tags_every_image_and_logs_every_call(client, seeded, fake):
    r = start(client, "tag_images")
    assert r.status_code == 202 and r.json()["total"] == 3
    run_jobs(seeded)
    job = client.get(f"/jobs/{r.json()['job']['id']}").json()
    assert job["job"]["status"] == "done" and job["items"] == {"done": 3}
    # one vision + one embedding call per image, each with a cost entry
    assert job["cost"]["calls"] == 6 and job["cost"]["failed_calls"] == 0
    expected = 3 * costs.cost_micros("vision", 400, 100) + 3 * costs.cost_micros("embedding", 12, 0)
    assert job["cost"]["cost_micros"] == expected
    imgs = client.get("/images").json()["images"]
    assert {i["kind"] for i in imgs} == {"fox", "wolf", "dog"}
    assert all(i["kind_prob"] > 0.9 for i in imgs)


def test_a_low_confidence_image_is_flagged_not_accepted(client, seeded, fake):
    fake.kind_p["wolf"] = 0.41
    start(client, "tag_images")
    run_jobs(seeded)
    flagged = client.get("/images?flagged=true").json()["images"]
    assert [i["kind"] for i in flagged] == ["wolf"]
    assert flagged[0]["flag_reason"].startswith("uncertain kind: 'wolf' p=0.41")


def test_invalid_model_output_is_retried_then_failed_never_stored(client, seeded, fake):
    fake.behaviour = "invalid"
    r = start(client, "tag_images")
    for _ in range(3):                       # 3 attempts; skip the backoff wait between them
        seeded.execute("UPDATE job_items SET next_attempt_at = now()")
        run_jobs(seeded)
    job = client.get(f"/jobs/{r.json()['job']['id']}").json()
    assert job["job"]["status"] == "done_with_failures" and job["items"] == {"failed": 3}
    assert "schema validation failed" in job["failures"][0]["last_error"]
    assert seeded.execute("SELECT COUNT(*) AS n FROM image_metadata").fetchone()["n"] == 0
    calls = seeded.execute("SELECT status, COUNT(*) AS n FROM ai_calls GROUP BY status").fetchall()
    assert calls == [{"status": "invalid_output", "n": 9}]       # every bad call is still in the cost log
    alerts = seeded.execute("SELECT kind FROM alerts ORDER BY id").fetchall()
    assert [a["kind"] for a in alerts].count("job_item_failed") == 3


def test_a_model_outage_is_retried_and_recovers(client, seeded, fake):
    fake.behaviour = "down"
    r = start(client, "tag_images")
    run_jobs(seeded)
    assert client.get(f"/jobs/{r.json()['job']['id']}").json()["items"] == {"pending": 3}
    fake.behaviour = "ok"
    seeded.execute("UPDATE job_items SET next_attempt_at = now()")
    run_jobs(seeded)
    assert client.get(f"/jobs/{r.json()['job']['id']}").json()["items"] == {"done": 3}


def test_the_same_idempotency_key_returns_the_same_job(client, seeded, fake):
    a = start(client, "tag_images", key="nightly-2026-10-06")
    b = start(client, "tag_images", key="nightly-2026-10-06")
    assert a.status_code == 202 and b.status_code == 200
    assert a.json()["job"]["id"] == b.json()["job"]["id"] and b.json()["created"] is False
    assert seeded.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"] == 1


def test_a_rerun_only_does_what_is_missing(client, seeded, fake):
    start(client, "tag_images")
    run_jobs(seeded)
    calls = fake.calls
    r = start(client, "tag_images")
    assert r.json()["total"] == 0 and r.json()["job"]["status"] == "done"
    assert fake.calls == calls


def test_the_budget_guard_stops_a_job_before_it_spends(client, seeded, fake, monkeypatch):
    monkeypatch.setitem(config.PRICING["budget"], "max_calls_per_job", 2)
    r = start(client, "tag_images")
    run_jobs(seeded)
    job = client.get(f"/jobs/{r.json()['job']['id']}").json()
    assert job["job"]["status"] == "stopped_budget"
    assert job["cost"]["calls"] == 2 and job["items"] == {"done": 1, "skipped": 2}
    assert seeded.execute("SELECT kind FROM alerts").fetchone()["kind"] == "budget_exceeded"


# --- matching and the guard over HTTP ---------------------------------------------------------------

def index_everything(client, con):
    start(client, "tag_images")
    start(client, "index_posts")
    run_jobs(con)


def test_fox_post_ranks_the_fox_first_and_suggests_it(client, seeded, fake):
    index_everything(client, seeded)
    r = client.get("/posts/1/images").json()
    assert [c["image"]["kind"] for c in r["candidates"]] == ["fox", "wolf", "dog"]
    assert r["status"] == "match" and r["suggestion"]["image"]["kind"] == "fox"
    wolf = r["candidates"][1]
    assert wolf["verdict"] == "rejected" and "expected fox, detected wolf" in wolf["reasons"][0]


def test_forcing_the_wolf_onto_the_fox_post_is_rejected(client, seeded, fake):
    index_everything(client, seeded)
    wolf_id = next(i["id"] for i in client.get("/images").json()["images"] if i["kind"] == "wolf")
    r = client.get(f"/posts/1/images/{wolf_id}/check").json()
    assert r["result"] == "REJECTED"
    assert r["reasons"][0] == "Animal category mismatch: expected fox, detected wolf"


def test_a_post_with_no_suitable_image_says_so(client, seeded, fake):
    index_everything(client, seeded)
    r = client.get("/posts/3/images").json()
    assert r["status"] == "no_confident_match" and r["suggestion"] is None
    assert r["reasons"][0].startswith("Best candidate")
    assert "0 of 3 images passed" in r["reasons"][1]


def test_review_approve_and_reject_are_safe_to_repeat(client, seeded, fake):
    index_everything(client, seeded)
    sid = client.get("/posts/1/images").json()["suggestion"]["suggestion_id"]
    first = client.post(f"/suggestions/{sid}/approve", json={"note": "good"}).json()
    again = client.post(f"/suggestions/{sid}/approve", json={"note": "good"}).json()
    assert first == again                                   # same decision, same timestamp
    assert client.post(f"/suggestions/{sid}/reject").json()["decision"] == "rejected"
    why = client.get(f"/suggestions/{sid}").json()
    assert why["decision"] == "rejected" and why["reasons"][0] == "Subject matches: fox"
    assert "<table>" in client.get("/review").text


# --- validation at the boundary --------------------------------------------------------------------------

def test_bad_input_is_a_clean_4xx(client, seeded):
    assert client.post("/jobs", json={"kind": "delete_everything"}).status_code == 422
    assert client.post("/jobs", json={"kind": "tag_images", "extra": 1}).status_code == 422
    assert client.post("/jobs", json={"kind": "tag_images"}, headers={"Idempotency-Key": "a b"}).status_code == 400
    assert client.get("/posts/abc/images").status_code == 422
    assert client.get("/posts/0/images").status_code == 422
    assert client.get("/posts/999/images").status_code == 404
    assert client.get("/posts/1/images").status_code == 409          # not indexed yet
    assert client.get("/suggestions/999").status_code == 404
    assert client.post("/suggestions/999/approve").status_code == 404
    assert client.post("/suggestions/1/approve", json={"note": "x" * 501}).status_code == 422
    body = client.post("/jobs", json={"kind": "nope"}).json()
    assert body["error"] == "invalid_request" and json.dumps(body)
