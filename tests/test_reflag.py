from conftest import run_jobs

from app import config
from app.config import Thresholds
from app.reflag import reflag


def test_changing_a_threshold_reflags_without_calling_the_model(client, seeded, fake, monkeypatch):
    fake.kind_p["wolf"] = 0.72                       # the real misnamed-wolf case
    client.post("/jobs", json={"kind": "tag_images"})
    run_jobs(seeded)
    calls = fake.calls
    monkeypatch.setattr(config, "load_thresholds", lambda: Thresholds(0.45, 0.6, 0.5))
    assert reflag(seeded) == 0                       # 0.72 passes a 0.60 bar
    monkeypatch.setattr(config, "load_thresholds", lambda: Thresholds(0.45, 0.8, 0.5))
    assert reflag(seeded) == 1                       # and is flagged at 0.80
    assert fake.calls == calls                       # no model call was needed
