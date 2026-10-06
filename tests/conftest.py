"""Tests use a real Postgres (a separate images_test database, rebuilt every run) and a FAKE
model server, so they run in seconds with no Ollama and no network."""
import base64
import hashlib
import json
import math
import os

import psycopg
from psycopg.conninfo import make_conninfo

from app import config

TEST_DB = "images_test"
_base = os.environ.get("DATABASE_URL", config.DATABASE_URL)
TEST_URL = make_conninfo(_base, dbname=TEST_DB)
config.DATABASE_URL = TEST_URL

import pytest  # noqa: E402

from app import db, ollama  # noqa: E402

FILES = {"fox": "img-048.jpg", "wolf": "img-050.jpg", "dog": "img-049.jpg"}   # a real fox, wolf and dog
FAKE_TAGS = {
    "fox": {"kind": "fox", "subject": "red fox", "category": "animal", "attributes": ["orange fur", "rain"],
            "caption": "A red fox standing in the rain.", "confidence": 0.95},
    "wolf": {"kind": "wolf", "subject": "arctic wolf", "category": "animal", "attributes": ["white fur", "snow"],
             "caption": "A white wolf on open ground.", "confidence": 0.93},
    "dog": {"kind": "dog", "subject": "bernese mountain dog", "category": "animal", "attributes": ["tricolor"],
            "caption": "A Bernese mountain dog lying down.", "confidence": 0.97},
}
# Hand-made 3-d "embeddings": fox and wolf close-ish, dog further, octopus nowhere near.
VECTORS = {"fox": [1.0, 0.2, 0.0], "wolf": [0.7, 0.7, 0.0], "dog": [0.3, 0.9, 0.1], "octopus": [0.0, 0.0, 1.0]}


def pytest_sessionstart(session):
    with psycopg.connect(make_conninfo(_base, dbname="postgres"), autocommit=True) as con:
        con.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        con.execute(f"CREATE DATABASE {TEST_DB}")
    with db.connect(TEST_URL) as con:
        db.migrate(con)


def tokens_for(content: str, kind: str, kind_p: float) -> list[dict]:
    """Logprobs shaped like the real server's: the kind value is one token with probability kind_p."""
    head, tail = content.split(json.dumps(kind), 1)
    return [{"token": head + '"', "logprob": 0.0},
            {"token": kind, "logprob": math.log(kind_p),
             "top_logprobs": [{"token": kind, "logprob": math.log(kind_p)},
                              {"token": "dog", "logprob": math.log(max(1e-6, (1 - kind_p) / 2)) }]},
            {"token": '"' + tail, "logprob": 0.0}]


class FakeModels:
    """Stands in for app.ollama. Tests change .behaviour to make it misbehave."""

    def __init__(self):
        root = config.IMAGES_DIR
        self.by_image = {base64.b64encode((root / f).read_bytes()).decode(): k for k, f in FILES.items()}
        self.kind_p = {"fox": 0.97, "wolf": 0.95, "dog": 0.99}
        self.behaviour = "ok"            # ok | invalid | down
        self.calls = 0

    def chat(self, model, prompt, *, images=None, schema=None, logprobs=False, temperature=0.0):
        self.calls += 1
        if self.behaviour == "down":
            raise ollama.ModelError("ConnectError: connection refused")
        if images:
            key = self.by_image[images[0]]
            tags = dict(FAKE_TAGS[key])
            content = json.dumps(tags) if self.behaviour != "invalid" else '{"kind": "Red Fox!!", "confidence": 7}'
            lp = tokens_for(content, tags["kind"], self.kind_p[key]) if self.behaviour == "ok" else None
            return ollama.ChatResult(content, 400, 100, 5, lp)
        post = prompt.split("POST TITLE:", 1)[1].lower()     # the instructions name animals as examples
        kind = next((k for k in ("fox", "wolf", "dog", "octopus") if k in post), "octopus")
        subject = {"fox": "red fox", "wolf": "gray wolf", "dog": "dog", "octopus": "octopus"}[kind]
        content = json.dumps({"kind": kind, "subject": subject, "category": "animal",
                              "summary": f"A photo of a {subject}."})
        return ollama.ChatResult(content, 300, 40, 5, None)

    def embed(self, model, texts):
        self.calls += 1
        if self.behaviour == "down":
            raise ollama.ModelError("ConnectError: connection refused")
        text = texts[0].lower()
        kind = next((k for k in ("fox", "wolf", "dog", "octopus") if k in text), "octopus")
        return [VECTORS[kind]], 12, 2


@pytest.fixture
def fake(monkeypatch):
    f = FakeModels()
    monkeypatch.setattr(ollama, "chat", f.chat)
    monkeypatch.setattr(ollama, "embed", f.embed)
    return f


@pytest.fixture
def con():
    with db.connect(TEST_URL) as c:
        c.execute("TRUNCATE images, posts, embeddings, jobs, ai_calls, alerts RESTART IDENTITY CASCADE")
        yield c


@pytest.fixture
def seeded(con):
    """Three real images (fox, wolf, dog) and three posts."""
    from app import repo
    for f in FILES.values():
        repo.upsert_image(con, f, f, "pdm 1.0", "", hashlib.sha256((config.IMAGES_DIR / f).read_bytes()).hexdigest())
    repo.upsert_post(con, "foxes", "The behavior of red foxes", "All about the fox.")
    repo.upsert_post(con, "wolves", "Wolf packs", "All about the wolf.")
    repo.upsert_post(con, "octopus", "Octopus camouflage", "All about the octopus.")
    return con


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


def run_jobs(con) -> None:
    from app import worker
    while worker.run_once(con):
        pass
