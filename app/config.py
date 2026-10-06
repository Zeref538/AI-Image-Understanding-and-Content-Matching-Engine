import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # never overrides a variable that is already set

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/images")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3.5:4b")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "qwen3.5:4b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-minilm")
MODEL_TIMEOUT_S = float(os.environ.get("MODEL_TIMEOUT_S", "120"))


@dataclass(frozen=True)
class Thresholds:
    min_similarity: float
    min_kind_prob: float
    min_confidence: float


def load_thresholds() -> Thresholds:
    raw = json.loads((ROOT / "config" / "guard.json").read_text(encoding="utf-8"))
    return Thresholds(raw["min_similarity"], raw["min_kind_prob"], raw["min_confidence"])


PRICING = json.loads((ROOT / "config" / "pricing.json").read_text(encoding="utf-8"))
