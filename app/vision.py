"""Image understanding: one vision call, validated, with a measured confidence."""
import base64
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app import config, ollama
from app.config import Thresholds

CATEGORIES = ["animal", "person", "landscape", "object", "other"]

# Sent to the server as the output format. `kind` comes first so the model commits to
# it before it writes a subject or caption that could steer the answer.
SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"},
        "subject": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "caption": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["kind", "subject", "category", "attributes", "caption", "confidence"],
}

PROMPT = (
    "Look at this photo and describe its main subject as JSON.\n"
    "kind: ONE lowercase word for the type of animal or thing (for example fox, wolf, dog, bear, deer, coyote, "
    "tracks, fur). Name only what you can actually see.\n"
    "subject: the most specific common name you are sure of (for example 'red fox', 'golden retriever').\n"
    "category: animal, person, landscape, object or other.\n"
    "attributes: 3 to 6 short visual tags.\n"
    "caption: one plain sentence.\n"
    "confidence: 0 to 1, how sure you are of kind and subject. Use less than 0.5 when the subject is unclear, "
    "tiny, partly hidden, or could be several different things."
)


class ImageTags(BaseModel):
    """The contract. Anything the model returns that doesn't fit this is never stored."""
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(min_length=2, max_length=30, pattern=r"^[a-z][a-z\-]*$")
    subject: str = Field(min_length=2, max_length=80)
    category: Literal["animal", "person", "landscape", "object", "other"]
    attributes: list[str] = Field(min_length=1, max_length=8)
    caption: str = Field(min_length=10, max_length=400)
    confidence: float = Field(ge=0, le=1)

    @field_validator("kind", "subject", mode="before")
    @classmethod
    def tidy(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("attributes")
    @classmethod
    def tidy_tags(cls, v: list[str]) -> list[str]:
        tags = list(dict.fromkeys(t.strip().lower() for t in v if t.strip()))
        if not tags or any(len(t) > 40 for t in tags):
            raise ValueError("attributes must be 1 to 8 short tags")
        return tags


class InvalidOutput(Exception):
    def __init__(self, message: str, call: ollama.ChatResult):
        super().__init__(message)
        self.call = call


@dataclass
class TagResult:
    tags: ImageTags
    kind_prob: float | None
    alternatives: list[dict]
    flagged: bool
    flag_reason: str | None
    call: ollama.ChatResult


def kind_probability(logprobs: list[dict] | None, kind: str) -> tuple[float | None, list[dict]]:
    """The probability the model gave to the tokens of its `kind` answer, and the runners-up
    for the first of those tokens. This is measured, unlike the self-reported confidence."""
    if not logprobs:
        return None, []
    text, spans = "", []
    for t in logprobs:
        spans.append((len(text), len(text) + len(t["token"]), t))
        text += t["token"]
    m = re.search(r'"kind"\s*:\s*"([^"]*)"', text)
    if not m:
        return None, []
    start, end = m.span(1)
    value_tokens = [t for s, e, t in spans if s < end and e > start]
    if not value_tokens:
        return None, []
    p = math.prod(math.exp(t["logprob"]) for t in value_tokens)
    alternatives = []
    for alt in value_tokens[0].get("top_logprobs", []):
        piece = alt["token"].strip().strip('"')
        label = kind if piece and kind.startswith(piece) else piece   # "co" is the start of "coyote"
        alternatives.append({"kind": label, "p": round(math.exp(alt["logprob"]), 3)})
    return p, alternatives


def flag(tags: ImageTags, kind_prob: float | None, alternatives: list[dict], t: Thresholds) -> str | None:
    if kind_prob is not None and kind_prob < t.min_kind_prob:
        others = ", ".join(f"{a['kind']} {a['p']:.2f}" for a in alternatives if a["kind"] != tags.kind)
        return f"uncertain kind: '{tags.kind}' p={kind_prob:.2f}" + (f" (also: {others})" if others else "")
    if tags.confidence < t.min_confidence:
        return f"model reported low confidence {tags.confidence:.2f}"
    return None


def tag_image(path: Path, thresholds: Thresholds, attempt: int = 1) -> TagResult:
    image_b64 = base64.b64encode(path.read_bytes()).decode()
    prompt = PROMPT
    temperature = 0.0
    if attempt > 1:
        # Same prompt at temperature 0 would give the same broken answer; vary it on a retry.
        prompt += "\nYour previous answer did not match the required JSON. Follow the field rules exactly."
        temperature = 0.4
    call = ollama.chat(config.VISION_MODEL, prompt, images=[image_b64], schema=SCHEMA, logprobs=True,
                       temperature=temperature)
    try:
        tags = ImageTags.model_validate_json(call.content)
    except ValidationError as exc:
        raise InvalidOutput(f"schema validation failed: {exc.error_count()} error(s): "
                            f"{exc.errors()[0]['loc']} {exc.errors()[0]['msg']}", call) from exc
    kind_prob, alternatives = kind_probability(call.logprobs, tags.kind)
    reason = flag(tags, kind_prob, alternatives, thresholds)
    return TagResult(tags, kind_prob, alternatives, reason is not None, reason, call)
