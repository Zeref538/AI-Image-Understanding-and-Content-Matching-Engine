"""Post understanding: what the post is about, in the same words the image side uses.

Measured before this step existed: a raw embedding of "Vulpes vulpes" scored 0.106 against a
red-fox caption with all-minilm, and nomic-embed-text ranked a golden retriever above the fox.
A language model knows the Latin name; the embedding model doesn't. So the post is described
first, then the description is embedded."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app import config, ollama
from app.vision import CATEGORIES, InvalidOutput

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string"},
        "subject": {"type": "string"},
        "category": {"type": "string", "enum": CATEGORIES},
        "summary": {"type": "string"},
    },
    "required": ["kind", "subject", "category", "summary"],
}

PROMPT = (
    "Read this blog post and say what it is mainly about, as JSON.\n"
    "kind: ONE lowercase word for the type of animal or thing, using its everyday English name even if the "
    "post only gives a scientific name (for example fox, wolf, dog, bear, deer, coyote, cat, octopus, bread).\n"
    "subject: the most specific everyday English name (for example 'red fox').\n"
    "category: animal, person, landscape, object or other.\n"
    "summary: one or two sentences describing the ideal photo for this post, using everyday names.\n\n"
    "POST TITLE: {title}\n\nPOST:\n{body}"
)


class PostUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(min_length=2, max_length=30, pattern=r"^[a-z][a-z\-]*$")
    subject: str = Field(min_length=2, max_length=80)
    category: Literal["animal", "person", "landscape", "object", "other"]
    summary: str = Field(min_length=10, max_length=500)

    @field_validator("kind", "subject", mode="before")
    @classmethod
    def tidy(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


def understand(title: str, body: str, attempt: int = 1) -> tuple[PostUnderstanding, ollama.ChatResult]:
    call = ollama.chat(config.TEXT_MODEL, PROMPT.format(title=title, body=body), schema=SCHEMA,
                       temperature=0.0 if attempt == 1 else 0.4)
    try:
        return PostUnderstanding.model_validate_json(call.content), call
    except ValidationError as exc:
        raise InvalidOutput(f"schema validation failed: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}",
                            call) from exc
