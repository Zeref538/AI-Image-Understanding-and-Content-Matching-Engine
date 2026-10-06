import math

from app import config, ollama


def image_text(subject: str, caption: str, attributes: list[str]) -> str:
    return f"{subject}. {caption} {', '.join(attributes)}."


def post_text(subject: str, summary: str) -> str:
    return f"{subject}. {summary}"


def embed_one(text: str, as_query: bool) -> tuple[list[float], int, int]:
    # nomic-embed-text was trained with these prefixes; all-minilm uses none
    if "nomic" in config.EMBED_MODEL:
        text = ("search_query: " if as_query else "search_document: ") + text
    vectors, tokens, ms = ollama.embed(config.EMBED_MODEL, [text])
    return vectors[0], tokens, ms


def cosine(a: list[float], b: list[float]) -> float:
    # ponytail: plain Python over 50 vectors; numpy or pgvector if the library grows
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
