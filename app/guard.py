"""The mismatch guard: is this candidate actually good enough? Pure functions, no I/O.

A candidate must pass every check. Each refusal says which check failed and why, in words
a reviewer can act on."""
from dataclasses import dataclass, field

from app.config import Thresholds

# Small, explicit, and only for words the models were seen to use. Not a taxonomy.
SYNONYMS = {"wolves": "wolf", "foxes": "fox", "dogs": "dog", "bears": "bear", "coyotes": "coyote",
            "puppy": "dog", "hound": "dog", "grizzly": "bear", "buck": "deer", "doe": "deer", "fawn": "deer",
            "kit": "fox", "vixen": "fox", "cats": "cat", "kitten": "cat", "octopuses": "octopus"}


def normalize_kind(kind: str | None) -> str:
    k = (kind or "").strip().lower()
    return SYNONYMS.get(k, k)


@dataclass
class Verdict:
    accepted: bool
    reasons: list[str] = field(default_factory=list)


def check(post: dict, image: dict | None, similarity: float, t: Thresholds) -> Verdict:
    """post: kind, category. image: kind, category, flagged, flag_reason (None = not analysed)."""
    if image is None or image.get("kind") is None:
        return Verdict(False, ["Image not analysed yet: run the tag_images job"])
    reasons = []
    if image["flagged"]:
        reasons.append(f"Image classification uncertain, held for review: {image['flag_reason']}")
    expected, detected = normalize_kind(post["kind"]), normalize_kind(image["kind"])
    if expected != detected:
        noun = "Animal category" if post["category"] == "animal" == image["category"] else "Subject"
        reasons.append(f"{noun} mismatch: expected {expected}, detected {detected}")
    if post["category"] != image["category"]:
        reasons.append(f"Category mismatch: expected {post['category']}, detected {image['category']}")
    if similarity < t.min_similarity:
        reasons.append(f"Similarity {similarity:.3f} is below the {t.min_similarity:.2f} threshold")
    if reasons:
        return Verdict(False, reasons)
    return Verdict(True, [f"Subject matches: {detected}",
                          f"Similarity {similarity:.3f} meets the {t.min_similarity:.2f} threshold",
                          f"Classification confident (kind p={image['kind_prob']:.2f})" if image.get("kind_prob")
                          is not None else "Classification confident"])


def no_match_summary(verdicts: list[tuple[dict, float, Verdict]]) -> list[str]:
    """Why nothing cleared the bar, from the best-ranked candidates down."""
    if not verdicts:
        return ["No analysed images to compare against: run the tag_images job"]
    image, sim, v = verdicts[0]
    lines = [f"Best candidate {image['file']} ({image.get('kind')}, similarity {sim:.3f}) was refused: "
             + "; ".join(v.reasons)]
    counts = {"subject mismatch": 0, "below similarity threshold": 0, "uncertain classification": 0}
    for _, _, verdict in verdicts:
        text = " ".join(verdict.reasons)
        counts["subject mismatch"] += "mismatch: expected" in text
        counts["below similarity threshold"] += "below the" in text
        counts["uncertain classification"] += "uncertain" in text
    lines.append(f"0 of {len(verdicts)} images passed: " + ", ".join(f"{n} {what}" for what, n in counts.items()))
    return lines
