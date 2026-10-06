"""Schema validation, the measured confidence, the flag rule and the guard. No database."""
import json
import math

import pytest
from pydantic import ValidationError

from app.config import Thresholds
from app.guard import Verdict, check, no_match_summary, normalize_kind
from app.vision import ImageTags, flag, kind_probability

T = Thresholds(min_similarity=0.5, min_kind_prob=0.6, min_confidence=0.5)
GOOD = {"kind": "fox", "subject": "red fox", "category": "animal", "attributes": ["orange fur", "snow"],
        "caption": "A red fox standing in the snow.", "confidence": 0.94}


# --- schema validation: invalid model output is never trusted ------------------------------

def test_valid_output_is_accepted_and_tidied():
    tags = ImageTags.model_validate({**GOOD, "kind": " Fox ", "attributes": ["Snow", "snow ", "orange fur"]})
    assert tags.kind == "fox" and tags.attributes == ["snow", "orange fur"]


@pytest.mark.parametrize("bad", [
    {**GOOD, "kind": "red fox"},              # two words
    {**GOOD, "kind": "Fox!!"},
    {**GOOD, "confidence": 1.5},
    {**GOOD, "category": "animals"},          # not in the enum
    {**GOOD, "attributes": []},
    {**GOOD, "caption": "fox"},               # too short to be a sentence
    {**GOOD, "surprise": True},               # unknown field
    {k: v for k, v in GOOD.items() if k != "subject"},
])
def test_invalid_output_is_refused(bad):
    with pytest.raises(ValidationError):
        ImageTags.model_validate(bad)


def test_non_json_is_refused():
    with pytest.raises(ValidationError):
        ImageTags.model_validate_json("Sure! Here is a fox: {kind: fox}")


# --- measured confidence ---------------------------------------------------------------------

def logprobs_for(kind: str, pieces: list[tuple[str, float]], alts: list[tuple[str, float]]):
    content = json.dumps({**GOOD, "kind": kind})
    head, tail = content.split(json.dumps(kind), 1)
    toks = [{"token": head + '"', "logprob": 0.0}]
    for i, (piece, p) in enumerate(pieces):
        t = {"token": piece, "logprob": math.log(p)}
        if i == 0:
            t["top_logprobs"] = [{"token": a, "logprob": math.log(q)} for a, q in alts]
        toks.append(t)
    toks.append({"token": '"' + tail, "logprob": 0.0})
    return toks


def test_kind_probability_reads_the_real_coyote_case():
    # Real numbers from qwen3.5:4b on a coyote photo: 'co' 0.53, 'fox' 0.23, 'wolf' 0.07, then 'y','ote' ~1.0
    lp = logprobs_for("coyote", [("co", 0.53), ("y", 0.999), ("ote", 0.997)],
                      [("co", 0.53), ("fox", 0.23), ("wolf", 0.07)])
    p, alts = kind_probability(lp, "coyote")
    assert p == pytest.approx(0.53 * 0.999 * 0.997, rel=1e-6)
    assert alts == [{"kind": "coyote", "p": 0.53}, {"kind": "fox", "p": 0.23}, {"kind": "wolf", "p": 0.07}]


def test_no_logprobs_means_no_measured_probability():
    assert kind_probability(None, "fox") == (None, [])


def test_flag_uses_the_measured_probability_not_just_the_claim():
    tags = ImageTags.model_validate({**GOOD, "kind": "coyote", "confidence": 0.95})   # the model claims 0.95
    reason = flag(tags, 0.53, [{"kind": "coyote", "p": 0.53}, {"kind": "fox", "p": 0.23}], T)
    assert reason == "uncertain kind: 'coyote' p=0.53 (also: fox 0.23)"
    assert flag(ImageTags.model_validate(GOOD), 0.97, [], T) is None
    assert flag(ImageTags.model_validate({**GOOD, "confidence": 0.3}), 0.97, [], T).startswith("model reported low")


# --- the guard ------------------------------------------------------------------------------------

FOX_POST = {"kind": "fox", "category": "animal"}


def image(kind, flagged=False, reason=None, category="animal", p=0.95, file="img-001.jpg"):
    return {"kind": kind, "category": category, "flagged": flagged, "flag_reason": reason, "kind_prob": p, "file": file}


def test_the_wolf_on_a_fox_post_is_rejected_with_the_reason():
    v = check(FOX_POST, image("wolf"), 0.81, T)      # similar-looking: high similarity is not enough
    assert v.accepted is False
    assert v.reasons == ["Animal category mismatch: expected fox, detected wolf"]


def test_a_matching_confident_similar_image_is_accepted():
    v = check(FOX_POST, image("fox"), 0.72, T)
    assert v.accepted and v.reasons[0] == "Subject matches: fox"


def test_low_similarity_is_rejected_even_when_the_kind_matches():
    v = check(FOX_POST, image("fox"), 0.31, T)
    assert not v.accepted and v.reasons == ["Similarity 0.310 is below the 0.50 threshold"]


def test_a_flagged_image_is_held_for_review():
    v = check(FOX_POST, image("fox", flagged=True, reason="uncertain kind: 'fox' p=0.41"), 0.9, T)
    assert not v.accepted and v.reasons[0].startswith("Image classification uncertain")


def test_an_unanalysed_image_is_never_accepted():
    assert check(FOX_POST, None, 0.99, T).accepted is False


def test_plural_and_synonym_kinds_still_match():
    assert normalize_kind("Foxes") == "fox" and normalize_kind("wolves") == "wolf" and normalize_kind("puppy") == "dog"
    assert check({"kind": "foxes", "category": "animal"}, image("fox"), 0.8, T).accepted


def test_no_match_summary_explains_the_best_candidate_and_the_counts():
    verdicts = [(image("coyote", file="img-012.jpg"), 0.41, Verdict(False, [
                    "Animal category mismatch: expected octopus, detected coyote",
                    "Similarity 0.410 is below the 0.50 threshold"])),
                (image("fox", file="img-003.jpg"), 0.30, Verdict(False, [
                    "Animal category mismatch: expected octopus, detected fox"]))]
    lines = no_match_summary(verdicts)
    assert lines[0].startswith("Best candidate img-012.jpg (coyote, similarity 0.410) was refused")
    assert lines[1] == "0 of 2 images passed: 2 subject mismatch, 1 below similarity threshold, 0 uncertain classification"
