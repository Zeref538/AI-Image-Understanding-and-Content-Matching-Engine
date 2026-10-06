"""The brief's six acceptance probes, against the live stack over HTTP.

    docker compose up -d --build --wait
    docker compose exec api python -m app.seed
    python scripts/probes.py          # starts the batch jobs if needed (about 10 minutes on a laptop GPU)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
BASE = os.environ.get("EVAL_BASE_URL", "http://localhost:8001")
http = httpx.Client(base_url=BASE, timeout=120)
labels = json.loads((ROOT / "eval" / "image_labels.json").read_text())
failed = 0


def check(name: str, ok: bool, detail="") -> None:
    global failed
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"  {detail}"))
    failed += not ok


def run_job(kind: str) -> dict:
    job = http.post("/jobs", json={"kind": kind}).json()
    jid = job["job"]["id"]
    while (p := http.get(f"/jobs/{jid}").json())["job"]["status"] in ("queued", "running"):
        print(f"      {kind}: {p['items']}", end="\r")
        time.sleep(5)
    return p


def post_id(slug: str) -> int:
    return next(p["id"] for p in http.get("/posts").json()["posts"] if p["slug"] == slug)


print(f"probing {BASE}\n")

# PROBE 1: the batch job tags every image with schema-valid tags; at least one is flagged
run_job("index_posts")
job = run_job("tag_images")
images = http.get("/images").json()["images"]
valid = [i for i in images if i["kind"] and i["subject"] and i["category"] and i["caption"] and i["tags"]]
flagged = [i for i in images if i["flagged"]]
check(f"P1 every image has schema-valid tags ({len(valid)}/{len(images)})", len(valid) == len(images) == 50)
check(f"P1 at least one low-confidence image is flagged, not guessed ({len(flagged)} flagged)", len(flagged) >= 1)
for i in flagged:
    print(f"      flagged {i['file']} ({labels[i['file']]['subject']}): {i['flag_reason']}")

# PROBE 2: the red fox article ranks the fox first; wolf and dog clearly lower
fox_post = post_id("red-fox-behavior")
r = http.get(f"/posts/{fox_post}/images", params={"top": 50}).json()
truth = [labels[c["image"]["file"]]["subject"] for c in r["candidates"]]
first_wolf = truth.index("gray wolf") + 1
first_dog = truth.index("dog") + 1
check("P2 the first-ranked image for the red fox post is a red fox", truth[0] == "red fox", truth[:5])
check(f"P2 wolf and dog rank clearly lower (first wolf at #{first_wolf}, first dog at #{first_dog})",
      first_wolf > 3 and first_dog > 3)
check("P2 the suggestion is a fox", r["status"] == "match" and labels[r["suggestion"]["image"]["file"]]["subject"] == "red fox")

# PROBE 3: force the wolf onto the fox post
wolf = next(i for i in images if labels[i["file"]]["subject"] == "gray wolf" and not i["flagged"])
forced = http.get(f"/posts/{fox_post}/images/{wolf['id']}/check").json()
check(f"P3 the wolf ({wolf['file']}) forced onto the fox post is REJECTED with a category mismatch",
      forced["result"] == "REJECTED" and any("mismatch: expected fox, detected wolf" in x for x in forced["reasons"]),
      forced)
print(f"      reasons: {forced['reasons']}")

# PROBE 4: a post with no suitable image
octo = http.get(f"/posts/{post_id('octopus-camouflage')}/images").json()
check("P4 the octopus post answers no_confident_match with reasons",
      octo["status"] == "no_confident_match" and len(octo["reasons"]) >= 2, octo.get("status"))
for line in octo.get("reasons", []):
    print(f"      {line}")

# PROBE 5: the eval script
out = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval.py")], capture_output=True, text=True)
print("      " + out.stdout.strip().replace("\n", "\n      "))
readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
result = json.loads((ROOT / "eval" / "results.json").read_text())["all"]
headline = f"{result['top1_correct']}/{result['posts_with_a_correct_image']}"
check(f"P5 eval reports top-1 precision {headline}, and the README states the same number", headline in readme,
      "README does not contain " + headline)

# PROBE 6: every model call has a cost entry
costs = http.get("/costs").json()
calls = sum(p["calls"] for p in costs["by_purpose"])
expected_min = 2 * 50 + 2 * 16          # vision + embedding per image, understanding + embedding per post
check(f"P6 every vision/embedding call is in the cost log with a cost ({calls} calls, at least {expected_min})",
      calls >= expected_min and all(c["cost_micros"] is not None for c in costs["recent"]))
for p in costs["by_purpose"]:
    print(f"      {p['purpose']:<19} {p['model']:<12} {p['calls']:>4} calls  {p['input_tokens']:>7} in "
          f"{p['output_tokens']:>6} out  {p['cost_micros']:>6} micro-$  avg {p['avg_ms']} ms")

print(f"\n{'all probes passed' if not failed else f'{failed} check(s) failed'}")
sys.exit(1 if failed else 0)
