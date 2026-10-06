# AI Image Understanding & Content Matching Engine

### [Read the case study →](https://zeref538.github.io/AI-Image-Understanding-and-Content-Matching-Engine/)

Looks at an image library, works out what is in each image, and matches images to blog posts by meaning, not filenames. A red-fox post gets a red-fox photo. A wolf that looks similar is refused with a reason. When nothing fits, it says "no confident match" and explains why.

FlyRank backend track capstone. Python, FastAPI, PostgreSQL, local models through Ollama ($0, no account, no key). Design: [DESIGN.md](DESIGN.md). Proof for every requirement: [EVIDENCE.md](EVIDENCE.md).

## Results

Measured on 6 Oct 2026 with qwen3.5:4b (vision and text) and all-minilm (embeddings), all local. Full output: [docs/eval-run.txt](docs/eval-run.txt), [eval/results.json](eval/results.json).

| | Result |
|---|---|
| **Top-1 precision** (first suggested image is correct) | **13/13** posts that have a correct image (held-out test posts: 6/6) |
| Wrong suggestions | 0 |
| Posts with no suitable image, correctly refused | 3/3 (indoor cats, octopus, sourdough) |
| Ranking alone, before the guard | 13/13 first-ranked images correct |
| Images tagged by the batch job | 50/50, 0 failed calls, 5 flagged |
| Reference cost of the full run (132 calls) | 5,278 micro-dollars (about half a cent); actual cost $0 |

**Read the 13/13 with its sample size:** 16 posts on six well-separated animals. It shows the pipeline works end to end; it does not show it would score 100% on a real blog. The eval set is small enough that one more hard post could move it a lot.

**The fox/wolf boundary, in numbers.** On the 4 fox posts, similarity alone does not separate the animals: 28 of 52 wolf or coyote pairings scored above the weakest real fox image (0.420), and the best wolf or coyote reached 0.677. What refuses them is the guard's kind check, not the threshold. Suggestion precision with the guard: 13/13, with no wolf or coyote ever suggested for a fox post.

**What the vision model got wrong.** It named 3 of the 8 wolves as something else: `img-015` "arctic fox" (p=0.72), `img-040` "fox" (p=0.42, wolf 0.41), `img-050` "white dog" (p=0.75). All three are flagged, so none can be suggested; see the thresholds below.

### How the thresholds were picked

`scripts/eval.py --tune` sweeps the thresholds on the 9 **tune** posts only ([docs/tune-sweep.txt](docs/tune-sweep.txt)).

- **min_similarity = 0.45.** Every value from 0.20 to 0.45 gave 7/7 on the tune posts with no wrong pick; at 0.50 the dog post was missed (its best image scored 0.49). 0.45 is the strictest value that missed nothing.
- **min_kind_prob = 0.80.** The tune posts gave no signal here (their top-ranked images were all confident). The image labels did: the three misnamed wolves scored 0.42, 0.72 and 0.75, and every correctly named image of a lookalike animal scored 0.74 or more. 0.80 flags all three errors at the cost of two correct images (a wolf at 0.74, a coyote at 0.76) held for review. At 0.60, `img-015` (a wolf called "arctic fox") would be eligible for fox posts. This threshold was chosen on all 50 image labels, with no held-out images.
- **min_confidence = 0.50.** The model's own confidence was 0.85 to 0.98 on every image, so this never fires on this corpus. It is kept as a floor, not relied on.

Changing a threshold doesn't need the images tagged again: `python -m app.reflag` (which also runs on every api start) re-applies `config/guard.json` to the stored probabilities.

## Run it

Needs Docker and [Ollama](https://ollama.com) on your machine.

```bash
ollama pull qwen3.5:4b && ollama pull all-minilm       # vision + text model (3.4 GB), embeddings (45 MB)
cp .env.example .env
docker compose up -d --build --wait                    # db + api + worker, API on http://localhost:8001
docker compose exec api python -m app.seed             # 50 images, 16 posts
python scripts/probes.py                               # starts the batch jobs, then runs the 6 acceptance probes
python scripts/eval.py                                 # top-1 precision on the labeled set
docker compose exec api python -m pytest -q            # 33 tests, fake model server, separate test database
```

The batch jobs take about 9 seconds per image on a laptop GPU (measured: average 8.3 s per vision call), so the first run is roughly 8 minutes. You can also start them yourself and watch:

```bash
curl -X POST localhost:8001/jobs -H "Content-Type: application/json" -d '{"kind": "index_posts"}'
curl -X POST localhost:8001/jobs -H "Content-Type: application/json" -d '{"kind": "tag_images"}'
curl localhost:8001/jobs/2                             # progress, failures, cost so far
curl localhost:8001/posts/1/images                     # ranked, guarded suggestions for post 1
open http://localhost:8001/review                      # the review table
```

## Architecture

```
data/images/*.jpg --(POST /jobs tag_images)--> worker
    vision model (qwen3.5:4b) --> JSON --> Pydantic validation --(invalid: retry x3, then fail + alert)
        |  kind_prob from token log-probabilities  -->  flagged if p < 0.80
        v
    image_metadata, image_tags  --embed(subject + caption + tags)-->  embeddings (all-minilm)

data/posts.json --(POST /jobs index_posts)--> worker
    text model reads the post --> {kind, subject, summary} (validated) --embed--> embeddings

GET /posts/:id/images
    cosine(post vector, every image vector)  -->  ranked candidates
    mismatch guard, per candidate:
        image flagged?            --> "Image classification uncertain, held for review: ..."
        kind differs?             --> "Animal category mismatch: expected fox, detected wolf"
        similarity < threshold?   --> "Similarity 0.41 is below the 0.45 threshold"
    first accepted candidate = the suggestion, or "no_confident_match" + reasons
    every candidate is saved as a suggestion --> POST /suggestions/:id/approve | reject

every model call --> ai_calls (tokens, ms, reference cost)  -- budget guard checks before each call
```

| Layer | Files |
|---|---|
| HTTP | `app/main.py` |
| Logic | `app/matching.py`, `app/guard.py`, `app/jobs.py`, `app/costs.py` |
| Model calls | `app/vision.py`, `app/posts.py`, `app/embeddings.py`, all through `app/ollama.py` |
| Data | `app/repo.py`, `app/db.py`, `migrations/*.sql` |
| Background job | `app/worker.py` |

## The corpus

50 photos, all **CC0 or Public Domain Mark**, found through the [Openverse](https://openverse.org) API (most are U.S. National Park Service photos from Yellowstone). Each one was checked by eye before it went in. They are renamed `img-001.jpg` to `img-050.jpg` in shuffled order, so a filename says nothing about what is in it.

| Subject | Images | Why it's there |
|---|---|---|
| red fox | 10 | the brief's target, including one dark "silver" fox |
| gray wolf | 8 | the brief's lookalike |
| coyote | 5 | a second lookalike, between fox and wolf |
| dog | 8 | the "generic dog" that should rank poorly |
| grizzly bear | 8 | |
| mule deer | 8 | |
| ambiguous | 3 | paw prints, bird tracks, a close-up of fur: what flagging is for |

`data/corpus.json` lists each image's source page, creator and licence. `scripts/fetch_corpus.py` re-downloads them.

## The eval set

16 posts in `data/posts.json`. The answers live separately in `eval/post_labels.json`, which the pipeline never reads: 13 posts have a correct subject in the corpus, and 3 (indoor cats, octopus camouflage, sourdough) have none, so the right answer is a refusal. Two posts use only a Latin name ("Vulpes vulpes", "Ursus arctos horribilis").

The posts are split: 9 **tune** posts, which the thresholds were picked from, and 7 **test** posts held out, so the headline number is not only measured on the data it was tuned on.

## Limitations

- **Small eval.** 16 posts, 50 images, six animals that are easy to tell apart in text. Top-1 13/13 is real but would not survive a harder set unchanged.
- **The kind threshold has no held-out data.** It was picked from all 50 image labels.
- **The guard relies on one word.** Posts and images are compared by a one-word `kind` (with a short synonym list). A post about "canids" or "wildlife of Yellowstone" has no single kind and would be refused; a topic taxonomy would fix that.
- **Self-reported confidence is near useless** with this model (0.85 to 0.98 everywhere), which is why flagging uses log-probabilities. A model or server without logprobs falls back to the self-reported number only.
- **Single tenant.** One library, one set of posts, no accounts.
- **Slow on CPU.** About 8 seconds per image on a laptop GPU; much slower without one.
- **Embeddings live in a Postgres array** and ranking is a Python loop over 50 vectors. pgvector is the upgrade past a few thousand images.
