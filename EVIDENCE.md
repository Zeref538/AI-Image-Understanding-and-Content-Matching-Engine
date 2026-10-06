# Evidence

One proof per requirement in Section 6 of the brief. Everything was run on 6 Oct 2026 against the Docker stack with the real local models (qwen3.5:4b, all-minilm). Full outputs: [docs/probes-run.txt](docs/probes-run.txt) (the 6 acceptance probes), [docs/eval-run.txt](docs/eval-run.txt), [docs/test-run.txt](docs/test-run.txt) (33 tests inside the api container), [docs/review-transcript.txt](docs/review-transcript.txt), [docs/schema.txt](docs/schema.txt), [docs/tune-sweep.txt](docs/tune-sweep.txt).

## AI processing

- [x] **Vision model produces structured output validated against a schema; invalid responses are never trusted.**
  `app/vision.py` `ImageTags` (Pydantic, `extra="forbid"`, one-word `kind`, enum `category`, `confidence` in 0..1). Probe 1: `PASS  P1 every image has schema-valid tags (50/50)`. `test_invalid_output_is_refused` (8 malformed shapes) and `test_invalid_model_output_is_retried_then_failed_never_stored PASSED`: 3 attempts, then the item fails with an alert, `image_metadata` stays empty, and all 9 bad calls are still in the cost log as `invalid_output`.

- [x] **Low-confidence classifications are flagged instead of accepted.**
  ```
  PASS  P1 at least one low-confidence image is flagged, not guessed (5 flagged)
        flagged img-040.jpg (gray wolf): uncertain kind: 'fox' p=0.42 (also: wolf 0.41, animal 0.09)
        flagged img-015.jpg (gray wolf): uncertain kind: 'fox' p=0.72 (also: wolf 0.17, animal 0.08)
        flagged img-050.jpg (gray wolf): uncertain kind: 'dog' p=0.75 (also: wolf 0.18, animal 0.03)
  ```
  The probability is measured from the model's token log-probabilities, not its self-reported confidence (which was 0.95 on all three). Flagged images are never suggested: `test_a_flagged_image_is_held_for_review PASSED`.

- [x] **Images are processed through a batch background job with retries.**
  `POST /jobs {"kind": "tag_images"}` returns 202; `app/worker.py` processes items with a lease, so no model call runs inside a request or an open transaction. Job 2 on the real run: `tag_images done {'done': 49, 'flagged': 1} calls 100 failed_calls 0` (before the final thresholds re-flagged 5). `test_a_model_outage_is_retried_and_recovers PASSED` (server down, items wait and retry, then finish), `test_the_same_idempotency_key_returns_the_same_job PASSED`.

- [x] **Vision and embedding costs are tracked per call.**
  Probe 6: `PASS  P6 every vision/embedding call is in the cost log with a cost (132 calls, at least 132)`
  ```
  embedding           all-minilm     66 calls     2156 in      0 out     433 micro-$  avg 51 ms
  post_understanding  qwen3.5:4b     16 calls     2951 in    965 out     682 micro-$  avg 4334 ms
  vision              qwen3.5:4b     50 calls    21740 in   4970 out    4163 micro-$  avg 8160 ms
  ```
  One row from `GET /costs`: `{"id": 132, "job_id": 2, "purpose": "embedding", "target_type": "image", "target_id": 50, "model": "all-minilm", "input_tokens": 25, "duration_ms": 16, "cost_micros": 5, "status": "ok"}`. Prices are Gemini's paid-tier rates (pinned in `config/pricing.json` with the source); the local run cost $0. The budget guard: `test_the_budget_guard_stops_a_job_before_it_spends PASSED`.

## Matching system

- [x] **Image and post embeddings are stored; posts return ranked image suggestions.**
  Table `embeddings` (`float8[]`, primary key `owner_type, owner_id, model`). Probe 2: `PASS  P2 the first-ranked image for the red fox post is a red fox` and `PASS  P2 wolf and dog rank clearly lower (first wolf at #7, first dog at #26)`. Transcript in [docs/review-transcript.txt](docs/review-transcript.txt).

- [x] **Semantic matching works for equivalent concepts: "red fox" matches "Vulpes vulpes".**
  The post "Vulpes vulpes: a field guide" never says "fox". `GET /posts/2`: `{"kind": "fox", "subject": "red fox", "summary": "A photo of a red fox with a white-tipped tail, black legs, and narrow muzzle..."}`. Its suggestion: `img-006.jpg red fox, similarity 0.8512`. Same for "Ursus arctos horribilis" (picked `img-028.jpg`, a grizzly). Why the understanding step exists: a raw embedding of "Vulpes vulpes" scored 0.106 against a red-fox caption (BUILDLOG.md, item 3).

## Safety layer

- [x] **The mismatch guard rejects incorrect recommendations: the wolf-on-a-fox-post scenario provably fails.**
  ```
  PASS  P3 the wolf (img-012.jpg) forced onto the fox post is REJECTED with a category mismatch
        reasons: ['Animal category mismatch: expected fox, detected wolf', 'Similarity 0.421 is below the 0.45 threshold']
  ```
  `test_the_wolf_on_a_fox_post_is_rejected_with_the_reason PASSED` (a wolf at similarity 0.81 is still refused). Across the eval, no wolf or coyote was suggested for any of the 4 fox posts, although 28 of 52 such pairings scored above the weakest real fox.

- [x] **Rejections include a human-readable explanation.**
  Every candidate carries `reasons`, e.g. `"Animal category mismatch: expected fox, detected wolf"`, `"Similarity 0.300 is below the 0.45 threshold"`, `"Image classification uncertain, held for review: uncertain kind: 'dog' p=0.75 (also: wolf 0.18, animal 0.03)"`.

- [x] **When no image clears the bar, the system answers "no confident match" with reasons.**
  ```
  PASS  P4 the octopus post answers no_confident_match with reasons
        Best candidate img-050.jpg (dog, similarity 0.300) was refused: ...; Animal category mismatch:
        expected octopus, detected dog; Similarity 0.300 is below the 0.45 threshold
        0 of 50 images passed: 50 subject mismatch, 50 below similarity threshold, 5 uncertain classification
  ```
  All 3 no-image posts (indoor cats, octopus, sourdough) were refused in the eval.

## Backend

- [x] **Database models for images, tags, embeddings, posts, suggestions, approvals/rejections, with the required indexes.**
  From [docs/schema.txt](docs/schema.txt): `images`, `image_metadata`, `image_tags` (index on `tag`), `embeddings`, `posts`, `post_understanding`, `suggestions` (`UNIQUE (post_id, image_id)`, index `(post_id, rank)`), `reviews`, `jobs` (`UNIQUE idempotency_key`), `job_items` (`UNIQUE (job_id, target_type, target_id)`, partial index on due items), `ai_calls` (indexes on job, target, day), `alerts`. Schema as migrations: `migrations/001_init.sql`.

- [x] **API endpoints validated; the review workflow (approve / reject / inspect why) exists.**
  From [docs/review-transcript.txt](docs/review-transcript.txt): approving suggestion 1 twice returns the same decision and the same `reviewed_at`; `GET /suggestions/1` shows rank, similarity, verdict, reasons and the decision; `POST /jobs {"kind": "resize_everything"}` is a 422 with the field named; `GET /posts/999/images` is a 404. `GET /review` renders every post's top suggestions as an HTML table. `test_bad_input_is_a_clean_4xx PASSED` (10 bad inputs, no 500).

## Quality & documentation

- [x] **A small labeled evaluation dataset measures top-1 precision; the number is in the README.**
  ```
  top-1 precision: 13/13 = 1.0   (held-out test split: 6/6 = 1.0)
  suggestions made: 13, precision 1.0, wrong: none, missed: none
  posts with no suitable image correctly refused: 3/3
  PASS  P5 eval reports top-1 precision 13/13, and the README states the same number
  ```
  Labels: `eval/post_labels.json` (16 posts, 9 tune / 7 test), `eval/image_labels.json` (50 images, checked by eye). The README states the sample size and the limits of a perfect score on 16 posts.

- [x] **README with architecture explanation and diagram; the required files present.**
  `README.md` (diagram, run + seed, how the thresholds were picked, limitations), `capstone.yaml`, `EVIDENCE.md`, `BUILDLOG.md`, `.env.example`, `DESIGN.md`, `LICENSE`.

## Shared requirements

| # | Requirement | Where |
|---|---|---|
| 1 | Layered architecture | `main.py` (HTTP) / `matching.py`, `guard.py`, `jobs.py`, `costs.py` (logic) / `vision.py`, `posts.py`, `embeddings.py`, `ollama.py` (model calls) / `repo.py` (data) |
| 2 | Bad input is a clean 4xx | `test_bad_input_is_a_clean_4xx PASSED`; transcript above |
| 3 | Background job with retries and a failure alert | `app/worker.py`, `app/jobs.py`: 3 attempts with backoff, then `failed` + an `alerts` row + an ERROR log; `done_with_failures` jobs alert too |
| 4 | Schema as migrations, indexes | `migrations/001_init.sql`; single tenant (see README limitations) |
| 5 | Idempotency | `Idempotency-Key` on `POST /jobs`; re-runs only process what is missing; approve/reject are safe to repeat |
| 6 | Secrets clean | No keys exist: models run locally. `.env` is git-ignored and docker-ignored |
| 7 | Cost tracked, with a budget guard | `ai_calls` per call; `costs.check_budget` runs before every call (per-job call cap, daily cost cap) |
