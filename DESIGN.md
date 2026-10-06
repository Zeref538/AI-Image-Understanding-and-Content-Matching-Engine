# Design: AI Image Understanding & Content Matching Engine

## Problem

Given an image library and a set of blog posts, suggest the right image for each post from what the images show, not from filenames. A red-fox post gets a red-fox photo. A wolf that looks similar is refused with a reason. When nothing fits, the system says "no confident match" instead of guessing.

## Image metadata schema

The vision model must return exactly this, validated with Pydantic before anything trusts it:

```json
{
  "kind": "fox",
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "snow", "alert"],
  "caption": "A red fox standing in the snow.",
  "confidence": 0.94
}
```

`kind` is one lowercase word (fox, wolf, dog), asked for **first**, so the model commits to it before writing anything that could steer it. `confidence` is the model's own guess, which small models overstate, so it is not trusted alone: the request also asks for **token log-probabilities**, and the probability the model gave its `kind` answer is stored as `kind_prob`, with the runner-up answers ("coyote 0.53, fox 0.23, wolf 0.07").

An image is **flagged**, not accepted, when `kind_prob < 0.60` or `confidence < 0.50`, or when the output fails validation twice.

## Matching strategy

1. **Images:** vision model → validated tags → embed `subject + caption + attributes`.
2. **Posts:** the same treatment. A text call reads the post and returns `{kind, subject, category, summary}`, then `subject + summary` is embedded. This step exists because a raw embedding does not know that "Vulpes vulpes" is a fox (measured: all-minilm 0.106, nomic-embed-text ranked a retriever above the fox).
3. **Ranking:** cosine similarity between the post vector and every image vector.

## Guard rules (a candidate must pass all of them)

| Check | Refusal text |
|---|---|
| image analysed and not flagged | "Image classification uncertain: coyote (p=0.53) vs fox (0.23); held for review" |
| `kind` matches the post's `kind` | "Animal category mismatch: expected fox, detected wolf" |
| `category` matches | "Category mismatch: expected animal, detected object" |
| similarity ≥ threshold | "Similarity 0.41 is below the 0.55 threshold" |

The thresholds are picked from a labeled eval set, not guessed, and pinned in `config/guard.json`.

## Database

`images`, `image_metadata`, `image_tags`, `posts`, `post_understanding`, `embeddings` (`float8[]`, fine at 50 images), `suggestions` (`UNIQUE (post_id, image_id)`, index `(post_id, rank)`), `reviews`, `jobs`, `job_items` (`UNIQUE (job_id, target_type, target_id)`), `ai_calls` (the cost log), `alerts`.

## API surface

| Method | Path | Does |
|---|---|---|
| POST | `/jobs` | start a batch job (`tag_images`, `index_posts`); `Idempotency-Key` makes a retried start return the same job |
| GET | `/jobs/{id}` | progress, failures, cost so far |
| GET | `/images`, `/images/{id}` | metadata, tags, flag reason |
| GET | `/posts`, `/posts/{id}` | post and what the model understood |
| GET | `/posts/{id}/images` | ranked candidates, each with the guard's verdict, and the suggestion or "no confident match" |
| GET | `/posts/{id}/images/{image_id}/check` | force one pairing through the guard (the wolf on the fox post) |
| POST | `/suggestions/{id}/approve`, `/reject` | the human review |
| GET | `/suggestions/{id}` | why it was picked or refused |
| GET | `/review` | a plain HTML table of every suggestion |
| GET | `/costs` | the per-call cost log and totals |

## Layers

`app/main.py` (HTTP) → `app/matching.py`, `app/guard.py`, `app/jobs.py` (logic) → `app/vision.py`, `app/posts.py`, `app/embeddings.py` (model calls, all through `app/ollama.py`) → `app/repo.py` (SQL). `app/worker.py` runs the jobs.

## Non-goal

No image upload or storage service. The corpus is a fixed folder of 50 committed images; adding images means adding files and re-running the job.
