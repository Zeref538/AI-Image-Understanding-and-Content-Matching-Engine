# Build log

I built this with an AI coding assistant. It drafted most of the code; I set the rules, read every file, picked every image by eye, and let the tests, the eval and the probes decide what stayed. This log records where it, or the models, were wrong.

## Where it helped

- Drafting the schema, the job runner, the guard and the HTTP layer from DESIGN.md.
- A fake model server for the tests, so the whole pipeline is tested in seconds without Ollama.
- The eval and probe scripts.

## Where things were wrong, and what changed

1. **Unsplash and Pexels refused scripted access.** Unsplash answered with a "making sure you're not a bot" page, Pexels with 403. I did not try to get around either. The corpus comes from Openverse instead, filtered to CC0 and Public Domain Mark only, which is freer than the Unsplash licence. Every image's source and licence is in `data/corpus.json`.

2. **Search results were not what their titles said.** The "red fox" search returned four event-decoration photos from a hall named after a fox. The "gray wolf" search returned forestry workers, a museum sign and drawings. Every one of the 50 images was checked by eye on contact sheets before it went in, and the labels in `eval/image_labels.json` come from that check, not from the titles.

3. **The embedding model doesn't know Latin.** The brief's "Vulpes vulpes matches red fox" failed at the first step: all-minilm scored "Vulpes vulpes" 0.106 against a red-fox caption, and nomic-embed-text ranked a golden retriever (0.396) above the fox (0.380). So posts now go through the same understanding step as images: a language model reads the post and names the animal, and that description is embedded. The two Latin-only posts were then read as "red fox" and "grizzly bear".

4. **Self-reported confidence was useless.** The vision model said 0.95 for a photo of paw prints in snow. Flagging on that number alone would flag nothing. The vision call now also asks for token log-probabilities, and the image is flagged on the probability the model actually gave its answer. On a coyote photo that was 0.53, with "fox" at 0.23 and "wolf" at 0.07: the real uncertainty, which the 0.95 hid.

5. **Two FastAPI bugs the tests caught.** One shared `Path(ge=1)` object across all routes made FastAPI expect a `job_id` on every route (`GET /posts/999/images` answered 422 instead of 404); it is now an `Annotated` type. And `JSONResponse` cannot serialise a `datetime`, which broke `POST /jobs`; it now goes through `jsonable_encoder`.

6. **The vision model misnamed 3 of the 8 wolves**, on the real run: one "arctic fox" (p=0.72), one "fox" (p=0.42), one "white dog" (p=0.75). With the first flag threshold (0.60) only the 0.42 one was flagged, so a wolf labelled "arctic fox" could have been suggested for a fox post. The threshold is now 0.80, picked from the image labels, which flags all three; two correctly named images (0.74, 0.76) are held for review as the price. `app/reflag.py` re-applies a new threshold to the stored probabilities without tagging again.

7. **The first similarity threshold (0.50) missed both dog posts**, whose best image scored 0.49. The sweep on the tune posts showed 0.45 is the strictest value that misses nothing; the eval went from 11/13 to 13/13.

8. **Checking the tests can fail.** I disabled the guard's kind check: 3 tests failed, including the wolf-on-a-fox-post test. I disabled the low-confidence flag: 2 tests failed.

## Decisions I made

- **`kind` is asked for first** in the output schema, so the model commits to the animal before it writes a caption that could steer it, and so its probability is measured before anything else influences it.
- **A retry changes the temperature.** At temperature 0 the same prompt gives the same broken answer, so retries 2 and 3 use 0.4 and say the last answer did not match.
- **Costs are priced at a real cloud rate** (Gemini 2.5 Flash-Lite and Gemini Embedding 2, from Google's pricing page) even though the local run costs nothing, so the cost log and the budget guard have real numbers to work with.
- **The images are committed** (2.9 MB, 50 files) as well as the download script, so the eval is reproducible even if an original disappears.
