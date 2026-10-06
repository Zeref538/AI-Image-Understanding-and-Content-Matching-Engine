"""Download the 50-image corpus listed in data/corpus.json into data/images/.

Every image is CC0 or Public Domain Mark (found through the Openverse API, openverse.org),
so it can be used and redistributed with no conditions. Images are resized to at most
640 px on the long side. The committed images are the reference copy; this script
re-creates them if the originals are still online.

    python scripts/fetch_corpus.py           # skips files that already exist
"""
import io
import json
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "images"
UA = {"User-Agent": "flyrank-capstone-image-relevance (github.com/Zeref538)"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = json.loads((ROOT / "data" / "corpus.json").read_text(encoding="utf-8"))
    with httpx.Client(headers=UA, timeout=60, follow_redirects=True) as http:
        for item in corpus:
            dest = OUT / item["file"]
            if dest.exists():
                continue
            if not item["license"].startswith(("cc0", "pdm")):
                raise SystemExit(f"{item['file']}: license {item['license']} is not CC0/PDM")
            im = Image.open(io.BytesIO(http.get(item["url"]).raise_for_status().content)).convert("RGB")
            im.thumbnail((640, 640))
            im.save(dest, "JPEG", quality=82, optimize=True)
            print(f"{item['file']}  {im.size[0]}x{im.size[1]}  {item['license']}  {item['title'][:50]}")
    total = sum(f.stat().st_size for f in OUT.glob("*.jpg"))
    print(f"{len(list(OUT.glob('*.jpg')))} images, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
