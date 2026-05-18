"""Embed NQ-style items using OpenAI embeddings, caching to npz.

Used as a (sharper) drop-in alternative to the sentence-transformers cache
in `analysis/cache_embeddings.npz` from the prior N5 run.

Input JSONLs are concatenated and deduped by item_id. Each input file becomes
one named array in the output npz with shape (n_items, dim).

Usage:
    export OPENAI_API_KEY=sk-...
    .venv_analysis/bin/python -m analysis.embed_items \\
        --input train:data/nq_open_hard_10k.jsonl \\
        --input indist:data/nq_open_hard_heldout_2k.jsonl \\
        --input ood:data/nq_open_test_hard.jsonl \\
        --input synthetic:data/synthetic/hard.jsonl \\
        --output analysis/cache_openai_embeddings.npz \\
        --model text-embedding-3-large
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI, APIError

logger = logging.getLogger(__name__)


def parse_input_spec(s: str) -> tuple[str, Path]:
    if ":" not in s:
        raise argparse.ArgumentTypeError("expected name:path")
    name, p = s.split(":", 1)
    return name, Path(p)


def load_items(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            iid = d.get("item_id")
            prompt = d.get("prompt") or d.get("question") or ""
            if not iid or not prompt:
                continue
            out.append({"item_id": iid, "prompt": prompt})
    return out


def embed_texts(client: OpenAI, model: str, texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embed a list of strings, returning a (n, dim) float32 array."""
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        for attempt in range(1, 4):
            try:
                resp = client.embeddings.create(model=model, input=chunk)
                for row in resp.data:
                    all_vecs.append(row.embedding)
                break
            except APIError as e:
                wait = 2**attempt
                logger.warning("embed batch %d failed: %s — retry in %ds", i // batch_size + 1, e, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"embed batch failed after retries at offset {i}")
        if (i // batch_size + 1) % 10 == 0:
            logger.info("  %d / %d items embedded", min(i + batch_size, len(texts)), len(texts))
    return np.array(all_vecs, dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", action="append", required=True, type=parse_input_spec, help="name:path (repeatable)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="text-embedding-3-large")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    arrays: dict[str, np.ndarray] = {}
    ids: dict[str, list[str]] = {}

    for name, path in args.input:
        items = load_items(path)
        logger.info("'%s' — %d items from %s", name, len(items), path)
        if not items:
            continue
        texts = [it["prompt"] for it in items]
        t0 = time.time()
        vecs = embed_texts(client, args.model, texts, batch_size=args.batch_size)
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        vecs = vecs / norm
        logger.info("  embedded shape=%s in %.1fs", vecs.shape, time.time() - t0)
        arrays[name] = vecs
        ids[name] = [it["item_id"] for it in items]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict = dict(arrays)
    for name, arr_ids in ids.items():
        save_dict[f"{name}_ids"] = np.array(arr_ids, dtype=object)
    np.savez(args.output, **save_dict)
    logger.info("Saved %d named arrays to %s", len(arrays), args.output)


if __name__ == "__main__":
    main()
