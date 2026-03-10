"""Download and prepare NQ Open dataset for the testing-effect pipeline.

Usage:
    python -m testing_effect_pipeline.prepare_nq_dataset \
        --output-dir data \
        --train-subsample 3000 \
        --hf-token YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NQ Open dataset.")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument(
        "--train-subsample",
        type=int,
        default=3000,
        help="Number of training items to subsample from ~87k (default: 3000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading google-research-datasets/nq_open from HuggingFace...")
    ds = load_dataset("google-research-datasets/nq_open", token=args.hf_token)

    train_ds = ds["train"]
    test_ds = ds["validation"]
    print(f"Full train: {len(train_ds)} items | Test (validation split): {len(test_ds)} items")

    rng = random.Random(args.seed)
    indices = list(range(len(train_ds)))
    rng.shuffle(indices)
    train_indices = sorted(indices[: args.train_subsample])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "nq_open_train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for idx in train_indices:
            row = train_ds[idx]
            answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
            item = {
                "item_id": f"nq-train-{idx}",
                "prompt": row["question"],
                "target": "|||".join(answers),
                "domain_tag": "nq_open",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_indices)} train items -> {train_path}")

    test_path = output_dir / "nq_open_test.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for idx in range(len(test_ds)):
            row = test_ds[idx]
            answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
            item = {
                "item_id": f"nq-test-{idx}",
                "prompt": row["question"],
                "target": "|||".join(answers),
                "domain_tag": "nq_open",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(test_ds)} test items -> {test_path}")

    stats = {
        "source": "google-research-datasets/nq_open",
        "train_full_size": len(train_ds),
        "train_subsample_size": len(train_indices),
        "test_size": len(test_ds),
        "subsample_seed": args.seed,
        "target_format": "pipe-separated alternatives (|||)",
    }
    stats_path = output_dir / "nq_open_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote dataset stats -> {stats_path}")


if __name__ == "__main__":
    main()
