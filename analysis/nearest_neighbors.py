"""N5 nearest-neighbor lift analysis.

For each held-out item, find its max-cosine-similarity neighbor in the training set.
Then for each RP/SFT LoRA, compute:

  - mean / median similarity of correct vs wrong items
  - P(correct | sim ≥ τ) vs P(correct | sim < τ) for τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
  - "lift_at_τ" = P(correct | sim ≥ τ) − P(correct | sim < τ)

Outputs (one row per LoRA × held-out set):
    analysis/results/neighbors_summary.csv
    analysis/results/neighbors_wins.csv         (RP-only wins by similarity bucket)
    analysis/results/neighbors_per_item.jsonl   (item_id, sim, rp_correct, sft_correct)

Usage:
    .venv_analysis/bin/python -m analysis.nearest_neighbors \
        --embeddings analysis/cache_openai_embeddings.npz \
        --results-dir analysis/results \
        --train-key train \
        --sets indist ood synthetic
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def load_correct(path: Path) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["item_id"]] = bool(d["correct"])
    return out


def max_sims(query_vecs: np.ndarray, train_vecs: np.ndarray, batch: int = 512) -> np.ndarray:
    """Return the max cosine sim of each query against any training row.

    Both inputs are L2-normalized so dot product == cosine.
    """
    out = np.empty(query_vecs.shape[0], dtype=np.float32)
    train_T = train_vecs.T
    for i in range(0, query_vecs.shape[0], batch):
        chunk = query_vecs[i : i + batch]
        sims = chunk @ train_T
        out[i : i + batch] = sims.max(axis=1)
    return out


def summarize_run(
    run_name: str,
    set_name: str,
    method: str,
    correct: Dict[str, bool],
    ids: List[str],
    sims: np.ndarray,
) -> dict:
    """Compute lift-at-threshold stats for one LoRA × set."""
    mask = np.array([iid in correct for iid in ids])
    if not mask.any():
        return {}
    sub_ids = [iid for iid, m in zip(ids, mask) if m]
    sub_sims = sims[mask]
    sub_correct = np.array([correct[iid] for iid in sub_ids])

    n = len(sub_ids)
    n_c = int(sub_correct.sum())
    row = {
        "run": run_name,
        "method": method,
        "split": set_name,
        "n": n,
        "n_correct": n_c,
        "overall_acc": round(n_c / n, 4),
        "mean_sim_correct": round(float(sub_sims[sub_correct].mean()) if n_c else 0.0, 4),
        "mean_sim_wrong":   round(float(sub_sims[~sub_correct].mean()) if (n - n_c) else 0.0, 4),
        "median_sim_correct": round(float(np.median(sub_sims[sub_correct])) if n_c else 0.0, 4),
        "median_sim_wrong":   round(float(np.median(sub_sims[~sub_correct])) if (n - n_c) else 0.0, 4),
    }
    for tau in THRESHOLDS:
        hi = sub_sims >= tau
        lo = ~hi
        n_hi = int(hi.sum()); n_lo = int(lo.sum())
        p_hi = sub_correct[hi].mean() if n_hi else 0.0
        p_lo = sub_correct[lo].mean() if n_lo else 0.0
        row[f"n_above_{tau}"] = n_hi
        row[f"p_above_{tau}"] = round(float(p_hi), 4)
        row[f"p_below_{tau}"] = round(float(p_lo), 4)
        row[f"lift_at_{tau}"] = round(float(p_hi - p_lo), 4)
    return row


def parse_run_meta(stem: str) -> tuple[str, str]:
    """Best-effort extract (run_name, method) from filename stem."""
    s = stem.replace(".lora", "")
    if "retrievalpractice" in s or "retrieval_practice" in s or "_retrieval" in s:
        method = "retrieval_practice"
    elif "standard_ft_mastered" in s or "_mastered" in s:
        method = "standard_ft_mastered"
    elif "standardft" in s or "_standard" in s:
        method = "standard_ft"
    else:
        method = "unknown"
    return s, method


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("analysis/results"))
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    ap.add_argument("--train-key", default="train")
    ap.add_argument("--sets", nargs="+", default=["indist", "ood", "synthetic"])
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embeddings from %s", args.embeddings)
    data = np.load(args.embeddings, allow_pickle=True)
    train_vecs = data[args.train_key]
    logger.info("  train vecs: shape=%s", train_vecs.shape)

    set_vecs: Dict[str, np.ndarray] = {}
    set_ids: Dict[str, List[str]] = {}
    for s in args.sets:
        if s not in data:
            logger.warning("  set %s missing from npz, skipping", s)
            continue
        set_vecs[s] = data[s]
        set_ids[s] = [str(x) for x in data[f"{s}_ids"]]
        logger.info("  %-10s vecs: shape=%s", s, set_vecs[s].shape)

    sims_per_set: Dict[str, np.ndarray] = {}
    for s, v in set_vecs.items():
        logger.info("Computing max sim for %s …", s)
        sims_per_set[s] = max_sims(v, train_vecs)

    rows: list[dict] = []
    for s in set_vecs.keys():
        set_dir = args.results_dir / s
        if not set_dir.exists():
            logger.warning("  no results dir %s", set_dir)
            continue
        for jl in sorted(set_dir.glob("*.jsonl")):
            stem = jl.stem
            run_name, method = parse_run_meta(stem)
            correct = load_correct(jl)
            row = summarize_run(run_name, s, method, correct, set_ids[s], sims_per_set[s])
            if row:
                rows.append(row)

    summary_path = args.output_dir / "neighbors_summary.csv"
    if rows:
        with summary_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info("Wrote %s (%d rows)", summary_path, len(rows))

    pi_path = args.output_dir / "neighbors_per_item.jsonl"
    with pi_path.open("w") as f:
        for s, sims in sims_per_set.items():
            ids = set_ids[s]
            for iid, sim in zip(ids, sims):
                f.write(json.dumps({"set": s, "item_id": iid, "max_sim": float(sim)}) + "\n")
    logger.info("Wrote per-item sims to %s", pi_path)


if __name__ == "__main__":
    main()
