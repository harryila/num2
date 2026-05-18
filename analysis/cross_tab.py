"""N4 cross-tab analysis: join taxonomy × per-item RP/SFT correctness.

For each (RP, SFT) comparable LoRA pair, decompose the held-out wins by:
  - q_type (when/who/where/what/how/why/which/other)
  - a_type (date/person/place/number/organization/title/event/other)
  - topic  (geography/history/pop_culture/science/sports/literature/politics/other)
  - specificity (single/multiple_valid)

For each label value we report:
  - n_items in the held-out set
  - p_RP_correct, p_SFT_correct
  - gap = p_RP - p_SFT  (positive → RP wins on this slice)

Plus a "wins decomposition":
  - items where RP correct AND SFT wrong  → "RP wins"
  - items where SFT correct AND RP wrong  → "SFT wins"
  - items where both correct
  - items where both wrong
distributed by label, to show *which kinds of items* RP differentially solves.

Outputs:
    analysis/results/cross_tab_<contrast>.csv         per-label aggregates
    analysis/results/cross_tab_<contrast>_wins.csv    {RP_only, SFT_only, both, neither} × labels
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

LABEL_FIELDS = ("q_type", "a_type", "topic", "specificity")


def load_taxonomy(path: Path) -> Dict[str, Dict[str, str]]:
    tax: Dict[str, Dict[str, str]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            iid = rec["item_id"]
            tax[iid] = {k: rec.get(k, "unknown") for k in LABEL_FIELDS}
    return tax


def load_correct(path: Path) -> Dict[str, bool]:
    """Return {item_id: correct} from a result JSONL."""
    out: Dict[str, bool] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["item_id"]] = bool(d["correct"])
    return out


def aggregate(
    rp: Dict[str, bool],
    sft: Dict[str, bool],
    tax: Dict[str, Dict[str, str]],
) -> Tuple[Dict[Tuple[str, str], Dict[str, int]], Dict[Tuple[str, str], Dict[str, int]]]:
    """Return per-label aggregates and per-label wins-decomposition counts.

    per_label[(field, value)] = {n, rp_correct, sft_correct}
    wins[(field, value)]      = {rp_only, sft_only, both, neither}
    """
    per_label: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "rp_correct": 0, "sft_correct": 0}
    )
    wins: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"rp_only": 0, "sft_only": 0, "both": 0, "neither": 0}
    )

    items_with_tax = 0
    for iid in rp.keys() & sft.keys():
        labels = tax.get(iid)
        if labels is None:
            continue
        items_with_tax += 1
        rp_ok = rp[iid]
        sft_ok = sft[iid]

        bucket = (
            "both" if rp_ok and sft_ok
            else "rp_only" if rp_ok
            else "sft_only" if sft_ok
            else "neither"
        )

        for field in LABEL_FIELDS:
            value = labels.get(field, "unknown")
            key = (field, value)
            per_label[key]["n"] += 1
            per_label[key]["rp_correct"] += int(rp_ok)
            per_label[key]["sft_correct"] += int(sft_ok)
            wins[key][bucket] += 1

    logger.info("  matched %d items across RP, SFT, and taxonomy", items_with_tax)
    return per_label, wins


def write_per_label_csv(per_label: Dict[Tuple[str, str], Dict[str, int]], out_path: Path) -> None:
    rows = []
    for (field, value), agg in per_label.items():
        n = agg["n"]
        if n == 0:
            continue
        p_rp = agg["rp_correct"] / n
        p_sft = agg["sft_correct"] / n
        rows.append({
            "field": field,
            "value": value,
            "n": n,
            "p_rp": round(p_rp, 4),
            "p_sft": round(p_sft, 4),
            "gap_pp": round(100 * (p_rp - p_sft), 2),
        })
    rows.sort(key=lambda r: (r["field"], -r["n"]))
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["field", "value", "n", "p_rp", "p_sft", "gap_pp"])
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s (%d rows)", out_path, len(rows))


def write_wins_csv(wins: Dict[Tuple[str, str], Dict[str, int]], out_path: Path) -> None:
    rows = []
    for (field, value), agg in wins.items():
        total = sum(agg.values())
        if total == 0:
            continue
        rows.append({
            "field": field,
            "value": value,
            "n": total,
            "rp_only": agg["rp_only"],
            "sft_only": agg["sft_only"],
            "both": agg["both"],
            "neither": agg["neither"],
            "rp_only_frac": round(agg["rp_only"] / total, 4),
            "sft_only_frac": round(agg["sft_only"] / total, 4),
        })
    rows.sort(key=lambda r: (r["field"], -r["n"]))
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "field", "value", "n",
                "rp_only", "sft_only", "both", "neither",
                "rp_only_frac", "sft_only_frac",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s (%d rows)", out_path, len(rows))


CONTRASTS = {
    "r16_heldout_seed1": (
        "set2_stage3_r16_retrieval_seed1_held",
        "set2_stage3_r16_standard_seed1_held",
    ),
    "r8_heldout_seed1": (
        "set2_stage3_r8_retrieval_seed1_held",
        "set2_stage3_r8_standard_seed1_held",
    ),
    "r16_8k": (
        "set2_stage3_r16_retrieval_8k",
        "set2_stage3_r16_standard_8k",
    ),
    "quartile_1_easy": (
        "set1_quartile_1_easy_retrievalpractice",
        "set1_quartile_1_easy_standardft",
    ),
    "quartile_2": (
        "set1_quartile_2_retrievalpractice",
        "set1_quartile_2_standardft",
    ),
    "quartile_3": (
        "set1_quartile_3_retrievalpractice",
        "set1_quartile_3_standardft",
    ),
    "quartile_4_hard": (
        "set1_quartile_4_hard_retrievalpractice",
        "set1_quartile_4_hard_standardft",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("analysis/results"))
    ap.add_argument("--taxonomy", type=Path, default=Path("data/nq_open_taxonomy.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    ap.add_argument("--sets", nargs="+", default=["indist", "ood", "synthetic"],
                    help="held-out tags to process")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading taxonomy from %s", args.taxonomy)
    tax = load_taxonomy(args.taxonomy)
    logger.info("  %d items have labels", len(tax))

    for set_tag in args.sets:
        set_dir = args.results_dir / set_tag
        if not set_dir.exists():
            logger.warning("Skipping %s (dir does not exist)", set_dir)
            continue
        logger.info("=== set: %s ===", set_tag)
        for contrast_name, (rp_stem, sft_stem) in CONTRASTS.items():
            rp_path = set_dir / f"{rp_stem}.jsonl"
            sft_path = set_dir / f"{sft_stem}.jsonl"
            if not (rp_path.exists() and sft_path.exists()):
                continue
            logger.info("Contrast %s: %s ⟷ %s", contrast_name, rp_stem, sft_stem)
            rp = load_correct(rp_path)
            sft = load_correct(sft_path)
            per_label, wins = aggregate(rp, sft, tax)
            base = f"cross_tab_{set_tag}_{contrast_name}"
            write_per_label_csv(per_label, args.output_dir / f"{base}.csv")
            write_wins_csv(wins, args.output_dir / f"{base}_wins.csv")


if __name__ == "__main__":
    main()
