"""N2: compute soft-accuracy metrics over per-item generations from the offline LoRA eval.

Reads JSONLs produced by `analysis.eval_lora_offline` (one per LoRA × held-out set).
For each item we computes a battery of "is the answer close to gold?" scores:

  - strict_em      : NQ-style exact match (the original metric)
  - lenient_em     : normalized substring match (gold appears in prediction or vice versa)
  - token_f1       : SQuAD-style token-level F1 between gold and prediction
  - edit_sim       : 1 - normalized Levenshtein distance
  - first_token_match : does the first generated content token match the gold's first token?
  - char_substr    : does gold appear as char-substring in prediction (case-insensitive)?

Outputs:
    analysis/results/soft_acc_summary.csv
    analysis/results/soft_acc_per_item.jsonl   (every item, every metric)

Usage:
    .venv_analysis/bin/python -m analysis.soft_accuracy \\
        --results-dir analysis/results \\
        --output-dir analysis/results \\
        [--limit-per-run 200]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from testing_effect_pipeline.nq_eval import normalize_nq_answer

logger = logging.getLogger(__name__)


@dataclass
class SoftScore:
    strict_em: float
    lenient_em: float
    token_f1: float
    edit_sim: float
    first_token_match: float
    char_substr: float


# ----- per-item metric helpers -----


def _norm_for_tokens(s: str) -> list[str]:
    """Lowercase + strip punct, return token list."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return s.split()


def token_f1(gold: str, pred: str) -> float:
    g = _norm_for_tokens(gold)
    p = _norm_for_tokens(pred)
    if not g or not p:
        return 1.0 if (not g and not p) else 0.0
    g_count = Counter(g)
    p_count = Counter(p)
    overlap = sum((g_count & p_count).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def edit_distance(a: str, b: str) -> int:
    """Levenshtein. O(n*m) — fine for short answers."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = cur
    return prev[-1]


def edit_sim(gold: str, pred: str) -> float:
    g = gold.lower().strip()
    p = pred.lower().strip()
    L = max(len(g), len(p))
    if L == 0:
        return 1.0
    return 1.0 - edit_distance(g, p) / L


def first_token_match(gold: str, pred: str) -> bool:
    g = _norm_for_tokens(gold)
    p = _norm_for_tokens(pred)
    if not g or not p:
        return False
    return g[0] == p[0]


def lenient_em(gold: str, pred: str) -> bool:
    ng = normalize_nq_answer(gold)
    npd = normalize_nq_answer(pred)
    if not ng or not npd:
        return False
    return ng == npd or ng in npd or npd in ng


def char_substr(gold: str, pred: str) -> bool:
    return gold.lower().strip() in pred.lower().strip()


def best_score_against_targets(pred: str, targets_raw: str, fn) -> float:
    """For multi-target items (gold = 'a|||b'), take the best per-metric score."""
    best = 0.0
    for t in targets_raw.split("|||"):
        t = t.strip()
        if not t:
            continue
        v = fn(t, pred)
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        if v > best:
            best = v
    return best


def score_item(target: str, prediction: str, strict_correct: bool) -> SoftScore:
    return SoftScore(
        strict_em=1.0 if strict_correct else 0.0,
        lenient_em=best_score_against_targets(prediction, target, lenient_em),
        token_f1=best_score_against_targets(prediction, target, token_f1),
        edit_sim=best_score_against_targets(prediction, target, edit_sim),
        first_token_match=best_score_against_targets(prediction, target, first_token_match),
        char_substr=best_score_against_targets(prediction, target, char_substr),
    )


# ----- aggregation -----


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=Path("analysis/results"), help="dir with <set>/<run>.jsonl files")
    p.add_argument("--output-dir", type=Path, default=Path("analysis/results"))
    p.add_argument("--limit-per-run", type=int, default=None, help="cap items per run (debug)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    set_dirs = [d for d in args.results_dir.iterdir() if d.is_dir()]
    if not set_dirs:
        sys.exit(f"no held-out set subdirs in {args.results_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "soft_acc_summary.csv"
    per_item_path = args.output_dir / "soft_acc_per_item.jsonl"

    fields = ["set", "run", "n", "strict_em", "lenient_em", "token_f1", "edit_sim", "first_token_match", "char_substr"]

    with summary_path.open("w") as sf, per_item_path.open("w") as pf:
        writer = csv.DictWriter(sf, fieldnames=fields)
        writer.writeheader()

        for set_dir in sorted(set_dirs):
            set_name = set_dir.name
            for run_path in sorted(set_dir.glob("*.jsonl")):
                run_name = run_path.stem
                totals = {f: 0.0 for f in fields[3:]}
                n = 0
                with run_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        score = score_item(d["target"], d.get("prediction", ""), bool(d.get("correct")))
                        n += 1
                        if args.limit_per_run and n > args.limit_per_run:
                            break
                        totals["strict_em"] += score.strict_em
                        totals["lenient_em"] += score.lenient_em
                        totals["token_f1"] += score.token_f1
                        totals["edit_sim"] += score.edit_sim
                        totals["first_token_match"] += score.first_token_match
                        totals["char_substr"] += score.char_substr
                        pf.write(
                            json.dumps(
                                {
                                    "set": set_name,
                                    "run": run_name,
                                    "item_id": d["item_id"],
                                    "target": d["target"],
                                    "prediction": d.get("prediction", ""),
                                    "strict_em": score.strict_em,
                                    "lenient_em": score.lenient_em,
                                    "token_f1": score.token_f1,
                                    "edit_sim": score.edit_sim,
                                    "first_token_match": score.first_token_match,
                                    "char_substr": score.char_substr,
                                }
                            )
                            + "\n"
                        )
                row = {"set": set_name, "run": run_name, "n": n}
                for k, v in totals.items():
                    row[k] = v / max(1, n)
                writer.writerow(row)
                logger.info(
                    "%-10s %-50s n=%d strict=%.3f lenient=%.3f F1=%.3f edit=%.3f first_tok=%.3f substr=%.3f",
                    set_name,
                    run_name[:50],
                    n,
                    row["strict_em"],
                    row["lenient_em"],
                    row["token_f1"],
                    row["edit_sim"],
                    row["first_token_match"],
                    row["char_substr"],
                )

    logger.info("Wrote %s and %s", summary_path, per_item_path)


if __name__ == "__main__":
    main()
