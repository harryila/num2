from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import BaselineConfig, BaselineTrainer
from .dataset import build_sample_dataset, load_closed_book_jsonl
from .model import MockMemoryModel
from .scheduler import FSRSScheduler, LeitnerScheduler
from .trainer import TestingEffectTrainer, TrainConfig
from .types import QAItem


def _metrics_to_dict(metrics):
    return {
        "forgetting_snapshots": [s.__dict__ for s in metrics.forgetting_snapshots],
        "cumulative_mastered": metrics.cumulative_mastered,
        "mastery_throughput": metrics.mastery_throughput,
        "step_allocations": [s.__dict__ for s in metrics.step_allocations],
        "budget_snapshots": [s.__dict__ for s in metrics.budget_snapshots],
        "remastery_events": metrics.remastery_events,
        "total_remastery_events": metrics.total_remastery_events,
        "stopped_early_budget": metrics.stopped_early_budget,
    }


def _with_model_relative_difficulty(items: list[QAItem], seed: int, noise_std: float) -> list[QAItem]:
    """Assign difficulty from model loss on first exposure (before training)."""

    calibrator = MockMemoryModel(seed=seed + 1000, noise_std=noise_std)
    out: list[QAItem] = []
    for item in items:
        _, loss = calibrator.test(item)
        difficulty = item.difficulty if item.difficulty is not None else loss
        out.append(
            QAItem(
                item_id=item.item_id,
                prompt=item.prompt,
                target=item.target,
                domain_tag=item.domain_tag,
                difficulty=difficulty,
            )
        )
    return out


def run(args: argparse.Namespace) -> dict:
    if args.dataset_path:
        items = load_closed_book_jsonl(args.dataset_path)
    else:
        items = build_sample_dataset(args.sample_size)

    out: dict = {}

    for seed in range(args.seeds):
        seed_key = f"seed_{seed}"
        out[seed_key] = {}
        items_seed = _with_model_relative_difficulty(items, seed=seed, noise_std=args.mock_noise_std)

        for method in args.methods:
            model = MockMemoryModel(seed=seed, noise_std=args.mock_noise_std)
            if method in {"test_only", "test_reinforce"}:
                cfg = TrainConfig(
                    total_steps=args.steps,
                    batch_size=args.batch_size,
                    eval_every_steps=args.eval_every,
                    max_training_tokens=args.max_training_tokens,
                )
                scheduler = LeitnerScheduler() if args.scheduler == "leitner" else FSRSScheduler()
                trainer = TestingEffectTrainer(
                    items=items_seed,
                    model=model,
                    scheduler=scheduler,
                    config=cfg,
                    mode=method,
                    seed=seed,
                )
                metrics = trainer.train()
            else:
                bcfg = BaselineConfig(
                    total_steps=args.steps,
                    batch_size=args.batch_size,
                    eval_every_steps=args.eval_every,
                    max_training_tokens=args.max_training_tokens,
                )
                trainer = BaselineTrainer(items=items_seed, model=model, cfg=bcfg, policy=method, seed=seed)
                metrics = trainer.train()

            out[seed_key][method] = _metrics_to_dict(metrics)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run testing-effect pipeline experiment (offline mock).")
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--scheduler", choices=["leitner", "fsrs"], default="leitner")
    p.add_argument("--max-training-tokens", type=int, default=None)
    p.add_argument("--mock-noise-std", type=float, default=0.05)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["test_only", "test_reinforce", "standard_ft", "random_replay", "curriculum", "loss_replay"],
    )
    p.add_argument("--output", type=str, default="artifacts/experiment_metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
