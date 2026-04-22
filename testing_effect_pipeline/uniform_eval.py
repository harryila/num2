from __future__ import annotations

from .types import QAItem, UniformEvalResult


def run_uniform_eval(model, items: list[QAItem], step: int = -1, include_per_item: bool = True) -> UniformEvalResult:
    """Run exact-match generation on all items. Used for cross-method comparison.

    Set include_per_item=False for periodic snapshots to avoid JSON bloat.
    """
    results = model.test_batch(items)
    per_item = [(item.item_id, correct, loss) for item, (correct, loss) in zip(items, results)]
    correct_count = sum(1 for _, c, _ in per_item if c)
    total = len(items)
    mean_loss = sum(loss for _, _, loss in per_item) / max(1, total)
    return UniformEvalResult(
        step=step,
        correct_count=correct_count,
        total=total,
        accuracy=correct_count / max(1, total),
        mean_loss=mean_loss,
        per_item=per_item if include_per_item else [],
    )
