from __future__ import annotations

from dataclasses import dataclass

from .types import QAItem


def estimate_item_tokens(item: QAItem) -> int:
    """Cheap deterministic token proxy for budget accounting."""

    prompt_tokens = max(1, len(item.prompt.split()))
    target_tokens = max(1, len(item.target.split()))
    return prompt_tokens + target_tokens


@dataclass
class TokenBudgetTracker:
    max_training_tokens: int | None = None
    training_tokens_used: int = 0
    test_inference_tokens_used: int = 0

    def add_study(self, item: QAItem) -> None:
        self.training_tokens_used += estimate_item_tokens(item)

    def add_reinforce(self, item: QAItem) -> None:
        self.training_tokens_used += estimate_item_tokens(item)

    def add_test_inference(self, item: QAItem) -> None:
        self.test_inference_tokens_used += estimate_item_tokens(item)

    def over_budget(self) -> bool:
        if self.max_training_tokens is None:
            return False
        return self.training_tokens_used > self.max_training_tokens
