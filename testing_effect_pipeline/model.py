from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import random

from .types import QAItem


class ModelAdapter(ABC):
    """Abstract interface for plugging real LLM train/test calls into pipeline."""

    @abstractmethod
    def study_update(self, item: QAItem) -> None:
        raise NotImplementedError

    @abstractmethod
    def reinforce_update(self, item: QAItem) -> None:
        raise NotImplementedError

    @abstractmethod
    def test(self, item: QAItem) -> tuple[bool, float]:
        """Return (correct, loss)."""
        raise NotImplementedError

    def compute_loss(self, item: QAItem) -> float:
        """Forward-only loss on an item (no gradient, no generation).
        Override in subclasses that have side effects in test()."""
        _, loss = self.test(item)
        return loss

    def flush(self) -> None:
        """Flush pending buffered updates (no-op for non-buffering adapters)."""

    def test_batch(self, items: list[QAItem]) -> list[tuple[bool, float]]:
        """Batched test. Override for GPU-parallel generation."""
        return [self.test(item) for item in items]


class MockMemoryModel(ModelAdapter):
    """Offline noisy memory model for validating scheduling mechanics without GPUs."""

    def __init__(self, seed: int = 7, decay: float = 0.01, noise_std: float = 0.05) -> None:
        self.rng = random.Random(seed)
        self.strength = defaultdict(float)
        self.decay = decay
        self.noise_std = noise_std

    def _touch(self, item_id: str) -> None:
        self.strength[item_id] = max(0.0, self.strength[item_id] - self.decay)

    def _noisy_prob(self, base: float) -> float:
        noise = self.rng.gauss(0.0, self.noise_std)
        return min(1.0, max(0.0, base + noise))

    def study_update(self, item: QAItem) -> None:
        self._touch(item.item_id)
        self.strength[item.item_id] = min(1.0, self.strength[item.item_id] + 0.08)

    def reinforce_update(self, item: QAItem) -> None:
        self._touch(item.item_id)
        self.strength[item.item_id] = min(1.0, self.strength[item.item_id] + 0.12)

    def test(self, item: QAItem) -> tuple[bool, float]:
        self._touch(item.item_id)
        p = self._noisy_prob(self.strength[item.item_id])
        correct = self.rng.random() < p
        loss = 1.0 - p if correct else min(1.5, 1.5 - p)
        return correct, loss

    def compute_loss(self, item: QAItem) -> float:
        p = self._noisy_prob(self.strength[item.item_id])
        return 1.0 - p
