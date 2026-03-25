from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from .types import ItemState


class Scheduler(ABC):
    @abstractmethod
    def on_result(self, state: ItemState, step: int, correct: bool) -> int:
        """Update state and return next_due_step."""
        raise NotImplementedError


class LeitnerScheduler(Scheduler):
    def __init__(self, min_interval: int = 50, max_interval: int = 20_000, grow: float = 2.0):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.grow = grow

    def on_result(self, state: ItemState, step: int, correct: bool) -> int:
        if correct:
            state.estimated_half_life = max(
                self.min_interval,
                min(self.max_interval, state.estimated_half_life * self.grow),
            )
        else:
            state.estimated_half_life = max(self.min_interval, state.estimated_half_life / self.grow)
        interval = int(max(self.min_interval, min(self.max_interval, state.estimated_half_life)))
        state.next_due_step = step + interval
        return state.next_due_step


class FSRSScheduler(Scheduler):
    """Lightweight FSRS-inspired half-life update.

    Not a full FSRS implementation; this approximates half-life regression behavior for
    controlled ablations inside this pipeline.
    """

    def __init__(self, min_interval: int = 50, max_interval: int = 20_000):
        self.min_interval = min_interval
        self.max_interval = max_interval

    def on_result(self, state: ItemState, step: int, correct: bool) -> int:
        difficulty = min(1.0, state.failure_count / max(1, state.total_tests))
        stability = max(1.0, state.estimated_half_life)

        if correct:
            gain = 1.15 + (0.35 * (1.0 - difficulty))
            stability = stability * gain
        else:
            penalty = 0.55 + 0.25 * difficulty
            stability = max(self.min_interval, stability * penalty)

        interval = int(max(self.min_interval, min(self.max_interval, stability)))
        state.estimated_half_life = stability
        state.next_due_step = step + interval
        return state.next_due_step


class RandomMatchedScheduler(Scheduler):
    """Uniform random intervals in [50, 500]. Ignores correctness.

    Matches typical FSRS early/mid-training review frequency so the ablation
    compares scheduling *intelligence* at matched review rate.
    """

    def __init__(self, min_interval: int = 50, max_interval: int = 500, seed: int = 0):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.rng = random.Random(seed)

    def on_result(self, state: ItemState, step: int, correct: bool) -> int:
        interval = self.rng.randint(self.min_interval, self.max_interval)
        state.next_due_step = step + interval
        return state.next_due_step


class RandomWideScheduler(Scheduler):
    """Log-uniform random intervals in [50, 20_000]. Ignores correctness.

    Biased toward shorter intervals (geometric mean ~1,000) so items do come
    due occasionally, but much less frequently than FSRS or RandomMatched.
    """

    def __init__(self, min_interval: int = 50, max_interval: int = 20_000, seed: int = 0):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.rng = random.Random(seed)
        self._log_min = math.log(self.min_interval)
        self._log_max = math.log(self.max_interval)

    def on_result(self, state: ItemState, step: int, correct: bool) -> int:
        interval = int(math.exp(self.rng.uniform(self._log_min, self._log_max)))
        state.next_due_step = step + interval
        return state.next_due_step
