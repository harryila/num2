from __future__ import annotations

from dataclasses import dataclass
import random

from .budget import TokenBudgetTracker
from .model import ModelAdapter
from .scheduler import Scheduler
from .types import BudgetSnapshot, ItemState, QAItem, RetentionSnapshot, StepAllocation, TrainingMetrics


@dataclass
class TrainConfig:
    total_steps: int = 25_000
    batch_size: int = 16
    mastery_k: int = 2
    eval_every_steps: int = 1_000
    min_study_fraction: float = 0.3
    max_test_fraction: float = 0.5
    max_training_tokens: int | None = None


class TestingEffectTrainer:
    def __init__(
        self,
        items: list[QAItem],
        model: ModelAdapter,
        scheduler: Scheduler,
        config: TrainConfig,
        mode: str = "test_only",
        seed: int = 7,
    ) -> None:
        if mode not in {"test_only", "test_reinforce"}:
            raise ValueError("mode must be 'test_only' or 'test_reinforce'")

        self.items = items
        self.item_by_id = {i.item_id: i for i in items}
        self.model = model
        self.scheduler = scheduler
        self.config = config
        self.mode = mode
        self.rng = random.Random(seed)

        self.state = {i.item_id: ItemState(item_id=i.item_id) for i in items}
        self.metrics = TrainingMetrics()
        self.budget = TokenBudgetTracker(max_training_tokens=config.max_training_tokens)
        self._study_cursor = 0

    def _next_study_items(self, n: int) -> list[QAItem]:
        out: list[QAItem] = []
        for _ in range(n):
            item = self.items[self._study_cursor % len(self.items)]
            self._study_cursor += 1
            out.append(item)
        return out

    def _due_items(self, step: int) -> list[QAItem]:
        due = [self.item_by_id[item_id] for item_id, st in self.state.items() if st.next_due_step <= step]
        self.rng.shuffle(due)
        return due

    def _retention_at_horizon(self, step: int, delta: int) -> float:
        eligible = [
            st.item_id
            for st in self.state.values()
            if st.is_mastered and st.mastered_at_step is not None and st.mastered_at_step <= step - delta
        ]
        if not eligible:
            return 0.0
        items = [self.item_by_id[iid] for iid in eligible]
        for item in items:
            self.budget.add_test_inference(item)
        results = self.model.test_batch(items)
        return sum(int(c) for c, _ in results) / len(results)

    def _snapshot(self, step: int) -> None:
        mastered = sum(1 for st in self.state.values() if st.is_mastered)
        self.metrics.forgetting_snapshots.append(
            RetentionSnapshot(
                step=step,
                total_mastered=mastered,
                retained_after_5k=self._retention_at_horizon(step, 5_000),
                retained_after_10k=self._retention_at_horizon(step, 10_000),
                retained_after_20k=self._retention_at_horizon(step, 20_000),
            )
        )
        self.metrics.cumulative_mastered.append((step, mastered))
        self.metrics.budget_snapshots.append(
            BudgetSnapshot(
                step=step,
                training_tokens_used=self.budget.training_tokens_used,
                test_inference_tokens_used=self.budget.test_inference_tokens_used,
            )
        )

        if len(self.metrics.cumulative_mastered) > 1:
            prev_step, prev_mastered = self.metrics.cumulative_mastered[-2]
            span = max(1, step - prev_step)
            throughput = (mastered - prev_mastered) / span * 1000.0
            self.metrics.mastery_throughput.append((step, throughput))

    def train(self) -> TrainingMetrics:
        cfg = self.config
        for step in range(1, cfg.total_steps + 1):
            due = self._due_items(step)
            due_pressure = min(1.0, len(due) / max(1, cfg.batch_size))

            test_fraction = min(cfg.max_test_fraction, due_pressure)
            study_fraction = max(cfg.min_study_fraction, 1.0 - test_fraction)

            n_test = int(cfg.batch_size * test_fraction)
            n_study = int(cfg.batch_size * study_fraction)
            n_reinforce = max(0, cfg.batch_size - n_test - n_study)

            study_items = self._next_study_items(n_study)
            for item in study_items:
                self.model.study_update(item)
                self.budget.add_study(item)
                self.state[item.item_id].last_study_step = step

            test_items = due[:n_test] if n_test > 0 else []
            failures: list[QAItem] = []

            if test_items:
                for item in test_items:
                    self.budget.add_test_inference(item)
                test_results = self.model.test_batch(test_items)

                for item, (correct, loss) in zip(test_items, test_results):
                    st = self.state[item.item_id]
                    st.last_test_step = step
                    st.total_tests += 1
                    st.total_correct += int(correct)
                    alpha = 0.2
                    st.test_accuracy_ema = (1 - alpha) * st.test_accuracy_ema + alpha * float(correct)
                    st.test_loss_ema = (1 - alpha) * st.test_loss_ema + alpha * loss

                    if correct:
                        st.success_streak += 1
                    else:
                        if st.is_mastered:
                            st.is_mastered = False
                        st.success_streak = 0
                        st.failure_count += 1
                        failures.append(item)

                    self.scheduler.on_result(st, step, correct)

                    if st.success_streak >= cfg.mastery_k:
                        if st.mastered_at_step is not None and not st.is_mastered:
                            st.remastery_count += 1
                            self.metrics.total_remastery_events += 1
                            self.metrics.remastery_events.append((step, item.item_id))
                        st.is_mastered = True
                        if st.mastered_at_step is None:
                            st.mastered_at_step = step

            reinforced = 0
            if self.mode == "test_reinforce" and n_reinforce > 0 and failures:
                for item in failures[:n_reinforce]:
                    self.model.reinforce_update(item)
                    self.budget.add_reinforce(item)
                    reinforced += 1

            self.metrics.step_allocations.append(
                StepAllocation(step=step, study_count=n_study, test_count=len(test_items), reinforce_count=reinforced)
            )

            if step % cfg.eval_every_steps == 0:
                self._snapshot(step)

            if self.budget.over_budget():
                self.metrics.stopped_early_budget = True
                break

        self.model.flush()
        return self.metrics
