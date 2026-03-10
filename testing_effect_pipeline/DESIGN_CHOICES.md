# Testing Effect Pipeline — Design Choices Log

This document records explicit and implicit design choices in the current implementation.

## 1) Core architecture choices

1. Implemented a Python package (`testing_effect_pipeline/`) rather than notebook-only logic.
2. Kept the first implementation framework-light and offline-capable.
3. Added a `ModelAdapter` abstraction to swap mock and real model backends.
4. Prioritized correctness of protocol plumbing over GPU-scale optimization in v1.

## 2) Data and scoring choices

5. Standardized closed-book schema to `item_id/prompt/target` (+ optional `domain_tag/difficulty`).
6. Used JSONL IO for deterministic, streamable datasets.
7. Normalized answers (lowercase + whitespace collapse) for exact-match-style checks.
8. Added synthetic dataset generation for deterministic smoke tests.

## 3) State/memory choices

9. Centralized per-item memory in `ItemState`.
10. Included `is_mastered` and `mastered_at_step` for forgetting probes over learned items.
11. Added EMA tracking (`test_accuracy_ema`, `test_loss_ema`) for noisy probe smoothing.
12. Added `remastery_count` to capture mastered→forgotten→re-mastered cycles.

## 4) Scheduler and test protocol choices

13. Implemented two test protocols: `test_only` and `test_reinforce`.
14. Implemented dynamic queue-pressure allocation instead of fixed ratios.
15. Added per-step allocation logging (`study/test/reinforce` counts) to avoid posthoc ratio reconstruction.
16. Implemented Leitner scheduler as MVP.
17. Implemented FSRS-inspired scheduler as a principled ablation (explicitly approximate).
18. Kept streak/failure bookkeeping in trainer to avoid hidden scheduler side effects.

## 5) Mastery/remastery choices

19. Mastery threshold is configurable (`K` consecutive successes; default 2).
20. On a failed probe of a mastered item, item is marked unmastered.
21. Re-mastery events are logged globally and per-item.

## 6) Baseline choices

22. Implemented four baselines: `standard_ft`, `random_replay`, `curriculum`, `loss_replay`.
23. Defined curriculum ranking as ascending difficulty.
24. For real datasets without provided `difficulty`, difficulty is assigned from model-relative first-exposure loss (calibration pass before training).
25. Defined loss replay priority by descending `test_loss_ema`.

## 7) Budget/fairness/accounting choices

26. Added explicit token accounting utility (`TokenBudgetTracker`).
27. Counted only study + reinforce tokens toward **training token budget** (gradient-bearing phases).
28. Tracked test-phase inference tokens separately as **overhead**.
29. Added budget snapshots to metrics at eval cadence.
30. Added hard stop behavior when training-token budget is exceeded.
31. Applied the same accounting policy to testing-effect and baseline trainers.

## 8) Mock-model behavior choices

32. Kept nonzero forgetting via per-touch decay.
33. Added configurable Gaussian noise (`noise_std`) to make retrieval outcomes stochastic.
34. Preserved separate study vs reinforce gains to reflect different update strengths.

## 9) Output/runner choices

35. CLI outputs JSON metrics with snapshots, allocation logs, budget logs, and remastery stats.
36. Runner supports multi-seed execution and method matrix in one command.
37. Runner exposes budget/noise controls via CLI args.

## 10) Known limitations and deferred items

38. FSRS is an approximation, not canonical FSRS parameter fitting.
39. No production LLM backend integrated yet (mock-first validation).
40. No tokenization-accurate accounting yet (uses deterministic word-count proxy).
41. No built-in plotting utility yet; JSON artifacts intended for downstream analysis.
42. No CI unit tests yet; smoke checks currently validate runnability.
