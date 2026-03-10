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
35. The mock's `_touch` decay applies on every call, including test. This means retrieval weakens item strength — the opposite of the testing effect in humans. This is intentional: the mock should not build in the hypothesis being tested. If the testing-effect method wins despite the mock penalizing retrieval, the result is more robust (conservative bias against the experimental condition).

## 9) Output/runner choices

36. CLI outputs JSON metrics with snapshots, allocation logs, budget logs, and remastery stats.
37. Runner supports multi-seed execution and method matrix in one command.
38. Runner exposes budget/noise controls via CLI args.
39. Added `--require-budget` flag that errors out if `--max-training-tokens` is not set. Replay baselines consume more training tokens per step than testing-effect methods (study batch + replay updates vs study + test-only overhead), so uncapped runs produce unfair comparisons. Real experiments should always use `--require-budget`.

## 10) Retention horizon minimum-step requirements

40. Forgetting probes measure retention at T+5k, T+10k, and T+20k steps after mastery. For the 5k window to produce any signal, training must run at least ~5k steps past the first mastery event. For the 20k window to produce meaningful data, total steps should be at least 25k–30k. The example real config uses 20k steps; this is sufficient for 5k and 10k horizons but marginal for 20k. Scale `--steps` accordingly for the retention horizon you care about most.

## 11) Known limitations and deferred items

41. FSRS is an approximation, not canonical FSRS parameter fitting.
42. No production LLM backend integrated yet (mock-first validation).
43. No tokenization-accurate accounting yet (uses deterministic word-count proxy).
44. No built-in plotting utility yet; JSON artifacts intended for downstream analysis.
45. No CI unit tests yet; smoke checks currently validate runnability.
