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

## 11) Real model backend choices

41. Chose **Qwen2.5-1.5B-Instruct** as the real model: small enough for 6 methods × 3 seeds × 25k steps on a single GPU, large enough for credible results, instruction-tuned for clean zero-shot QA generation.
42. Used **LoRA** (r=16, α=32, targets `q_proj`+`v_proj`) via PEFT instead of full fine-tuning. Dramatically cheaper per seed and enables running the full experiment matrix on one GPU.
43. Training loss is **masked to answer tokens only** (`labels[:prompt_len] = -100`). Prompt tokens contribute no gradient signal — this matches standard instruction-tuning practice and prevents the model from memorizing prompt structure.
44. Internal **gradient accumulation** (default 4 items) bridges the gap between the pipeline's item-by-item `study_update` calls and efficient optimizer steps with larger effective batch size.
45. `reset_adapter()` re-initializes LoRA parameters (Kaiming for A, zeros for B) + fresh optimizer between seed×method runs without reloading the 1.5B base model from disk. Saves ~18 model loads across the full matrix.
46. `study_update` and `reinforce_update` are **functionally identical** — both call `_train_on_item` with the same loss function. The `test_reinforce` variant is therefore "test + re-study on failures," not RL-style reinforcement. This is the cleaner experimental design: it isolates the effect of retrieval-gated scheduling without confounding the gradient signal.

## 12) Dataset choices

47. Chose **NQ Open** (`google-research-datasets/nq_open`) as the closed-book QA benchmark. 3,610 test items, ~87k train items. Standard for closed-book factual recall. Cleaner answer deduplication than SQuAD.
48. Train set is **subsampled to ~3,000 items** for first-pass experiments. Enough for meaningful scheduler dynamics; few enough that 25k steps gives multiple passes and forgetting dynamics are visible.
49. Multi-answer targets stored as `|||`-separated strings in the `target` field. Training always uses the first answer; exact-match scoring accepts any.
50. Scoring uses **NQ-standard normalization**: lowercase → remove articles (a/an/the) → remove punctuation → collapse whitespace, then string equality.
51. Difficulty calibration for curriculum baseline uses the **real model's pre-training loss** on each item (single forward pass before any LoRA training begins), not mock-model noise.

## 13) Generation and inference optimization

52. Added `test_batch()` to the `ModelAdapter` interface. Default implementation is sequential fallback; `RealModelAdapter` overrides with **left-padded batched generation** (`gen_batch_size=16`).
53. Trainer test phase and **retention probes** (`_retention_at_horizon`, `_retention`) both use `test_batch` instead of per-item `test()`. This is the main wall-clock optimization: retention probes can involve hundreds of eligible items per eval checkpoint, and autoregressive generation is ~10–20× slower than a single forward pass.
54. Baselines keep per-item `_probe()` for the in-loop test calls that interleave with `reinforce_update`, since those require flush semantics between each train/test pair. Only the retention snapshot probes are batched.
55. `flush()` is called once at the start of `test_batch`, not per-item. This is correct because test_batch is only called after all study/reinforce updates for that step have been applied.

## 14) Known limitations and deferred items

56. FSRS is an approximation, not canonical FSRS parameter fitting.
57. Token budget accounting still uses deterministic word-count proxy, not real tokenizer counts. This is fair across methods (same proxy for all) but not perfectly accurate.
58. No built-in plotting utility yet; JSON artifacts intended for downstream analysis.
59. No CI unit tests yet; smoke checks currently validate runnability.
60. Single-item loss computation in `test_batch` is sequential; could be batched with padding for further speedup if loss computation becomes a bottleneck (unlikely — generation dominates).

## 15) Signal quality: generation-based vs loss-based correctness

61. `scheduled_restudy` and `retrieval_practice` form a matched pair — same scheduler, same items, same gradient steps, same budget. The only difference is the correctness signal feeding the scheduler. `retrieval_practice` uses generation-based exact-match (greedy decode, then NQ-normalized string comparison). `scheduled_restudy` uses `_loss_to_correct`, which compares each item's loss to the running median of the last 200 losses.
62. The adaptive median converges to ~50% "correct" regardless of model competence, because the median splits any distribution in half by definition. As losses drop during training, the median tracks them down, so the pass rate stays ~50%. Items constantly oscillate between mastered and unmastered — 626 remastery events vs `retrieval_practice`'s 175.
63. This is not a bug in the implementation. It is evidence that this loss-to-correctness conversion produces poor scheduling decisions compared to generation-based exact-match. Consistent with KR-Test (2026), which from a different angle showed that loss/perplexity is insufficient for detecting whether a model has actually internalized factual knowledge.
64. The original framing was "does the model generating an answer before the gradient step change the quality of learning?" The more precise framing that emerged from the data: using generation-based exact-match as a scheduling signal produces better scheduling decisions than using loss-based signals, and those better decisions are what drive the learning improvement.

## 16) Why both methods can't just generate

65. If `scheduled_restudy` also generated answers, the only remaining difference would be temporal ordering of generation vs gradient step. But generation runs under `torch.no_grad()` — weights are identical before the backward pass regardless of whether generation preceded it. Same weights → same forward pass → same loss → same gradient → same update. No mechanism for ordering to matter in standard transformers. Both methods would be functionally identical.
66. This is why the comparison is generation-based signal vs loss-based signal, not "retrieval before gradient step vs no retrieval." The active ingredient is what information the scheduler receives about item-level competence, not whether generation happened.

## 17) Fixed-threshold sweep (restudy_fixed) as a follow-up ablation

67. The adaptive median is one (bad) way to convert loss into a correctness signal. The fixed-threshold sweep asks: can loss-based scoring work if calibrated better? Thresholds are computed from the model's pre-training loss distribution percentiles (p10=0.81, p25=1.36, p50=2.32, p75=3.75 on real Qwen2.5-1.5B-Instruct). The threshold doesn't move during training, so as items improve, more clear the bar and stay cleared.
68. However, mastery counts from `restudy_fixed` (994–1000) cannot be compared to `retrieval_practice` (979) because loss-based correctness is strictly easier than exact-match — a model can have low loss on correct tokens while its top prediction is a different string. Loss says "correct," exact-match says "wrong." The higher mastery counts reflect a lenient criterion, not a better signal.
69. A uniform end-of-training evaluation — exact-match generation on all items for every method — is needed to make this comparison valid. The retention probes (`_retention_at_horizon`) already use uniform exact-match via `model.test_batch()`, but at 2,000 steps they produce no data because the minimum retention horizon is 5,000 steps. Either add an explicit end-of-training eval or run at 25k+ steps to get retention probe data.

## 18) Random scheduler ablations

70. Two random schedulers test whether adaptive scheduling matters or whether just doing extra gradient steps on revisited items is sufficient. `RandomMatchedScheduler`: uniform intervals 50–500, matches FSRS's typical review frequency, ignores correctness entirely. `RandomWideScheduler`: log-uniform intervals 50–20,000, biased toward shorter intervals but items rarely come due. Both take the trainer seed for independent runs across seeds.
71. Result: FSRS beats `random_matched` by ~31 items (979 vs 948), but `random_matched` still achieves 948/1000. The protocol (generate → score → restudy) matters more than the scheduling algorithm. `random_wide` (246 mastered) confirms that review frequency matters — items need to actually come due for the protocol to work.
72. The low remastery for `random_matched` (33 vs FSRS's 175) likely reflects items not coming due often enough to detect forgetting (7,646 tests vs 12,529), not better retention. Fewer probes means fewer opportunities to observe and record a forgetting event.

## 19) Budget fairness across methods

73. Methods naturally consume different training token budgets at the same step count. `test_only` ~187k (no gradient on due items), `test_reinforce` ~235k (gradient only on failures), `retrieval_practice`/`standard_ft`/`curriculum` ~364k, replay methods ~409k (study + replay). The `--require-budget` flag enforces that `--max-training-tokens` is set.
74. Replay methods (`random_replay`, `loss_replay`) stopped early when they exceeded the 500k token budget cap at ~1,833 steps while other methods ran the full 2,000. Cross-method mastery comparisons are only valid between methods at matched training token budgets, or with per-token normalization.
75. `test_only` and `test_reinforce` comparisons to `retrieval_practice` are budget-confounded in the opposite direction — they use substantially less budget. `test_only` is the most token-efficient method (889 items at half the budget, 214 tokens per mastery), but whether it would plateau or continue climbing at matched budgets requires the 25k run.

## 20) End-of-training evaluation gap

76. The current pipeline measures mastery using each method's own correctness criterion during training. This makes mastery counts non-comparable across methods with different criteria (exact-match for `retrieval_practice`/`test_only`/`test_reinforce`, loss threshold for `restudy_fixed`, adaptive median for `scheduled_restudy`).
77. A uniform end-of-training eval pass — exact-match generation on all items for every method — is needed. The retention probes (`_retention_at_horizon`) already use uniform exact-match via `model.test_batch()`, but produce no data at 2,000 steps because the minimum retention horizon is 5,000 steps. Either add an explicit end-of-training eval or run at 25k+ steps to get retention probe data.
78. This is the most important next step for the pipeline. Without it, the `restudy_fixed` results (994–1000 mastered) cannot be meaningfully compared to `retrieval_practice` (979 mastered), and the headline claim depends on comparisons between methods that do share the same criterion (`retrieval_practice` vs `standard_ft`, `retrieval_practice` vs `scheduled_restudy`).
