# Testing Effect Pipeline

Tests whether **retrieval-conditioned scheduling** — using generation-based exact-match to decide what to review and when during LLM fine-tuning — produces better learning outcomes than passive training protocols.

The core mechanism is the scheduling signal, not the gradient step itself. Generation runs under `torch.no_grad()` and does not change weights. What changes is which items get revisited and when, driven by whether the model can actually produce the correct answer from memory.

## Methods (12)

| Method | Description |
|--------|-------------|
| `retrieval_practice` | Due item → generate answer → exact-match score → feed scheduler → gradient step |
| `scheduled_restudy` | Due item → forward-pass loss → loss-derived correctness → feed scheduler → gradient step |
| `test_only` | Due item → generate answer → score → feed scheduler → **no** gradient step on due item |
| `test_reinforce` | Like `test_only`, but failed items get a gradient step |
| `restudy_fixed_p10` | Like `scheduled_restudy`, but fixed loss threshold at 10th percentile of pre-training losses |
| `restudy_fixed_p25` | Fixed threshold at 25th percentile |
| `restudy_fixed_p50` | Fixed threshold at 50th percentile |
| `restudy_fixed_p75` | Fixed threshold at 75th percentile |
| `standard_ft` | Standard fine-tuning, no replay, no scheduling |
| `random_replay` | Standard fine-tuning + random replay of past items |
| `curriculum` | Fine-tuning with items ordered by ascending difficulty |
| `loss_replay` | Standard fine-tuning + replay prioritized by highest loss |

## Schedulers (4 + adaptive median)

| Scheduler | CLI flag | Description |
|-----------|----------|-------------|
| FSRS | `fsrs` | Adaptive half-life regression; correct → longer intervals, incorrect → shorter |
| Leitner | `leitner` | Heuristic interval growth/shrink |
| Random matched | `random_matched` | Uniform random intervals 50–500 steps, ignores correctness |
| Random wide | `random_wide` | Log-uniform random intervals 50–20,000 steps, items rarely come due |

`scheduled_restudy` also uses an internal adaptive median (running median of last 200 losses) as its correctness signal — this is not a scheduler but a signal-quality variant that feeds into whichever scheduler is selected.

## Model and dataset

- **Model**: Qwen2.5-1.5B-Instruct + LoRA (r=16, α=32, targeting `q_proj`+`v_proj`), 2.18M trainable / 1.55B total (0.14%)
- **Dataset**: NQ Open (`google-research-datasets/nq_open`), subsampled to 1,000 items for first-pass validation
- **Scoring**: NQ-standard normalization (lowercase → remove articles → remove punctuation → collapse whitespace) then exact string match
- **Mastery**: 2 consecutive correct test probes

## Headline results (3 seeds × 2,000 steps × 1,000 NQ items)

| Method | Mastered (of 1000) | Train Tokens | Remastery Events |
|--------|-------------------|--------------|-----------------|
| retrieval_practice | 979 ± 5 | 363,915 | 175 ± 11 |
| scheduled_restudy | 505 ± 21 | 364,566 | 626 ± 15 |
| standard_ft | 961 ± 6 | 363,552 | 214 ± 24 |
| test_only | 889 ± 18 | 186,758 | 284 ± 7 |
| test_reinforce | 975 ± 2 | 235,010 | 262 ± 30 |

`retrieval_practice` masters 979/1000 items vs `standard_ft`'s 961, with faster learning (885 vs 753 at step 1000) and fewer remastery events (175 vs 214). `scheduled_restudy` — same scheduler, items, gradient steps, budget — only masters 505, evidence that loss-based correctness signals are inadequate for scheduling factual recall.

### Scheduler ablation (retrieval_practice only)

| Scheduler | Mastered | Remastery |
|-----------|----------|-----------|
| FSRS | 979 ± 5 | 175 ± 11 |
| Random matched | 948 ± 5 | 33 ± 6 |
| Random wide | 246 ± 13 | 0 ± 0 |

The protocol (generate → score → restudy) matters more than the scheduling algorithm. Random scheduling still achieves 948/1000.

### Known limitation

`restudy_fixed` mastery counts (994–1000) are not directly comparable to `retrieval_practice` (979) because loss-based "correct" is strictly easier than exact-match. A model can have low loss on correct tokens while its top prediction is a different string. Uniform end-of-training exact-match evaluation across all methods is needed.

## Quick start

### Smoke test (mock model, no GPU)

```bash
python -m testing_effect_pipeline.run_experiment --steps 1000 --eval-every 200 --seeds 1
```

### Install dependencies (for real runs)

```bash
pip install -r requirements.txt
```

### Prepare the NQ Open dataset

```bash
python -m testing_effect_pipeline.prepare_nq_dataset \
  --output-dir data \
  --train-subsample 3000 \
  --hf-token YOUR_TOKEN
```

Writes `data/nq_open_train.jsonl` and `data/nq_open_test.jsonl`. Multi-answer targets stored as `|||`-separated strings.

### Run the real experiment

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_train.jsonl \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 2000 \
  --eval-every 500 \
  --seeds 3 \
  --scheduler fsrs \
  --max-training-tokens 500000 \
  --require-budget \
  --lora-r 16 \
  --lora-alpha 32 \
  --lr 2e-4 \
  --grad-accum-steps 4 \
  --dtype bfloat16 \
  --output artifacts/nq_real_experiment_metrics.json
```

The `--real` flag swaps `MockMemoryModel` for `RealModelAdapter` (LoRA fine-tuning + greedy-decode exact-match). LoRA weights are re-initialized between seeds so each seed is independent.

## Budget accounting

- **Training token budget** counts only study + reinforce updates (where gradients flow).
- **Test-phase inference tokens** tracked separately as overhead.
- `--require-budget` enforces that `--max-training-tokens` is set. Replay methods consume more tokens per step (study + replay) and stop early when they hit the cap.

## Next steps

- **25k+ step runs** for retention probe data (minimum 5k steps past mastery for the shortest horizon).
- **Uniform end-of-training eval**: exact-match generation on all items for every method, making mastery counts comparable across methods with different in-training correctness criteria.
- **Budget normalization**: per-token mastery comparison across methods with different token consumption rates.
