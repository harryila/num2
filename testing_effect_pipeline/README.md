# Testing Effect Pipeline (Implementation)

## Run smoke experiment

```bash
python -m testing_effect_pipeline.run_experiment --steps 1000 --eval-every 200 --seeds 1
```

This writes metrics to `artifacts/experiment_metrics.json`.

## Run with explicit training-token budget guardrail

```bash
python -m testing_effect_pipeline.run_experiment \
  --steps 20000 \
  --max-training-tokens 200000 \
  --seeds 3
```

Methods stop early if they exceed `--max-training-tokens`.

## Use custom closed-book dataset

```bash
python -m testing_effect_pipeline.run_experiment \
  --dataset-path data/closed_book_squad.jsonl \
  --steps 10000 \
  --seeds 3 \
  --scheduler fsrs
```

Expected jsonl fields per row:
- `item_id`
- `prompt`
- `target`
- optional `domain_tag`
- optional `difficulty`

If `difficulty` is missing, curriculum uses model-relative first-exposure loss for ranking.

## Budget accounting policy

- **Training token budget** counts only study + reinforce updates (where gradients flow).
- **Test-phase inference tokens** are tracked separately as overhead and reported in metrics.


## Real run: NQ Open + Qwen2.5-1.5B-Instruct + LoRA

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the NQ Open dataset

```bash
python -m testing_effect_pipeline.prepare_nq_dataset \
  --output-dir data \
  --train-subsample 3000 \
  --hf-token YOUR_TOKEN
```

This writes `data/nq_open_train.jsonl` (3 000 items) and `data/nq_open_test.jsonl` (3 610 items).
Multi-answer targets are stored as `|||`-separated strings.

### 3. Run the real experiment

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_train.jsonl \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 25000 \
  --eval-every 1000 \
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

The `--real` flag swaps `MockMemoryModel` for `RealModelAdapter` (LoRA fine-tuning + greedy-decode exact-match).
LoRA weights are re-initialised between seeds so each seed is independent.

### Real-run configuration template

- Example config: `testing_effect_pipeline/real_run_config.example.json`
