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


## Real-run configuration template

- Example config: `testing_effect_pipeline/real_run_config.example.json`
- Required inputs checklist: `testing_effect_pipeline/WHAT_I_NEED_FOR_REAL_RUN.md`
