# Experiment Runs

## Set 1: Random 50k subsample (Qwen2.5-0.5B + LoRA)

Tests `retrieval_practice` vs `standard_ft` under capacity pressure. Smaller model (0.5B vs 1.5B) and much larger dataset (50k vs 1k) so the LoRA can't trivially memorize. Two LoRA ranks (r=8 and r=16), each method on its own GPU instance, 4 GPUs total in parallel.

### Step 1: Prepare dataset (run once on any machine, ~5 min)

```bash
python -m testing_effect_pipeline.prepare_nq_dataset \
  --output-dir data \
  --output-name nq_open_50k_random.jsonl \
  --train-subsample 50000 \
  --seed 42 \
  --hf-token YOUR_TOKEN
```

Then ship `data/nq_open_50k_random.jsonl` to all 4 GPU instances (scp, rsync, whatever).

### Step 2: Create smoke-test subset (one-time per machine, ~1 sec)

The 50k JSONL is too large for smoke testing because each uniform eval pass takes ~50 minutes. Slice off the first 1k items for a fast smoke test:

```bash
head -n 1000 data/nq_open_50k_random.jsonl > data/nq_open_smoke.jsonl
```

### Step 3: GPU smoke test (run on each instance before the full job, ~5-10 min, ~$0.50)

Validates that the model loads, LoRA initializes, training runs, periodic uniform eval fires, and end-of-training eval fires.

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_smoke.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 200 --eval-every 100 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 200000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice standard_ft \
  --output artifacts/smoke_test_set1_gpu.json
```

### Step 4: Launch full runs (4 GPU instances in parallel, ~17-21 hours each)

Each GPU runs one (config, method) pair. Output filenames are distinct so the 4 JSONs don't collide when copied back to one machine for analysis.

#### Instance 1: r=8, retrieval_practice

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_50k_random.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 20000 --eval-every 2000 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 5000000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice \
  --output artifacts/set1_50k_r8_retrieval.json
```

#### Instance 2: r=8, standard_ft

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_50k_random.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 20000 --eval-every 2000 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 5000000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods standard_ft \
  --output artifacts/set1_50k_r8_standard.json
```

#### Instance 3: r=16, retrieval_practice

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_50k_random.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 20000 --eval-every 2000 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 5000000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice \
  --output artifacts/set1_50k_r16_retrieval.json
```

#### Instance 4: r=16, standard_ft

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_50k_random.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 20000 --eval-every 2000 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 5000000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods standard_ft \
  --output artifacts/set1_50k_r16_standard.json
```

### Step 5: Collect results

scp / rsync the 4 JSON files back to one machine:

- `artifacts/set1_50k_r8_retrieval.json`
- `artifacts/set1_50k_r8_standard.json`
- `artifacts/set1_50k_r16_retrieval.json`
- `artifacts/set1_50k_r16_standard.json`

For analysis, the two methods at each rank can be merged into a single dict keyed by method (matches the format the runner produces when both methods are launched together).

## Notes

### Wall-clock per GPU

Each GPU instance runs one method on one config. Plan for ~17-21 hours wall-clock total: ~8-10 hours of training plus ~9-11 hours of uniform eval (11 passes at ~50 min each on the 0.5B model with 50k items). Don't kill it at hour 20 thinking it's stuck.

### Budget cap

The 5M training token cap is non-binding for `retrieval_practice` and `standard_ft` at this config. Expected consumption is ~3.6M (20k steps x ~180 tokens). The cap exists as a fail-fast guard rail in case a method drifts unexpectedly, not as a binding constraint. If you want it to actually bind, drop to 4M.

### Periodic uniform eval

Already wired up in [testing_effect_pipeline/uniform_eval.py](testing_effect_pipeline/uniform_eval.py). At every `--eval-every` checkpoint, the trainer runs exact-match generation on all items and stores a summary. End-of-training also runs once with full per-item results. With `--eval-every 2000 --steps 20000`, that's 10 periodic checkpoints + 1 end-of-training = 11 uniform eval data points per run.

### Why this config

- **0.5B vs 1.5B model**: less parametric knowledge, more learning headroom
- **50k vs 1k items**: forces capacity pressure, prevents trivial memorization
- **20k vs 4k steps**: with 50k items at batch 16, ~6 average touches per item
- **r=8 alongside r=16**: tighter LoRA capacity should sharpen any retrieval_practice advantage if it exists

## Set 2 dataset prep: hard-filter NQ Open

Filters the base model's failures into a 10k training subset. Run this on a
single GPU instance. Should take ~45-90 minutes.

The filter shuffles NQ Open's full ~87k train split with `seed=42` (same seed
as Set 1's random subsample, so the two sets sample from the same distribution),
then runs the bare base model (no LoRA) greedy-decode on each item in batches.
Items the model gets wrong go to the "hard" JSONL; correct ones go to "known".
Stops at 10k unknowns. Resumable: state file tracks counters and the last
processed shuffled-index, so a kill+restart picks up where it left off.

### Step 1: GPU smoke test (~2 min)

Confirms the filter pipeline runs end-to-end before launching the long job.

```bash
python -m testing_effect_pipeline.filter_nq_unknown \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --target-unknown 50 \
  --batch-size 16 \
  --output-unknown data/nq_open_hard_smoke.jsonl \
  --output-known data/nq_open_known_smoke.jsonl \
  --state-path data/nq_open_filter_state_smoke.json \
  --seed 42 \
  --dtype bfloat16
```

Expect a handful of progress lines like:

```
Processed 64 | Known 21 | Unknown 43/50 | Failure rate: 67.2% | 12.3 items/s
```

Then a final `complete=true` state file with `total_unknown >= 50`. Delete the
smoke artifacts before running Step 2:

```bash
rm data/nq_open_hard_smoke.jsonl data/nq_open_known_smoke.jsonl data/nq_open_filter_state_smoke.json
```

### Step 2: Run the full filter

```bash
python -m testing_effect_pipeline.filter_nq_unknown \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --target-unknown 10000 \
  --batch-size 16 \
  --output-unknown data/nq_open_hard_10k.jsonl \
  --output-known data/nq_open_known.jsonl \
  --state-path data/nq_open_filter_state.json \
  --seed 42 \
  --dtype bfloat16
```

The state JSON is human-readable; `cat data/nq_open_filter_state.json` at any
time to check progress mid-run.

If the job dies (preemption, OOM, network blip), just re-run the same command.
The script reads the state file, validates that `--model-name` and `--seed`
match, opens the JSONLs in append mode, and resumes from `last_processed_index + 1`
in the same shuffled order.

### Step 3: Sanity check

Confirm 10k items in the unknown JSONL and check the known/unknown ratio:

```bash
wc -l data/nq_open_hard_10k.jsonl data/nq_open_known.jsonl
cat data/nq_open_filter_state.json
```

Expected: ~10k unknowns plus a few thousand knowns (depending on Qwen2.5-0.5B
zero-shot accuracy on NQ Open, which is typically 15-25%, so ~2-5k knowns
collected by the time we hit 10k unknowns).

### Step 4: Use the dataset for Set 2 training

Same launch commands as Set 1, but swap the dataset path and rescale steps +
budget for the smaller (10k vs 50k) dataset:

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_hard_10k.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 4000 --eval-every 500 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 1000000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice \
  --output artifacts/set2_hard_10k_r8_retrieval.json
```

Run the same four-instance fan-out as Set 1 (r=8/r=16 x retrieval_practice/standard_ft),
swapping output names to `artifacts/set2_hard_10k_r{8,16}_{retrieval,standard}.json`.

### Notes

- Don't launch Set 2 training until Set 1 has finished or been canceled. The
  filter run itself can happen in parallel with Set 1 since it only needs one
  GPU briefly.
- The filter uses the same chat template, system prompt, greedy decode settings,
  and NQ exact-match scorer as the training-time eval, so an item that fails
  the filter is failing under the same ruler training will be measured against.
