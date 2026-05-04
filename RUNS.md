# Experiment Runs

## Set 1: Random 50k subsample (Qwen2.5-0.5B + LoRA)

Tests `retrieval_practice` vs `standard_ft` under capacity pressure. Smaller model (0.5B vs 1.5B) and much larger dataset (50k vs 1k) so the LoRA can't trivially memorize. Two LoRA ranks (r=8 and r=16), each method on its own GPU instance, **4 independent GPUs running in parallel**.

### GPU layout

| GPU | LoRA rank | Method | Output JSON |
|-----|-----------|--------|-------------|
| 1 | r=8, alpha=16 | retrieval_practice | `set1_50k_r8_retrieval.json` |
| 2 | r=8, alpha=16 | standard_ft | `set1_50k_r8_standard.json` |
| 3 | r=16, alpha=32 | retrieval_practice | `set1_50k_r16_retrieval.json` |
| 4 | r=16, alpha=32 | standard_ft | `set1_50k_r16_standard.json` |

Hardware: RTX 4090 (24GB) per GPU is the recommended choice. Plan ~17-21 hours wall-clock per GPU (~$12-14 per GPU at $0.69/hr, ~$50-56 total).

---

## Common setup (run on each of the 4 GPUs)

These steps are identical on every instance. Repeat them on each fresh GPU before the per-GPU launch.

### Step A: Open a tmux session

The full run takes 17-21 hours. **Always run inside tmux** so the run survives ssh disconnects, laptop sleep, network blips, etc.

```bash
tmux new -s nq50k
```

If you ever get disconnected, ssh back in and reattach:

```bash
tmux attach -t nq50k
```

To detach without killing the session: press `Ctrl+b` then `d`.

### Step B: Clone the repo and install dependencies

```bash
git clone https://github.com/harryila/num2.git && cd num2
pip install -r requirements.txt
```

### Step C: Prepare the dataset (~5 min)

This pulls NQ Open from HuggingFace and writes a random 50k-item subsample to `data/nq_open_50k_random.jsonl`. The sampling is deterministic across machines because seed=42, so all 4 GPUs end up with the exact same 50k items.

```bash
python -m testing_effect_pipeline.prepare_nq_dataset \
  --output-dir data \
  --output-name nq_open_50k_random.jsonl \
  --train-subsample 50000 \
  --seed 42 \
  --hf-token YOUR_TOKEN
```

How the sampling works (verified): the script shuffles all ~87k train indices with `random.Random(seed=42)`, takes the first 50k from the shuffled list, then sorts those for nice output. It is NOT taking the first 50k items from the dataset.

### Step D: Make the smoke-test subset (~1 sec)

The 50k JSONL is too big for smoke testing because each uniform eval pass takes ~50 minutes on it. Slice off the first 1k items for fast smoke validation:

```bash
head -n 1000 data/nq_open_50k_random.jsonl > data/nq_open_smoke.jsonl
```

(For smoke testing we don't care about randomness within the subset, only that the pipeline runs end-to-end.)

---

## Per-GPU instructions

Run **only the section matching this GPU**. Each section has the smoke test (narrowed to that GPU's single method) and the full run command. Each writes a distinct output filename so the 4 JSONs don't collide when copied back.

---

### GPU 1: r=8, retrieval_practice

#### Smoke test (~5-10 min, ~$0.50)

Verifies model loads, LoRA initializes, training step runs, periodic uniform eval fires, end-of-training eval fires.

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_smoke.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 200 --eval-every 100 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 200000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods retrieval_practice \
  --output artifacts/smoke_test_set1_r8_retrieval.json
```

#### Full run (~17-21 hours)

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
  --output artifacts/set1_50k_r8_retrieval.json 2>&1 | tee artifacts/set1_50k_r8_retrieval.log
```

The `2>&1 | tee ...log` captures all stdout/stderr to a log file alongside the JSON, useful for tailing progress and debugging if anything goes wrong.

---

### GPU 2: r=8, standard_ft

#### Smoke test

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_smoke.jsonl \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 200 --eval-every 100 --seeds 1 \
  --scheduler fsrs \
  --max-training-tokens 200000 --require-budget \
  --lora-r 8 --lora-alpha 16 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --methods standard_ft \
  --output artifacts/smoke_test_set1_r8_standard.json
```

#### Full run

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
  --output artifacts/set1_50k_r8_standard.json 2>&1 | tee artifacts/set1_50k_r8_standard.log
```

---

### GPU 3: r=16, retrieval_practice

#### Smoke test

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
  --methods retrieval_practice \
  --output artifacts/smoke_test_set1_r16_retrieval.json
```

#### Full run

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
  --output artifacts/set1_50k_r16_retrieval.json 2>&1 | tee artifacts/set1_50k_r16_retrieval.log
```

---

### GPU 4: r=16, standard_ft

#### Smoke test

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
  --methods standard_ft \
  --output artifacts/smoke_test_set1_r16_standard.json
```

#### Full run

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
  --output artifacts/set1_50k_r16_standard.json 2>&1 | tee artifacts/set1_50k_r16_standard.log
```

---

## Collecting results

After all 4 GPUs finish, scp / rsync the 4 JSON files (and optionally the log files) back to one machine for analysis:

- `artifacts/set1_50k_r8_retrieval.json`
- `artifacts/set1_50k_r8_standard.json`
- `artifacts/set1_50k_r16_retrieval.json`
- `artifacts/set1_50k_r16_standard.json`

Each JSON has the same structure as previous runs (`seed_0` -> method -> metrics). Since each GPU only ran one method, each JSON has just one method key. For analysis, the two methods at each rank can be merged into a single dict keyed by method.

---

## Notes

### Wall-clock per GPU

Each GPU runs one method on one config. Plan for ~17-21 hours wall-clock total: ~8-10 hours of training plus ~9-11 hours of uniform eval (11 passes at ~50 min each on the 0.5B model with 50k items). Don't kill the run at hour 18 thinking it's stuck. Tail the log file to confirm progress:

```bash
tail -f artifacts/set1_50k_<config>_<method>.log
```

### Budget cap is non-binding

The 5M training token cap will not bind for either method at this config. Expected consumption is ~3.6M (20k steps x ~180 tokens). The cap exists as a fail-fast guard rail in case a method drifts unexpectedly. If you want it to actually bind, drop to 4M.

### Periodic uniform eval

Already wired up in [testing_effect_pipeline/uniform_eval.py](testing_effect_pipeline/uniform_eval.py). At every `--eval-every` checkpoint, the trainer runs exact-match generation on all items and stores a summary. End-of-training also runs once with full per-item results. With `--eval-every 2000 --steps 20000`, that's 10 periodic checkpoints + 1 end-of-training = 11 uniform eval data points per run.

### Why this config

- **0.5B vs 1.5B model**: less parametric knowledge, more learning headroom
- **50k vs 1k items**: forces capacity pressure, prevents trivial memorization
- **20k vs 4k steps**: with 50k items at batch 16, ~6 average touches per item
- **r=8 alongside r=16**: tighter LoRA capacity should sharpen any retrieval_practice advantage if it exists
- **One method per GPU**: parallelizes the run across 4 GPUs, finishes in ~20 hours instead of ~40 hours
