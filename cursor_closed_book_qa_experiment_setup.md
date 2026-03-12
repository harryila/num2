# Closed-book QA experiment setup
_Exported on 3/11/2026 at 14:48:02 PDT from Cursor (2.4.21)_

---

**User**

Dataset: Natural Questions (closed-book subset)

Go with nq_open from HuggingFace (google-research-datasets/nq_open). It's already formatted as closed-book QA — question string in, short answer string out. Clean exact-match scoring. 3,610 test items, ~87k train items. SQuAD closed-book works too but NQ is more standard for closed-book factual recall benchmarks and has cleaner answer deduplication.

Don't use the full 87k train items — subsample to ~2,000-5,000 items for your first pass. You want enough items that the scheduler has meaningful work to do, but few enough that 25k training steps gives multiple passes and you can see forgetting dynamics clearly.

Model: Qwen2.5-1.5B-Instruct

Reasons: small enough to run 6 methods × 3 seeds × 25k steps on a single GPU without it taking a week. Large enough that results are credible and reviewers won't dismiss it as a toy. Instruction-tuned so it can generate answers in the test phase without heavy prompt engineering. Qwen2.5 family is well-regarded right now and the 1.5B size is the sweet spot for your experiment matrix.



For the real model adapter, you'll need: HF Transformers + PEFT (LoRA fine-tuning, not full fine-tune — way cheaper and you can run more seeds). Use trl's SFTTrainer internals or just write a minimal training loop with AdamW + LoRA. For the test phase, greedy decode with max 32 new tokens, then exact-match against the normalized target.


hf token:
<YOUR_HF_TOKEN>

no need to run js do the implemntation of all that 

---

**Cursor**

Let me start by understanding the existing codebase structure and then implement everything needed for the real run.

Now let me read the complete files I need to understand the full implementation details.

Now I have a thorough understanding of the entire codebase. Let me implement everything needed for the real run with NQ Open + Qwen2.5-1.5B-Instruct + LoRA.

Now the core piece -- the real model adapter with LoRA.

Now update the existing files -- add `flush()` to `ModelAdapter`, wire real model into the runner, and ensure trainers flush at the end.

Now the big update -- rewrite `run_experiment.py` with full real-model support.

Now let me check for linter errors across all the files I touched.

Let me do a quick sanity check that the mock path still works (imports resolve, no breakage from the changes).

Good -- imports work, `flush()` no-op is fine, and the NQ normalization correctly removes articles/punctuation. Let me also verify the mock experiment still runs end-to-end.

Mock path is clean. Now let me verify the real model imports resolve (without actually loading the model, just checking the import chain).

Expected -- `peft` isn't installed locally. Let me verify the Python syntax of all new files is valid.

All files parse cleanly. Now let me update the README with the real-run instructions.

Now let me do a final review of all the files to make sure everything is consistent and correct.

One small cleanup -- `field` is imported but unused in `real_model.py`.

Everything is implemented and verified. Here's a summary of what was built:

---

**New files created:**

- **`requirements.txt`** -- torch, transformers, peft, datasets, accelerate
- **`testing_effect_pipeline/nq_eval.py`** -- NQ-standard answer normalization (lowercase, remove articles/punctuation, collapse whitespace) and exact-match scoring against multiple acceptable answers
- **`testing_effect_pipeline/prepare_nq_dataset.py`** -- Downloads `google-research-datasets/nq_open` from HuggingFace, subsamples train to 3000 items (configurable), writes train + test JSONL with `|||`-separated multi-answer targets
- **`testing_effect_pipeline/real_model.py`** -- `RealModelAdapter` implementing the `ModelAdapter` interface:
  - Loads **Qwen2.5-1.5B-Instruct** in bfloat16
  - Applies **LoRA** (r=16, alpha=32) targeting `q_proj` + `v_proj` via PEFT
  - **study_update / reinforce_update**: forward + backward with loss masked to answer tokens only, internal gradient accumulation (default 4 items per optimizer step), AdamW with grad clipping
  - **test**: flushes pending gradients, greedy decode (max 32 tokens), NQ exact-match against all `|||`-separated targets, returns `(correct, loss)`
  - **reset_adapter**: re-initializes LoRA weights (Kaiming for A, zeros for B) + fresh optimizer, enabling multi-seed runs without reloading the 1.5B base model
  - **compute_loss**: forward-only loss for difficulty calibration before training

**Files updated:**

- **`model.py`** -- Added `flush()` default no-op to `ModelAdapter` base class
- **`trainer.py`** + **`baselines.py`** -- Call `model.flush()` at end of `train()` to drain any pending accumulated gradients
- **`run_experiment.py`** -- Added `--real` flag plus all LoRA/model CLI args. When `--real` is set: loads model once, calibrates difficulty from base model loss, resets LoRA per seed x method combination
- **`real_run_config.example.json`** -- Updated for NQ Open + Qwen2.5 + 25k steps
- **`README.md`** -- Added full real-run instructions (install, prepare dataset, launch experiment)

---

**To run on your GPU box:**

```bash
pip install -r requirements.txt

# Step 1: prepare dataset
python -m testing_effect_pipeline.prepare_nq_dataset \
  --output-dir data --train-subsample 3000 \
  --hf-token <YOUR_HF_TOKEN>

# Step 2: run experiment
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_train.jsonl \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --hf-token <YOUR_HF_TOKEN> \
  --steps 25000 --eval-every 1000 --seeds 3 \
  --scheduler fsrs --max-training-tokens 500000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --output artifacts/nq_real_experiment_metrics.json
```

---

**User**

This is ready to run. The real model adapter is well-implemented. A few notes:
The _train_on_item implementation is correct and careful. Loss masked to answer tokens only (line 189: labels[0, :prompt_len] = -100), grad accumulation with proper loss scaling (line 193: loss / grad_accum_steps), grad clipping, flush before test — all right.
The reset_adapter approach is smart. Loading a 1.5B base model once and just re-initializing LoRA weights + optimizer between seeds avoids reloading from disk 18 times (6 methods × 3 seeds). That'll save significant wall-clock time.
One thing to watch on your Lambda box: the test phase calls model.generate() with greedy decode on every due item, every step. At 25k steps with potentially hundreds of due items, this will be your bottleneck — generation is much slower than forward/backward on single items. Monitor your actual steps-per-second during the first 1k steps. If it's too slow, you may need to either batch the generation calls (right now it's one item at a time in the test loop) or reduce eval frequency early on. The forgetting probe (_retention_at_horizon) also generates on every eligible item at every eval checkpoint — that could be hundreds of generate calls every 1k steps.
The study_update / reinforce_update distinction: right now they're identical (both call _train_on_item). That's fine for v1 — you're not claiming reinforcement uses a different loss. But note this means the test_reinforce variant is functionally "test + re-study on failures," not RL-style reinforcement. That's actually the cleaner experiment — just document it clearly.
Go run it. Start with --seeds 1 --steps 2000 --eval-every 500 on the Lambda box to verify the real model path works end-to-end before committing to the full 25k × 3-seed matrix.

---

**Cursor**

Good notes. The generation bottleneck is real and worth fixing before you burn GPU hours. Let me implement batched generation in the real adapter and wire it through the trainer/baselines retention probes, then document the study/reinforce equivalence.

Now I have full context. Let me implement the batched generation optimization, the trainer refactors, and the documentation update.

Now add the batched generation methods to `RealModelAdapter`.

Now refactor the trainer to use `test_batch` for the test phase and retention probes.

Now refactor the main test loop in the trainer's `train()` method to use `test_batch`.

Now the baselines -- batch the retention probe (keep `_probe` as-is since it interleaves with reinforce_update).

Good. Now verify the mock path still works end-to-end after the refactor.

All 6 methods x 2 seeds clean. Let me also verify the metrics structure is intact (retention probes producing values).

Now update DESIGN_CHOICES.md with the new decisions.

Let me do one final lint check and syntax verification on all touched files.

All clean. Here's what changed:

**Batched generation** -- the main performance fix for your GPU run:

- `ModelAdapter.test_batch(items)` added to the base class with a sequential fallback (mock model uses this).
- `RealModelAdapter.test_batch()` overrides with: sequential per-item loss computation (fast single forward passes) + **left-padded batched greedy generation** (`gen_batch_size=16` configurable). This is where the wall-clock win lives -- instead of 500 sequential `model.generate()` calls in a retention probe, you get ~31 batched calls.
- `TestingEffectTrainer` now uses `test_batch` for both the per-step test phase and `_retention_at_horizon` probes.
- `BaselineTrainer._retention()` also uses `test_batch`. The in-loop `_probe()` stays sequential because it interleaves with `reinforce_update` (needs flush semantics per item).

**Documentation** -- DESIGN_CHOICES.md now has sections 11-14 covering:
- study/reinforce equivalence (both call `_train_on_item` -- this is "test + re-study on failures," not RL reinforcement)
- NQ Open dataset rationale
- Qwen2.5 + LoRA configuration rationale
- Batched generation design and where flush semantics constrain it

For your validation run on the Lambda box:

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_train.jsonl \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --hf-token <YOUR_HF_TOKEN> \
  --steps 2000 --eval-every 500 --seeds 1 \
  --scheduler fsrs --max-training-tokens 500000 --require-budget \
  --output artifacts/nq_validation_run.json
```

