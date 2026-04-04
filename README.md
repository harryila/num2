# Retrieval-Conditioned Scheduling for LLM Fine-Tuning

Think about how you'd study for an exam. You could re-read your notes over and over, or you could close the book and try to answer questions from memory. When you get one wrong, you know to come back to it sooner. When you get one right a few times in a row, you can safely move on and trust that you'll remember it. The act of testing yourself does two things: it tells you what you actually know vs what you only think you know, and it tells you where to spend your limited study time. That cycle of test yourself, check the answer, decide what to revisit next is what this project does for LLM fine-tuning.

Standard fine-tuning is entirely passive. The model sees the answer, computes loss on it, updates weights, and moves on. It never attempts to produce the answer first. There's no mechanism for "do I actually know this?" to influence what gets trained next. Every item is treated the same regardless of whether the model has already learned it or is still struggling with it.

This project adds that mechanism. Before training on an item, the model tries to answer it from memory via greedy generation. The result (right or wrong) feeds a spaced-repetition scheduler that decides when to revisit the item. If the model gets it right, the scheduler pushes the next review further into the future. If it gets it wrong, the scheduler brings the review closer. The generation itself doesn't touch the weights (it runs under `torch.no_grad()`). What changes is the scheduling signal: which items get revisited and when, based on whether the model can actually produce the correct answer.

## Methods

There are 12 methods total. Four use a scheduler to decide when items come due for review, four are fixed-threshold ablations that test a different way of measuring correctness, and four are baselines that don't use any scheduler at all.

**Retrieval-conditioned methods**

These methods use a spaced-repetition scheduler. At each training step, the scheduler decides which items are "due" for review based on their history of correct and incorrect answers.

| Method | What happens when an item comes due |
|--------|-------------------------------------|
| `retrieval_practice` | The model generates an answer via greedy decoding, and the generated string is checked against the target using NQ-standard exact-match normalization. The binary result (correct or incorrect) feeds the scheduler, which updates the item's next review time. Then a gradient step is applied on that item. This is the core method. |
| `scheduled_restudy` | Instead of generating, the model does a forward pass and computes the loss on the target. That loss is converted into a binary correctness signal by comparing it to the running median of the last 200 losses. If below median, "correct." If above, "incorrect." That binary result feeds the same scheduler. Then a gradient step is applied. This is the control condition for `retrieval_practice`. |
| `test_only` | Same as `retrieval_practice` (generate, check, update scheduler) but the gradient step on due items is skipped entirely. The model only trains on new items from the study queue. This isolates the scheduling signal. If `test_only` still improves learning, it means the scheduling alone is valuable even without the extra gradient steps on reviewed items. |
| `test_reinforce` | Same as `test_only`, but items the model gets wrong also receive a gradient step. Items it gets right are left alone. This sits between `test_only` and `retrieval_practice` in terms of training budget. |

`retrieval_practice` and `scheduled_restudy` are designed as a matched pair. They use the same scheduler, see the same items, do the same number of gradient steps, and consume the same training token budget. The only thing that differs is how "correct" is determined: generation-based exact-match vs loss-based inference. This isolates the question of whether the correctness signal matters.

**Fixed-threshold ablations**

These work the same way as `scheduled_restudy` but replace the adaptive running median with a fixed loss threshold. The adaptive median in `scheduled_restudy` has a fundamental problem: it always classifies roughly 50% of items as "correct" because the median splits any distribution in half. As the model improves and all losses drop, the median drops with them, so the pass rate stays around 50% no matter what. The fixed-threshold variants ask: does loss-based scoring work better when the threshold is calibrated once and then held constant?

| Method | Threshold source |
|--------|-----------------|
| `restudy_fixed_p10` | 10th percentile of the model's pre-training loss distribution. This is the strictest threshold (0.81). An item must reach a very low loss to be considered "correct." |
| `restudy_fixed_p25` | 25th percentile (threshold = 1.36) |
| `restudy_fixed_p50` | 50th percentile (threshold = 2.32) |
| `restudy_fixed_p75` | 75th percentile (threshold = 3.75). This is the most lenient. Most items will clear this bar quickly during training. |

The thresholds are computed from the model's loss distribution before any fine-tuning begins and stay fixed throughout training. As items improve, more of them clear the bar and stay cleared, which avoids the oscillation problem of the adaptive median.

**Baselines**

These methods don't use any scheduler or retrieval probing. They represent the standard approaches to fine-tuning.

| Method | Description |
|--------|-------------|
| `standard_ft` | The simplest baseline. The model trains on items in round-robin order. A random sample of items is probed periodically to track mastery, but the probe results don't influence what gets trained next. This is how most fine-tuning works. |
| `random_replay` | Same as `standard_ft` but after each study batch, a random sample of 8 past items is replayed with additional gradient steps. This tests whether simply revisiting old items helps, without any intelligence about which items to revisit. |
| `curriculum` | Items are presented in ascending difficulty order (easiest first). Difficulty is calibrated from the model's pre-training loss on each item before fine-tuning begins. No replay, no scheduling. |
| `loss_replay` | Same as `random_replay`, but instead of replaying random items, it replays the items with the highest current loss. The idea is to focus extra training on what the model is worst at. |

**Schedulers**

The retrieval-conditioned methods and fixed-threshold ablations all need a scheduler to decide when items come due. There are four options:

| Scheduler | CLI flag | Description |
|-----------|----------|-------------|
| FSRS | `fsrs` | An adaptive half-life model inspired by the Free Spaced Repetition Scheduler from the research literature. When the model gets an item correct, the next review is pushed further into the future. When it gets one wrong, the review is pulled closer. The intervals adapt to each item's individual difficulty based on its correctness history. This is the default for real runs. |
| Leitner | `leitner` | A simpler heuristic approach. Intervals grow by a fixed multiplier on success and shrink on failure. Less principled than FSRS but computationally cheaper. Used as the default for mock-model smoke tests. |
| Random matched | `random_matched` | Assigns random intervals between 50 and 500 steps, uniformly sampled. Completely ignores correctness. Designed to match FSRS's typical review rate so that comparing the two isolates whether scheduling intelligence matters or whether just reviewing items at a reasonable rate is enough. |
| Random wide | `random_wide` | Assigns random intervals between 50 and 20,000 steps using a log-uniform distribution. Items rarely come due because most intervals are very long. This is a sanity check: if review frequency doesn't matter, `random_wide` should perform similarly to `random_matched`. It doesn't, which confirms that items need to actually come due at a reasonable rate. |

## Setup

Install dependencies (only needed for real model runs, not the mock smoke test):

```bash
pip install -r requirements.txt
```

Download and prepare the NQ Open dataset from HuggingFace. This subsamples the ~87k training items down to 3,000 and writes them as JSONL files with stable item IDs:

```bash
python -m testing_effect_pipeline.prepare_nq_dataset \
  --output-dir data --train-subsample 3000 --hf-token YOUR_TOKEN
```

This produces `data/nq_open_train.jsonl` (3,000 items) and `data/nq_open_test.jsonl` (3,610 items). Multi-answer targets are stored as `|||`-separated strings. Training always uses the first answer; exact-match scoring accepts any of them.

Smoke test with the mock model (no GPU needed, runs in seconds):

```bash
python -m testing_effect_pipeline.run_experiment --steps 1000 --eval-every 200 --seeds 1
```

Real run on GPU with Qwen2.5-1.5B-Instruct and LoRA:

```bash
python -m testing_effect_pipeline.run_experiment \
  --real \
  --dataset-path data/nq_open_train.jsonl \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --hf-token YOUR_TOKEN \
  --steps 2000 --eval-every 500 --seeds 3 \
  --scheduler fsrs \
  --max-training-tokens 500000 --require-budget \
  --lora-r 16 --lora-alpha 32 --lr 2e-4 --grad-accum-steps 4 \
  --dtype bfloat16 \
  --output artifacts/nq_real_experiment_metrics.json
```

The `--real` flag switches from the mock memory model to the real model adapter. The real adapter loads Qwen2.5-1.5B-Instruct in bfloat16, applies LoRA adapters to the attention layers, and uses a minimal AdamW training loop with gradient accumulation. LoRA weights and the optimizer are re-initialized between each seed so every seed starts from the same base model.

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--real` | off | Use the real model adapter (Qwen + LoRA) instead of the mock memory model. Without this flag, the pipeline runs with a lightweight mock that simulates forgetting and learning without any GPU. |
| `--dataset-path` | none | Path to a JSONL dataset file. Each line needs `item_id`, `prompt`, and `target` fields. If this is omitted, the pipeline generates a small synthetic dataset for smoke testing. |
| `--sample-size` | 200 | How many synthetic items to generate when no dataset path is given. Only relevant for mock runs. |
| `--model-name` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model identifier or a local path to model weights. Only used when `--real` is set. |
| `--hf-token` | none | HuggingFace API token. Needed if the model is gated or if you're downloading the NQ Open dataset. |
| `--steps` | 5000 | Total number of training steps to run. Each step processes one batch of items. For retention probes to produce meaningful data at the 5k/10k/20k horizons, this needs to be at least 25k. |
| `--batch-size` | 16 | Number of items processed per training step. For retrieval-conditioned methods, the batch is split dynamically between study items, test items, and reinforcement items based on scheduler pressure. |
| `--eval-every` | 1000 | How often to run evaluation snapshots and retention probes, measured in steps. Each snapshot records cumulative mastery, token usage, and retention at multiple horizons. |
| `--seeds` | 3 | Number of independent runs per method. Each seed gets fresh LoRA weights and a fresh optimizer. Results are averaged across seeds for the reported mean and standard deviation. |
| `--scheduler` | `leitner` | Which spaced-repetition scheduler to use. Options: `fsrs`, `leitner`, `random_matched`, `random_wide`. Only affects the retrieval-conditioned methods and fixed-threshold ablations. |
| `--methods` | all 12 | Space-separated list of which methods to run. Defaults to all 12. You can pass a subset like `--methods retrieval_practice standard_ft` to run only those two. |
| `--max-training-tokens` | none | Hard cap on training tokens. Only counts tokens from gradient-bearing operations (study and reinforce updates). When a method exceeds this cap, it stops early. This ensures fair cross-method comparisons. |
| `--require-budget` | off | Makes the pipeline error out immediately if `--max-training-tokens` is not set. Use this for real experiments to prevent accidentally running methods with no budget cap, which produces misleading comparisons because replay methods consume more tokens per step. |
| `--mock-noise-std` | 0.05 | Controls how noisy the mock model's retrieval outcomes are. Higher values make the mock more stochastic. Only relevant when `--real` is not set. |
| `--lora-r` | 16 | LoRA rank. Higher rank means more trainable parameters and more capacity, but also more memory and compute. 16 is a reasonable default for 1.5B models. |
| `--lora-alpha` | 32 | LoRA alpha scaling factor. Controls the effective learning rate of the LoRA layers relative to the base model. Typically set to 2x the rank. |
| `--lr` | 2e-4 | Learning rate for the AdamW optimizer. Only applies to the LoRA parameters; the base model is frozen. |
| `--grad-accum-steps` | 4 | How many items to accumulate gradients over before doing an optimizer step. The pipeline processes items one at a time through `study_update`, so this bridges the gap to a reasonable effective batch size. |
| `--max-seq-len` | 256 | Maximum sequence length for tokenization. Prompts and targets that exceed this are truncated. NQ Open items are short enough that 256 is plenty. |
| `--max-new-tokens` | 32 | Maximum number of tokens the model can generate during test-phase greedy decoding. NQ Open answers are typically 1 to 5 tokens, so 32 is generous. |
| `--gen-batch-size` | 16 | Batch size for batched generation during the test phase and retention probes. Larger values are faster but use more GPU memory. The pipeline collects all items that need generation and processes them in batches of this size. |
| `--dtype` | `bfloat16` | Precision for model weights. Options: `float32`, `float16`, `bfloat16`. bfloat16 is recommended for Ampere and newer GPUs. Uses half the memory of float32 with minimal accuracy loss. |
| `--output` | `artifacts/experiment_metrics.json` | Where to write the output JSON file containing all metrics, snapshots, allocation logs, and remastery events for every seed and method. |

## Results

These results are from real model runs, not the mock model.

Configuration: Qwen2.5-1.5B-Instruct with LoRA adapters (r=16, alpha=32, targeting q_proj and v_proj attention layers). That's 2.18M trainable parameters out of 1.55B total, or about 0.14% of the model. The dataset is NQ Open (Google Natural Questions, closed-book subset), subsampled to 1,000 items. Each run used 3 seeds, 2,000 training steps, the FSRS scheduler, and a 500k training token budget cap.

Mastery is defined as answering an item correctly on 2 consecutive test probes. Remastery counts how many times an item was mastered, then forgotten (failed a subsequent probe), then re-mastered. A high remastery count means items aren't sticking. Scoring uses NQ-standard normalization: lowercase the string, remove articles (a/an/the), strip punctuation, collapse whitespace, then check for exact string equality.

| Method | Mastered (of 1000) | Training Tokens | Remastery Events |
|--------|-------------------|----------------|-----------------|
| retrieval_practice | 979 +/- 5 | 363k | 175 +/- 11 |
| standard_ft | 961 +/- 6 | 363k | 214 +/- 24 |
| test_reinforce | 975 +/- 2 | 235k | 262 +/- 30 |
| test_only | 889 +/- 18 | 187k | 284 +/- 7 |
| scheduled_restudy | 505 +/- 21 | 364k | 626 +/- 15 |
| restudy_fixed_p10 | 994 +/- 2 | 363k | 41 +/- 4 |
| restudy_fixed_p25 | 997 +/- 2 | 363k | 29 +/- 4 |
| restudy_fixed_p50 | 999 +/- 1 | 363k | 5 +/- 2 |
| restudy_fixed_p75 | 1000 +/- 1 | 363k | 2 +/- 1 |
| random_replay | 993 +/- 3 | 409k | 38 +/- 6 |
| curriculum | 482 +/- 10 | 363k | 432 +/- 6 |
| loss_replay | 999 +/- 1 | 409k | 108 +/- 3 |

**retrieval_practice vs standard_ft.** Both consumed ~363k training tokens, so this is a fair comparison. `retrieval_practice` masters 979 items vs `standard_ft`'s 961. The gap looks modest at the end, but the learning curve tells a bigger story: at step 1000 (halfway through training), `retrieval_practice` has already mastered ~885 items while `standard_ft` is at ~753. It learns significantly faster. It also has fewer remastery events (175 vs 214), which means items that `retrieval_practice` masters tend to stay mastered rather than being forgotten and needing to be relearned.

**retrieval_practice vs scheduled_restudy.** This is the sharper and more informative comparison. Everything is matched: same scheduler, same items, same gradient steps, same token budget (~364k). The only thing that differs is how "correct" is determined. `retrieval_practice` uses generation-based exact-match, and it masters 979 items. `scheduled_restudy` uses loss-based scoring with an adaptive median, and it only masters 505 with 3.5x more remastery events (626 vs 175). The adaptive median converges to about 50% "correct" regardless of how well the model is actually doing, because the median tracks the loss distribution as it shifts during training. So even when the model has learned most items, about half still get marked "incorrect" and half get marked "correct" at every step. Items oscillate between mastered and unmastered constantly. This is evidence that loss is a poor proxy for whether the model actually knows a factual answer, which is consistent with KR-Test (2026) showing from a different angle that loss and perplexity are insufficient for detecting factual knowledge retention.

**restudy_fixed variants.** These report 994 to 1000 items mastered, which looks like they beat `retrieval_practice`'s 979. But the numbers are not directly comparable. Loss-based "correct" is strictly easier than exact-match. A model can assign high probability to the right tokens while its actual top prediction is a completely different string. For example, the model might put 40% probability on each token of "New York City" (giving low loss) while its greedy prediction is "The Big Apple" (at 45% per token). Loss says correct. Exact-match says wrong. The higher mastery counts for `restudy_fixed` reflect a more lenient correctness criterion, not better learning. A uniform end-of-training evaluation that uses exact-match generation across all methods is needed before these numbers can be meaningfully compared to `retrieval_practice`.

**random_replay and loss_replay.** Both show high mastery counts (993 and 999), but they consumed ~409k tokens and hit the 500k budget cap early, around step 1833. Meanwhile, `retrieval_practice` ran all 2,000 steps on ~364k tokens. These methods do more gradient-bearing work per step because they study a batch plus replay items, so they burn through the budget faster. Without normalizing for training tokens, the comparison is not fair.

**test_only and test_reinforce.** `test_only` masters 889 items on only 187k tokens, and `test_reinforce` masters 975 on 235k. These are the most token-efficient methods because they don't do gradient steps on all due items (only failures for `test_reinforce`, none for `test_only`). But the budget difference means they can't be directly compared to `retrieval_practice`'s 979 at 364k without normalization. The 25k step run will show whether `test_only` continues climbing or plateaus as old items start being forgotten.

**curriculum.** Only 482 items mastered with 432 remastery events. Training items in difficulty order without any adaptive review means early items get forgotten while the model is still working through harder items later. Consistent with the education research on massed vs spaced practice.

**Scheduler ablation** (all three use `retrieval_practice`, the only difference is which scheduler decides when items come due):

| Scheduler | Mastered | Remastery |
|-----------|----------|-----------|
| FSRS | 979 +/- 5 | 175 +/- 11 |
| Random matched | 948 +/- 5 | 33 +/- 6 |
| Random wide | 246 +/- 13 | 0 +/- 0 |

FSRS beats random matched scheduling by about 31 items (979 vs 948). But random matched, which completely ignores whether the model got an item right or wrong and just assigns random intervals between 50 and 500 steps, still achieves 948 out of 1000. That's a strong result for a scheduler with zero intelligence. It means the retrieval-practice protocol itself (generate an answer, check it, restudy the item) does most of the heavy lifting. The scheduling algorithm improves things, but it's secondary to the protocol.

`random_wide` only masters 246 items with zero remastery events, confirming that review frequency matters. When items barely come due, the protocol can't do its job. The zero remastery isn't because items are retained better; it's because items aren't being probed often enough to even detect forgetting.

The low remastery for `random_matched` (33 vs FSRS's 175) is also likely a detection artifact rather than a real retention difference. `random_matched` ran 7,646 total tests vs FSRS's 12,529. Fewer probes means fewer chances to catch a forgotten item.

## Budget accounting

The training token budget only counts operations where gradients flow, meaning study updates and reinforce updates. Test-phase generation (the `torch.no_grad()` generation calls used for scoring) is tracked separately as overhead and doesn't count toward the budget. This is the right policy because the generation calls don't change the model's weights.

The `--require-budget` flag makes it an error to run without `--max-training-tokens` set. This exists because replay methods naturally consume more training tokens per step than non-replay methods (they do study plus replay updates), so running without a budget cap produces comparisons where some methods got significantly more gradient updates than others. For real experiments, always set both `--max-training-tokens` and `--require-budget`.
