# What I Need From You To Run Real Experiments

This pipeline is configured and runnable now in mock mode. To switch to real runs, I need the following inputs/resources.

## 1) Dataset artifact (required)

Provide a closed-book JSONL dataset at a path like:

- `data/closed_book_squad.jsonl`

Per-row fields:

- `item_id` (stable unique ID)
- `prompt`
- `target`
- optional `domain_tag`
- optional `difficulty`

If `difficulty` is omitted, curriculum difficulty will be calibrated from first-exposure model loss.

## 2) Real model backend choice (required)

Pick one integration target for `ModelAdapter`:

- Hugging Face Transformers (recommended first)
- vLLM + external train loop
- custom in-house trainer

I then need:

- base model name/path,
- precision (fp16/bf16),
- max sequence length,
- generation settings for test probes (max new tokens, decoding policy).

## 3) Compute budget (required)

Provide hard constraints:

- GPU count/type,
- wall-clock cap,
- training-token cap.

Current accounting policy is already configured:

- training budget = study + reinforce tokens,
- test-phase inference tokens tracked separately as overhead.

## 4) Run matrix lock (required)

Confirm this first-pass matrix:

- Methods: `test_only`, `test_reinforce`, `standard_ft`, `random_replay`, `curriculum`, `loss_replay`
- Seeds: 3
- Scheduler: FSRS for main, Leitner as ablation
- Steps/eval cadence: from `real_run_config.example.json`

## 5) Acceptance criteria (required)

Confirm success criteria for proceeding:

1. Better forgetting curve than all critical baselines at matched training-token budget.
2. Non-inferior or improved mastery-efficiency trajectory.
3. Acceptable inference-overhead increase.

## 6) Optional but useful

- Preferred plotting format (notebook vs script)
- Confidence interval policy (bootstrap vs t-interval)
- Any target report template
