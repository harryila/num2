# Pipeline folder assessment and recommendations
_Exported on 3/11/2026 at 14:47:51 PDT from Cursor (2.4.21)_

---

**User**

figure out what im trying to do hear, read the repo and then this is what claude said about the current pipeline folder:

Let me try cloning it directly.Viewed 13 files, ran a commandViewed 13 files, ran a commandLet me do a quick smoke test to make sure it actually runs.Ran 2 commandsRan 2 commandsThis is well-built. The architecture is clean, the abstractions are right, and it runs. Here's my assessment:
What's good:
The code quality is solid. Clean separation of concerns — types, scheduler, trainer, baselines, budget, model adapter, experiment runner — all in their own files with minimal coupling. The ModelAdapter abstraction is exactly right: swapping MockMemoryModel for a real HF adapter will touch one file.
Budget tracking works. Training tokens and test inference tokens are tracked separately, which is the correct policy. The over_budget() early-stop is implemented.
Remastery tracking is in (remastery_count on ItemState, events logged in metrics). Good — that was a suggestion from earlier and it's already there.
The smoke test runs and produces sensible output.
Three real issues I see:
1. The baselines have a budget fairness problem. Look at the smoke test output: random_replay and loss_replay used 72,000 training tokens while test_only and test_reinforce used 41,622. That's a 1.7x difference. The replay baselines do study (full batch) plus replay (reinforce updates), so they're getting more gradient steps per global step. When you set --max-training-tokens, the testing-effect methods will run longer (more steps) before hitting the cap, while replay baselines will hit it sooner. This is fine if you're comparing at matched token budgets — but right now the default run with no token cap gives replay methods a huge training advantage. You need to either: (a) always run with --max-training-tokens set, or (b) adjust the baseline batch sizes so total training tokens per step are equal across methods. I'd go with (a) and make it mandatory, not optional.
2. The ret_5k=0.00 across the board is expected but worth understanding. With only 500 steps, no item can have been mastered at step ≤ -4500, so the 5k-horizon retention is trivially 0. This is fine for a smoke test but means your real runs need to be long enough for retention horizons to kick in. At 20k steps with eval every 500, the 5k retention window only starts producing data at step 5000+. Make sure your real config has enough steps that the 20k horizon gets meaningful signal — so at least 25k-30k total steps.
3. The MockMemoryModel's _touch decay applies on every call including test. Line 37: every time you call study_update, reinforce_update, or test, the strength decays by 0.01 first. This means testing an item weakens it slightly, which is the opposite of the testing effect in humans (where retrieval strengthens memory). For the mock, this is arguably fine — you want a model that forgets, and the mock shouldn't build in the hypothesis you're trying to test. But be aware: if someone reads the mock code, they might object that the mock is biased against the testing-effect method (since testing costs strength). That's actually a good thing for your paper — if the method wins despite the mock penalizing retrieval, the result is more robust. Just note this in DESIGN_CHOICES.md.
One structural suggestion: add a --require-budget flag that makes the runner error out if --max-training-tokens isn't set. For real experiments, you should never run without a budget cap. Make it impossible to accidentally produce unfair comparisons.

---

**Cursor**

Let me read through the repo to understand the full picture.

Now I have a clear picture of the project. Let me read the specific files that matter most for understanding where things stand and what you need.

Okay, here's what you're doing and where things stand.

---

**The project:** You're testing whether the **testing effect** from human cognitive science (retrieval practice strengthens memory more than passive re-study) transfers to LLM fine-tuning. The pipeline implements a study-test-reinforce loop with adaptive spacing (Leitner/FSRS schedulers), compares two testing-effect variants (`test_only`, `test_reinforce`) against four baselines (`standard_ft`, `random_replay`, `curriculum`, `loss_replay`), and measures long-term retention at matched token budgets.

**Current state:** The pipeline is fully built in mock mode -- clean architecture, all methods run, metrics collect, budget tracking works. You're at the transition point from "validated mock pipeline" to "real LLM experiments." `WHAT_I_NEED_FOR_REAL_RUN.md` is your checklist for that jump: you need a real dataset, a real model backend, compute budget, and locked-down run matrix.

**Claude's feedback** identified three real issues and one structural suggestion. All of them are valid and actionable right now. Want me to implement the fixes? Specifically:

1. **Budget fairness fix** -- Make `--max-training-tokens` mandatory for real runs (add `--require-budget` flag that errors out if no budget is set). This ensures replay baselines and testing-effect methods are always compared at matched token spend.

2. **Retention horizon documentation** -- Add a note to `DESIGN_CHOICES.md` about minimum step requirements for retention windows to produce meaningful data (e.g., 25k-30k steps minimum for the 20k horizon).

3. **Mock model decay-on-test note** -- Document in `DESIGN_CHOICES.md` that the mock penalizes retrieval (opposite of the testing effect in humans), and why that's actually a conservative design choice that strengthens any positive result.

4. **The `--require-budget` flag** in `run_experiment.py`.

Should I go ahead and make all four of these changes?

---

**User**

yea make those changes

---

**Cursor**

Now add the validation logic in `main()` and `run()`:

Now update the example config to include `require_budget`:

Now add both new notes to `DESIGN_CHOICES.md`:

Now let me verify the flag works -- it should error without a budget, and pass with one:

Errors correctly. Now verify it passes when a budget is provided:

Let me check lints on the edited files:

All four changes are in. Here's the summary:

**`run_experiment.py`** -- Added `--require-budget` flag. When set, the runner errors out immediately if `--max-training-tokens` is not provided, preventing accidental unfair comparisons where replay baselines burn ~1.7x more training tokens per step than testing-effect methods.

**`real_run_config.example.json`** -- Added `"require_budget": true` so the example config for real runs has the guard on by default.

**`DESIGN_CHOICES.md`** -- Added three things:
- **Item 35** (mock model): Documents that test-phase decay penalizes retrieval (opposite of the human testing effect), and explains why this is a conservative design choice that strengthens any positive result.
- **Item 39** (runner): Documents the `--require-budget` flag and the budget fairness rationale.
- **New section 10** (retention horizons): Documents the minimum step requirements for each retention window (5k/10k/20k) to produce meaningful data, noting that real runs need 25k-30k steps for the 20k horizon.

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

Let me start by understanding the existing codebase structure.

Now let me read the key files I need to modify/extend.

