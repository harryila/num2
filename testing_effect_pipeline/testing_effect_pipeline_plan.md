# Testing Effect Pipeline Plan (Revised v2)

This document operationalizes research direction **#4 (Testing Effect / Interleaved Retrieval Practice)** from `human_cognition_to_llm_deep_dive.md` into an implementation and experiment plan.

## 1) Research Claim

Retrieval-gated adaptive replay improves long-horizon retention and reduces forgetting in LLM fine-tuning compared with standard fine-tuning, random replay, curriculum ordering, and loss-prioritized replay, at matched compute/token budgets.

## 2) Minimal Publishable Unit (MPU)

Demonstrate in one declarative-knowledge domain that, at equal training budget:

- forgetting curves are flatter,
- delayed retention is better,
- learning efficiency is higher (items mastered faster),
- and transfer is non-inferior or improved,

for at least one testing-effect variant versus all critical baselines.

## 3) Domain and Data Choice (Locked for first experiment)

Use **SQuAD v1.1 reformatted as closed-book QA** for the first full experiment cycle.

Why this dataset first:

- well-studied and easy to reproduce,
- answer strings are clean enough for binary exact-match scoring,
- items are naturally separable and can be given stable IDs,
- strong fit for declarative retrieval (the cognitive regime where testing effects are expected to be strongest).

### Dataset Lock Rule

- Lock dataset this week and do not switch domains during the first experiment pass.
- Optional follow-up replication can use a Natural Questions closed-book subset after the first result is established.

### Data Requirements

Each training item must have a stable, persistent ID:

- `item_id`
- `prompt`
- `target`
- optional `domain_tag`
- optional `difficulty`

## 4) Pipeline Architecture

### 4.1 Memory State Store (per item)

Track:

- `item_id`
- `last_study_step`
- `last_test_step`
- `success_streak`
- `failure_count`
- `estimated_half_life` (or interval)
- `next_due_step`
- `is_mastered`
- `mastered_at_step`
- rolling `test_accuracy`
- rolling `test_loss`

This state is the key infrastructure differentiator from generic replay.

### 4.2 Three Queues

- **Study queue**: new/current items
- **Test queue**: due items selected by scheduler
- **Reinforcement queue**: failed retrieval items (for the reinforce variant)

### 4.3 Dynamic Batch Composition (not fixed 60/20/20)

Queue fractions are derived from scheduler pressure:

- if many items are due, test share increases,
- if due queue is small, study share dominates,
- reinforcement share depends on recent failures.

Use caps/floors to avoid starvation (e.g., minimum study fraction).

## 5) Two Required Testing-Effect Variants

### Variant A — Test-only (diagnostic retrieval, no test gradient)

1. Model generates answer for due items.
2. Score correctness.
3. Update scheduler state only.
4. No gradient update in test phase.

Purpose: isolate whether retrieval-derived scheduling alone drives gains.

### Variant B — Test + reinforce

1. Model generates answer for due items.
2. Score correctness.
3. Update scheduler state.
4. Apply reinforcement updates on failed items.

Purpose: measure benefit of retrieval practice with feedback.

## 6) Scheduler Design

### MVP Scheduler

Leitner-style heuristic:

- success → interval expands,
- failure → interval contracts/reset.

### Stronger Scheduler Ablation

Use a half-life model (e.g., **FSRS-style scheduling**) as an ablation against heuristic spacing.

Why: gives a principled forgetting-curve-based scheduler and strengthens reviewer confidence that spacing is not ad hoc.

## 7) Baselines (Critical)

At minimum:

1. Standard fine-tuning (no replay)
2. Random replay
3. Difficulty curriculum
4. Loss-based replay (high-loss prioritization)

Optional continual-learning comparator(s): replay-based CL/EWC-style regularization if available.

## 8) Evaluation: Headline is Forgetting + Efficiency

### 8.1 Mastery Definition (required)

Define item mastery explicitly:

- An item is **mastered** when it is answered correctly on **K=2 consecutive test probes**.
- `is_mastered` flips to true at the second consecutive success.
- `mastered_at_step` is recorded at that transition.

This prevents mixing “never learned” items with “learned then forgotten” items.

### 8.2 Periodic Forgetting Probe (required)

Every N steps, evaluate only items that were mastered at an earlier step and measure retention at horizons:

- `T+5k`,
- `T+10k`,
- `T+20k`.

Report:

- forgetting curve by item age,
- retention slope,
- area-under-retention-curve,
- fraction retained at each horizon.

### 8.3 Learning Efficiency Curve (required)

Track speed to mastery:

- plot cumulative mastered items vs training steps,
- report time-to-X%-mastery,
- report mastery throughput (items mastered per 1k steps).

This supports a two-part story: faster learning + slower forgetting.

### 8.4 Additional Metrics

- immediate in-domain performance,
- transfer/OOD performance,
- calibration/error profile,
- overhead (wall-clock, memory, extra tokens).

## 9) Fairness Controls

All methods must be matched for:

- total optimizer steps,
- total train tokens,
- eval cadence.

Run >=3 seeds for headline claims.

## 10) Related-Work Positioning (for paper draft)

Frame against:

- curriculum learning (ordering by difficulty),
- replay/rehearsal methods,
- catastrophic forgetting / continual learning.

Core distinction:

> This method is **retrieval-gated adaptive replay**: explicit testing determines what is revisited and when, rather than static schedules or loss-only prioritization.

## 11) Key Falsifiers (pre-registered internally)

The central claim is weakened if:

- gains vanish under strict token-budget control,
- random or loss-based replay matches results,
- improvements appear only in near-term metrics but not delayed retention,
- test-only provides no signal advantage over non-test scheduling.

## 12) 4-Week Execution Plan

### Week 1

- lock dataset to closed-book SQuAD v1.1 and finalize exact-match scorer,
- build stable-ID data pipeline and memory state store,
- implement standard FT + random replay,
- implement mastery logic and forgetting probe evaluation from day 1.

### Week 2

- implement test-only and test+reinforce variants,
- add Leitner scheduler,
- add FSRS-style scheduler ablation,
- ship retention/forgetting + mastery-efficiency dashboards.

### Week 3

- run full matrix: baselines + variants + scheduler ablations,
- add curriculum + loss-replay,
- run >=3 seeds.

### Week 4

- analyze forgetting curves as the primary result,
- analyze cumulative mastery curves as secondary headline,
- validate robustness at matched budgets,
- draft writeup focused on retention, efficiency, and catastrophic-forgetting comparisons.

## 13) Decision Rule

Proceed to paper drafting if at least one testing-effect variant produces both:

1. statistically meaningful forgetting-curve improvement over all critical baselines at matched budget, and
2. non-inferior or better mastery-efficiency trajectory,

with acceptable overhead.
