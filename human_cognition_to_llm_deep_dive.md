# Human Cognition → LLM Transfer: Deep Dive on 10 Research Directions

**Author: Harry (compiled with Claude)**  
**Date: March 2026**

---

## 1. Feeling of Knowing (Pre-Generation Confidence Probing)

### The Human Mechanism
Humans have a metacognitive "feeling of knowing" (FOK) — before attempting recall, we can sense whether we know the answer. This is distinct from confidence *after* answering. It's a fast, pre-retrieval signal that routes cognition: "I know this, let me think harder" vs. "I don't know this, let me look it up." First studied by Hart (1965), it's now well-established in metacognition research.

### What Exists Already
**This space is getting crowded fast — but there's still room.**

- **"No Answer Needed" (Marcinkevics et al., 2025)** — The most directly relevant paper. Published on OpenReview, under review. They train a single-dimensional linear probe on residual stream activations *after the question is read but before any tokens are generated*. Tested across 3 model families (7B-70B). Key findings:
  - A simple difference-of-means direction in mid-to-late layers reliably predicts correctness
  - Transfers across datasets (train on TriviaQA, test on others)
  - **Fails on mathematical reasoning** — the signal is strongest for factual recall
  - Correlates with models' "I don't know" responses
  
- **Kadavath et al. (2022, Anthropic)** — "Language Models (Mostly) Know What They Know." Tested similar probes on proprietary models. No released code.

- **CoCA (2025, last week!)** — "Confidence Before Answering." Goes further: jointly trains confidence generation and answer generation end-to-end with segment-specific GRPO rewards. Trains on Qwen2.5 models. This is the state-of-the-art.

- **Burns et al. (2022), Azaria & Mitchell (2023), Marks & Tegmark (2023)** — Foundational work on truth probes from hidden states, but focused on *post-hoc* truthfulness of complete statements, not pre-generation prediction.

- **Kudo et al. (2024)** — Studied when during CoT the model "commits" to its eventual answer internally.

- **Ferrando et al. (2025)** — Used Sparse Autoencoders to identify latent features distinguishing correct from incorrect answers.

### Where the Gap Is
The cognitive science framing is underexploited. Existing work treats this as a calibration/uncertainty problem. Nobody has:

1. **Modeled the FOK as a routing mechanism** — use the probe score to dynamically decide: generate normally, invoke retrieval (RAG), say "I don't know," or allocate more compute (longer CoT). The probe becomes an actual cognitive router, not just a post-hoc evaluation.

2. **Studied domain-specific FOK profiles** — the "No Answer Needed" paper shows failure on math. This mirrors human metacognition: FOK is strong for semantic memory (facts) but weak for procedural tasks. Characterizing *when* the signal exists vs. doesn't would be a contribution.

3. **Connected FOK to hallucination prevention** — use FOK as a gating mechanism to prevent generation when the model "knows it doesn't know." This is different from existing calibration work which is post-hoc.

### Concrete Research Direction
Train lightweight FOK probes across model families and task types. Build a taxonomy of when LLMs have accurate self-knowledge (factual QA) vs. when they don't (math, multi-hop reasoning). Then implement FOK-gated generation: if probe score < threshold, route to RAG or abstention instead of generating.

### Honest Assessment
**The core probing result is already published.** The novelty has to come from (a) the routing/gating application, (b) the cognitive science framing connecting to human FOK literature, or (c) the domain-specific failure analysis. A pure replication won't fly. But a paper that says "here's how to use the FOK signal to actually reduce hallucinations in production" could be strong.

---

## 2. Chunking Constraints as Beneficial Compression

### The Human Mechanism
Miller's (1956) "magical number seven, plus or minus two" isn't just a limitation — it's a feature. The working memory bottleneck forces humans to *chunk* information into meaningful units, which drives hierarchical abstraction. A chess grandmaster doesn't remember individual pieces; they remember board patterns as single chunks. The constraint itself is what makes expertise possible.

### What Exists Already
- **IBRO — "Revisiting LLM Reasoning via Information Bottleneck" (Lei et al., July 2025)** — Directly applies information bottleneck theory to LLM reasoning. Adds IB regularization to RL post-training that encourages CoT trajectories to be informative about the answer while generalizable. Token-level surrogate objective. Works with PPO and DAPO on math benchmarks.

- **"The Information Bottleneck of Chain-of-Thought and How Latent CoT Overcomes It" (2025, OpenReview)** — Theoretically establishes that CoT has a fundamental bottleneck: each forward pass activates many neurons but can only write one token. Proves that for problems like pointer chasing and parity, text CoT requires much longer chains than latent CoT.

- **Coconut / Latent Reasoning (Hao et al., 2024)** — Enables reasoning in continuous latent space rather than discrete tokens. The model reasons over hidden states, bypassing the text bottleneck.

- **Pause Tokens (Goyal et al., 2023)** — Learnable delay tokens that give the model extra compute before answering.

- **CoT Compression via Step Entropy (2025)** — Prunes reasoning steps based on entropy, arguing this "mimics human cognition (skipping entire thoughts, not words)." Shows you can remove 40-60% of CoT steps without accuracy loss.

- **Interlat (2025)** — Inter-agent communication in latent space rather than text, motivated by the observation that text is a lossy, low-bandwidth medium (~15 bits/token vs ~40k bits/hidden state).

### Where the Gap Is
Everyone is treating the bottleneck as a *problem* to overcome (make CoT shorter, move to latent space, etc.). Nobody has tested whether **intentionally imposing a tighter bottleneck at specific points** actually *helps* — which is the cognitive science prediction.

The human insight is: the bottleneck doesn't just compress, it forces **hierarchical re-representation**. When you can only hold 5-7 chunks, you're forced to abstract. Current approaches either remove the bottleneck (latent CoT) or compress within it (step pruning). Nobody forces the model through a narrow bottleneck and then *continues reasoning from the compressed representation*.

### Concrete Research Direction
Insert a "chunking layer" into multi-step reasoning:
1. Model reasons for N steps
2. Force it to compress its reasoning into K << N abstract "chunk" tokens (via a learned compression head)
3. Continue reasoning from only those K chunks

Test whether this forced abstraction improves generalization on harder variants of the same problem type. The prediction: it should help on tasks requiring hierarchical structure (multi-step math, planning) even if it hurts on tasks requiring detailed recall.

### Honest Assessment
IBRO is very close to this. The differentiation has to be: IBRO adds IB regularization to the *training objective*, while chunking would impose an architectural bottleneck at *inference time* that forces re-representation. The cognitive science framing (Miller's chunks, expertise research) is the differentiator. Medium risk that reviewers say "this is just prompt compression."

---

## 3. Predictive Processing at Inference (Surprise-Gated Compute)

### The Human Mechanism
The brain doesn't passively process all input equally. Under predictive processing / the free energy principle (Friston, 2005+), the brain generates predictions about incoming input and only deeply processes *prediction errors* (surprises). This is computationally efficient — you only allocate resources where your model is wrong. Predictive coding is now one of the dominant theories of brain function.

### What Exists Already
- **Active Inference in RL** — Several papers (FEPS, 2025) implement active inference agents using the free energy principle for RL environments. These are typically small-scale, grid-world settings.

- **Adaptive compute / early exit** — Work on routing tokens through different numbers of transformer layers based on difficulty. MoE architectures partially capture this.

- **Speculative decoding** — Uses a small model to draft tokens and a large model to verify. This is structurally similar to predictive processing (predict → check → correct) but isn't framed that way.

- **Differential Transformer (Oct 2024)** — Computes attention as difference of two softmax maps, which amplifies unexpected/novel patterns and suppresses expected ones. This is functionally similar to a prediction error signal.

### Where the Gap Is
Nobody has built an LLM inference pipeline that explicitly:
1. **Predicts** what the next tokens/chunks should be (using a fast, cheap process)
2. **Measures surprise** when the actual generation differs from the prediction
3. **Allocates more compute** (deeper processing, more samples, longer CoT) proportional to surprise

This is different from early exit (which routes based on token difficulty) because it's about the *discrepancy between prediction and generation*, not just generation difficulty.

### Concrete Research Direction
Implement a dual-process inference pipeline:
- **Process 1 (fast)**: Shallow forward pass or small model generates a "prediction" of what the answer should roughly contain
- **Process 2 (slow)**: Full model generates, but compute budget is modulated by how much the generation diverges from the prediction
- Measure: does surprise-gated compute improve accuracy/token efficiency compared to uniform compute or simple adaptive methods?

### Honest Assessment
This is theoretically beautiful but implementation is tricky. The main challenge: how do you define "prediction" and "surprise" in a way that's computable and meaningful? Token-level surprisal (perplexity) is too granular. Chunk-level semantic similarity might work but adds latency. Risk that the overhead of the prediction step exceeds the savings from adaptive compute. Probably needs to be demonstrated on a narrow domain first.

---

## 4. Testing Effect / Interleaved Retrieval Practice for LLM Training

### The Human Mechanism
The testing effect (Roediger & Karpicke, 2006) is one of the most robust findings in learning science: actively retrieving information from memory strengthens it far more than passive re-exposure. Spaced repetition + retrieval practice is dramatically more efficient than massed study. Interleaving different problem types during practice (rather than blocking) improves discrimination and transfer.

### What Exists Already
- **Curriculum Learning for LLMs** — Extensive work on ordering training data easy→hard (CAMPUS, TAPIR, Data-CUBE, etc.). This is about *sequencing*, not *testing*.

- **LECTOR (Aug 2025)** — Uses LLMs to enhance spaced repetition for *human* learners, not for training LLMs themselves.

- **Self-play / self-improvement** — RISE, AlphaLLM, and others have LLMs generate training data and improve from it. This captures some aspects of retrieval practice but isn't framed around the testing effect.

- **SLEEP paradigm** — Memory consolidation for LLMs (see separate paper). Includes a "dreaming" phase where the model generates and trains on its own outputs. Closest to the testing effect, but focused on consolidation, not retrieval-strengthened encoding.

### Where the Gap Is
Nobody has implemented the testing effect *as a training protocol* for LLMs:
- Train on batch → quiz the model on that batch → reinforce based on retrieval success/failure → space the re-presentation of failed items

This is fundamentally different from curriculum learning (which is about ordering) or self-play (which is about generating new data). It's about *interleaving encoding and retrieval of the same material* with adaptive spacing.

### Concrete Research Direction
Modify fine-tuning to alternate between:
1. **Study phase**: Standard training on a batch
2. **Test phase**: Present items from previous batches, measure recall (generate completions, check accuracy)
3. **Reinforcement phase**: Items the model got wrong get scheduled for re-presentation with expanding intervals; items it got right get longer delays

Compare against standard fine-tuning, curriculum learning, and random replay on knowledge retention benchmarks.

### Honest Assessment
The "is this just curriculum learning with extra steps?" objection is the main risk. The counter: curriculum learning sequences by difficulty, this sequences by *the model's own retrieval success*. It's personalized, adaptive, and grounded in a specific cognitive mechanism. The implementation is very tractable — it's just a modified training loop. But the benchmarking needs to carefully distinguish this from existing replay/rehearsal methods.

---

## 5. Salience Filtering (Dynamic Attention Routing)

### The Human Mechanism
Human attention is selective: we don't process all sensory input with equal depth. Salient stimuli (novel, unexpected, emotionally charged, goal-relevant) get deep processing; expected/irrelevant stimuli get filtered. This is implemented via top-down (goal-directed) and bottom-up (stimulus-driven) attention systems.

### What Exists Already
- **Sparse Attention** — Longformer, BigBird, etc. reduce attention to local + global patterns. This is structurally similar but not content-adaptive.

- **Mixture of Experts (MoE)** — Routes tokens to different expert networks. This is about *processing* routing, not *attention* filtering.

- **LLMLingua / prompt compression** — Compresses prompts by removing low-information tokens. This is a form of salience filtering at the input level.

- **Differential Transformer** — Amplifies unexpected patterns via attention difference maps. Closest to bottom-up salience.

- **IC-ICL (Jan 2025)** — Uses Information Bottleneck theory to retrieve and compress relevant examples into a task space, improving reasoning accuracy and speeding inference by 40%.

### Where the Gap Is
No system has a learned, dynamic salience gate that operates during generation (not just at input) to route different tokens/chunks into deep vs. shallow processing pathways based on content-dependent relevance. Current approaches are either static (sparse attention patterns) or input-level (prompt compression).

### Concrete Research Direction
Add a lightweight "salience scorer" that runs at each layer (or every K layers) and scores each token's relevance to the current generation goal. Low-salience tokens skip subsequent layers (early exit at the token level). High-salience tokens get full processing. The scorer is trained to predict which tokens most affect output quality.

### Honest Assessment
This overlaps significantly with token-level early exit research and adaptive compute. The differentiation needs to come from the *bidirectional* nature of salience (both stimulus-driven and goal-directed) and from operating during generation, not just encoding. Medium novelty, high practical impact. Might be better as a systems paper than a cognitive science paper.

---

## 6. Emotional Valence as Priority Signal

### The Human Mechanism
Emotionally significant information is processed more deeply, remembered better, and triggers faster decision-making. This isn't about "feeling emotions" — it's about a *valence signal* that modulates processing priority. The amygdala rapidly evaluates emotional significance and gates cortical processing depth.

### What Exists Already
- **Reward models in RLHF** — These assign scalar values to outputs, which is structurally similar to a valence signal, but they operate post-generation, not during processing.

- **Importance weighting in training** — Some curricula weight high-quality or high-impact examples more heavily. Static, not dynamic.

- **Episodic memory with emotional valence (EALLM architecture, 2025)** — Stores episodic traces with temporal markers and emotional valence, with dynamic compression. This is the closest: it uses valence as a *memory* priority signal.

### Where the Gap Is
Nobody has used a learned importance/urgency signal that modulates *processing depth during inference*. The idea: some tokens/positions in a sequence are more consequential than others. A fast "importance" circuit could flag these, triggering deeper processing (more attention heads, longer CoT, etc.) for high-stakes segments.

### Concrete Research Direction
Train an auxiliary "valence head" that scores each position in the input for downstream importance (how much does this token affect the final answer?). Use this to modulate: (a) attention weights, (b) compute allocation, (c) memory retention in context. Evaluate on tasks where missing a key detail is catastrophic (e.g., legal documents, medical records, code where one line has the bug).

### Honest Assessment
Risky. Hard to define "importance" without circularity (you need the answer to know what's important). May end up being "attention weights with extra steps." The cognitive science framing is interesting but the engineering contribution needs to be clear. Would need a very specific, well-chosen benchmark.

---

## 7. Cognitive Offloading (Proactive Externalization)

### The Human Mechanism
Humans are powerful reasoners partly because we routinely externalize cognition — writing notes, drawing diagrams, using tools. The key insight from Clark & Chalmers (1998, "The Extended Mind") is that the *act of externalizing* restructures the problem. Writing something down isn't just memory aid; it changes what you can see and think.

### What Exists Already
- **Tool use** — Extensive work on LLMs calling calculators, code interpreters, search engines.

- **Scratchpads** — Models that write intermediate reasoning to a scratchpad before answering.

- **ReAct (Yao et al., 2022)** — Interleaves reasoning traces and actions. Close to cognitive offloading but the model doesn't *choose* to externalize; the protocol mandates it.

- **Voyager** — Open-ended Minecraft agent that writes skill programs and stores them in a library. Good example of cognitive offloading in practice.

### Where the Gap Is
Current tool use is *reactive* (use a tool when prompted) not *proactive* (recognize when externalizing would help). Nobody has trained a model to learn *when* to write things down vs. keep reasoning internally. The decision itself — "this is too complex to hold in my head, I need to externalize" — is the interesting cognitive skill.

### Concrete Research Direction
Train a model with access to a scratchpad/notebook but give it the choice of whether to use it. Reward accurate final answers. See if the model learns to offload when working memory (context) gets overloaded — i.e., does it learn to recognize its own cognitive limits and compensate?

### Honest Assessment
Partially covered by existing tool-use and scratchpad work. The novelty is narrow: it's about the *meta-decision* to externalize, not the externalization itself. Could be a nice short paper or workshop paper but probably not a top-venue main track paper on its own.

---

## 8. Analogical Structure Mapping

### The Human Mechanism
Analogical reasoning — seeing that "an atom is like a solar system" — is considered central to human intelligence (Hofstadter, Gentner). Gentner's Structure-Mapping Theory (1983) formalizes this: analogy maps *relational structure* (not surface features) from a known source to a novel target domain. This enables far transfer: learning about one domain helps with a structurally similar but superficially different domain.

### What Exists Already
- **"Analogical Prompting" (Yasunaga et al., ICLR 2024)** — Prompts LLMs to self-generate relevant past problems before solving new ones. Inspired by Polya's "Do I know a related problem?" Achieved strong results on math and code generation. Not actually doing structure mapping — it's retrieving similar examples.

- **"Semantic Structure-Mapping in LLM and Human Analogical Reasoning" (Musker et al., 2024-2025)** — Directly tests Gentner's structure mapping in LLMs. Finds that advanced LLMs match humans on many tasks involving semantic structure transfer, but fail differently on certain variations and semantic distractors. Key finding: LLMs may have emergent structure-mapping ability but it's fragile and not fully human-like.

- **Lewis & Mitchell (2024)** — Uses counterfactual tasks to test generality of analogical reasoning in LLMs. Finds performance degrades substantially on stimuli unlikely to appear in training data, while humans are stable.

- **Webb et al. (2022)** — "Emergent Analogical Reasoning in Large Language Models." Found GPT-3 performed comparably to humans on certain analogy tasks, sparking the debate about whether this is genuine reasoning or pattern matching.

### Where the Gap Is
The question isn't "can LLMs do analogical reasoning?" (partially yes) but "can we *train* LLMs to do it better, and does this improve generalization?" Nobody has:

1. Used structure-mapping as a training signal (not just a benchmark)
2. Built an explicit structure-mapping module that extracts relational graphs from source and target, aligns them, and transfers inferences
3. Tested whether structure-mapping training improves far transfer to genuinely novel domains

### Concrete Research Direction
Fine-tune models on explicit structure-mapping tasks: given a source domain with known relational structure, map those relations onto a target domain and generate novel inferences. Evaluate on: (a) far transfer benchmarks, (b) counterfactual/novel domain problems where surface similarity is zero. The key metric: does analogical training improve performance on domains *not seen during training*?

### Honest Assessment
The Musker et al. paper is quite thorough on the evaluation side. The gap is on the *training* side — can you make models better at this? The evaluation problem is brutal: how do you measure "genuine analogical reasoning" vs. "sophisticated pattern matching"? Lewis & Mitchell's counterfactual approach is the best method so far. Hard to do as a single-person project because you need both the training infrastructure and novel benchmarks.

---

## 9. Incubation Effect (Context-Switch and Return)

### The Human Mechanism
When stuck on a problem, stepping away often helps. The incubation effect (Wallas, 1926; empirically validated by Sio & Ormerod, 2009) shows that periods of non-focused activity can lead to insight. The default mode network continues processing the problem unconsciously, making novel connections. Returning with "fresh eyes" often breaks the impasse.

### What Exists Already
- **Retry/refinement loops** — Models that attempt a problem, fail, and retry. Not incubation because there's no "stepping away."

- **RISE (NeurIPS 2024)** — Multi-turn self-improvement where the model iteratively refines answers. Close, but each turn is immediate — no delay or context-switching.

- **SLEEP paradigm** — Offline consolidation between active processing periods. The "dreaming" phase is structurally analogous to incubation, but it's about memory consolidation, not problem-solving insight.

### Where the Gap Is
Nobody has tested whether *context-switching* during problem-solving (work on problem A, switch to problem B, return to problem A) improves performance compared to continuous work on problem A. In humans, the benefit comes from:
1. Breaking fixation on incorrect approaches
2. Spreading activation to remote associations
3. Memory consolidation of partial solutions

### Concrete Research Direction
Design a multi-problem inference protocol:
1. Model attempts problem A, reaches an impasse (confidence drops / answers are unstable across samples)
2. Model compresses its progress on A into a summary
3. Model works on unrelated problem B
4. Model returns to A with the compressed summary + fresh context
Compare against: (a) continuous retry on A, (b) same total compute on A without context-switching.

### Honest Assessment
The experimental setup is awkward. How do you define "impasse"? How do you ensure the improvement (if any) comes from incubation and not just "more samples"? You'd need to control for total compute very carefully. The concept is fascinating but the execution is hard. Might work better as a section within a larger "cognitive inspiration" paper than as a standalone.

---

## 10. Social Learning with Theory of Mind

### The Human Mechanism
Humans learn powerfully by modeling other agents' beliefs, intentions, and knowledge states — then learning from their behavior in light of those inferred mental states. This isn't just imitation; it's imitation + understanding *why* the demonstrator did what they did. Theory of Mind (ToM) enables social learning that far exceeds what's possible from observation alone.

### What Exists Already
- **Multi-agent LLM systems** — AutoGen, CrewAI, etc. Agents communicate and collaborate, but don't model each other's beliefs.

- **ToM in LLMs** — GPT-4 shows some ToM capability (Kosinski, 2024; Strachan et al., 2024), performing at or above human level on certain ToM tasks. But this is evaluated as a capability, not used as a mechanism.

- **Generative Agents (Park et al., 2023)** — 25 agents with memory/reflection/planning. Show emergent social behaviors. But agents don't explicitly model each other's mental states; they observe behavior and react.

- **Debate / discussion frameworks** — Models argue and improve through multi-agent discussion. This captures some social learning but without explicit belief modeling.

### Where the Gap Is
Nobody has built a system where:
1. Agent A observes Agent B's behavior on a task
2. Agent A infers what Agent B *knows* and *believes* (including its errors)
3. Agent A uses this ToM model to decide what to learn from B and what to discard

This would enable selective social learning: learn from experts, filter out novice mistakes, and integrate insights from agents with complementary knowledge.

### Concrete Research Direction
Build a multi-agent system where:
- Agents have different training data / expertise levels
- Each agent maintains a "model" of other agents' competence profiles
- Learning is mediated through this model: accept demonstrations from agents believed to be competent on this task type, reject or downweight others
- Compare against naive imitation (copy everything) and no social learning (learn alone)

### Honest Assessment
This is the most ambitious and complex direction. Multi-agent training is expensive, hard to debug, and has many confounders. The ToM modeling adds another moving part. Hard to isolate what's working. Long-term project, probably needs a team. Not a quick paper. But if it works, it's a major contribution to both LLM research and cognitive science.

---

## Summary Table

| # | Direction | Novelty | Prior Work Density | Tractability | Impact | Best Venue |
|---|-----------|---------|-------------------|-------------|--------|------------|
| 1 | Feeling of Knowing | Medium (core result published) | High | Very High | High | EMNLP/ACL |
| 2 | Chunking Constraints | High | Medium (IB work exists but different angle) | Medium | High | NeurIPS/ICML |
| 3 | Predictive Processing | Very High | Low (mostly theoretical) | Low-Medium | Very High | ICLR |
| 4 | Testing Effect | High | Low (for LLM training) | High | Medium-High | NeurIPS |
| 5 | Salience Filtering | Medium | High (overlaps with adaptive compute) | Medium | High | Systems venues |
| 6 | Emotional Valence | Medium-High | Low | Medium | Medium | Workshop |
| 7 | Cognitive Offloading | Low-Medium | High (tool use) | High | Medium | Workshop |
| 8 | Analogical Structure Mapping | Medium (eval done, training gap) | Medium-High | Low | Very High | CogSci + AI venue |
| 9 | Incubation Effect | Very High | Very Low | Low | Unknown | Workshop/short paper |
| 10 | Social Learning + ToM | High | Medium | Very Low | Very High | Long-term project |

---

## My Recommendations

**For a fast workshop paper (camera-ready in weeks):**
- **#1 (FOK)** — Replicate the probe, add the routing/gating application, frame through cognitive science. The "No Answer Needed" paper leaves the application layer open.

**For a strong main conference paper (3-6 months):**
- **#4 (Testing Effect)** — Most tractable of the genuinely novel directions. Clear experimental setup, strong cognitive science grounding, differentiated from curriculum learning.
- **#2 (Chunking)** — If you can show that forced compression *helps* (counterintuitive result), this is a striking finding.

**For a high-ceiling longer project:**
- **#3 (Predictive Processing)** — If someone cracks this cleanly, it's a big deal. The Friston connection gives it theoretical gravitas. But execution risk is high.

**Skip for now:**
- **#7 (Cognitive Offloading)** — Too close to existing tool-use work.
- **#9 (Incubation)** — Fascinating but nearly impossible to evaluate cleanly.
