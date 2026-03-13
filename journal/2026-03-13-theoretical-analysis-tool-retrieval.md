# 2026-03-13 - Theoretical Analysis of SSM vs Transformer for Weak Tool Retrieval

## Context

Experiment 13 (`13_zipf_demo`) compares Transformer and Mamba2-Bonsai on a
synthetic tool-retrieval task built from layered first-order logic. The central
knob is `alpha`, which controls a Zipf distribution over demonstration quality:
low alpha yields many irrelevant tools (weak retriever), high alpha yields mostly
relevant tools (strong retriever). The key empirical finding is that Mamba2
excels when the retriever is weak (low alpha) and the eval context contains many
more tools than seen during training, while the Transformer is more competitive
when the retriever is strong.

This entry outlines directions for a theoretical analysis that explains and
formalizes this phenomenon.

## Task Summary (Experiment 13 Specifics)

The `depth3_fresh_icl` split uses:
- 3 layers, predicates per layer: (16, 256, 16), arity 0 (no variable binding)
- 256 rules per transition, distance 2 (two-step problems)
- Fresh layer-0 predicates per eval example (forces in-context learning)
- N demonstrations prepended to the prompt, sampled via Zipf(alpha) over 4 ranks
- Rank 1: applicable and goal-reachable (correct); Rank 2: applicable only;
  Rank 3: reachable only; Rank 4: neither

The arity-0 setting is important: there is no unification or variable binding.
Rule matching reduces to set membership (LHS atoms ⊆ current facts). This makes
the task significantly more tractable for theoretical analysis than the general
FOL case.

The model's job: given the query (facts + goal) and N tool demonstrations,
identify and output the correct rule. At low alpha, most demonstrations are
rank-3 or rank-4 distractors.

## Central Theoretical Question

**Why do SSMs handle long, noisy tool contexts better than Transformers, and why
does this advantage grow with context length beyond the training distribution?**

## Approach 1: Reduction to Noisy Associative Recall

### Idea

Strip the FOL structure and formalize the task as a pure retrieval problem:
- Context: N "tool slots," one containing the correct tool, N-1 containing
  distractors
- Query: a feature vector (or discrete key) that matches the correct tool
- Task: output the value associated with the matching tool
- Alpha controls the signal-to-noise ratio: the fraction of relevant vs
  irrelevant slots

This is a structured variant of associative recall (Olsson et al., 2022) with a
controlled noise parameter.

### Why This Is the Right Abstraction

In the arity-0 setting, rule matching is set containment (LHS ⊆ facts), which
is a fixed-complexity operation per rule. The dominant difficulty is not the
matching computation itself but *finding which demo to match* among N candidates.
The FOL structure is incidental to the retrieval bottleneck. This means results
about associative recall transfer directly.

### Analysis Sketch for Transformers

For a single-layer attention head performing retrieval, the probability of
attending to the correct demo is:

```
P(correct) = softmax(q · k_correct) / Σ_i softmax(q · k_i)
           = exp(s*) / (exp(s*) + Σ_{j ∈ distractors} exp(s_j))
```

where s* is the query-key dot product for the correct tool, and s_j for
distractors. Under standard distributional assumptions (e.g., keys are random
unit vectors in R^d):
- E[s_j] = 0, Var[s_j] = 1/d for distractors (random dot products)
- s* > 0 for the correct match

The sum of distractor exponentials concentrates around (N-1) · E[exp(s_j)].
By standard results on sums of log-normals / sub-exponential random variables:

```
P(correct) ≈ exp(s*) / (exp(s*) + (N-1) · exp(μ + σ²/2))
```

This decays as ~1/N when s* is not sufficiently large relative to the distractor
score distribution. For fixed model capacity, s* is bounded, so retrieval
accuracy necessarily degrades with N. The rate of degradation depends on the
embedding dimension d (which controls the concentration of distractor scores)
and the "margin" between correct and distractor scores.

**Key prediction**: Transformer retrieval accuracy decays polynomially in N for
fixed embedding dimension.

### Analysis Sketch for Selective SSMs

Mamba2's selective scan has input-dependent gating:

```
h_t = A · h_{t-1} + B(x_t) · x_t
y_t = C(x_t) · h_t
```

where B(x_t) and C(x_t) are learned functions of the input. The critical
observation: if the model learns a binary gating function g(x_t) such that
B(x_t) ≈ 0 for distractor demos and B(x_t) ≈ B* for relevant demos, then the
state h only accumulates information from the ~k relevant demos out of N total.

The effective context seen by the state is O(k), not O(N), where k is the number
of relevant demos. Since k depends on alpha but not on N (for fixed alpha, the
expected number of rank-1 demos is ~N/Z(alpha) · 1, but the *information needed*
is just one correct rule), the SSM can maintain constant retrieval accuracy as
N grows, provided:
1. The gating function g is learnable (depends on the expressivity of the
   B-projection network)
2. d_state is sufficient to store the relevant information: d_state > O(log N)
   to maintain the "address" of the correct tool

**Key prediction**: SSM retrieval accuracy is approximately constant in N once
d_state exceeds a threshold that scales as log(N) or slower.

### Crossover Analysis

The SSM advantage over the Transformer should emerge at some critical N* where
the Transformer's attention dilution overcomes its advantage in exact retrieval.
N* depends on:
- Embedding dimension d (higher d → Transformer concentrates better → larger N*)
- d_state for the SSM (higher → SSM can handle more complex scenes)
- Alpha (lower alpha → more distractors → smaller N*)

This gives a testable 2D phase diagram: (N, alpha) → {Transformer wins, SSM
wins}, with a boundary curve that should be empirically verifiable against
experiment 13 results.

## Approach 2: Information-Theoretic Channel Analysis

### Setup

Frame the task as communication through a noisy channel:
- Source: the identity of the correct rule r* (one of R possible rules)
- Encoding: a context sequence of N demos, one of which instantiates r*
- Channel noise: the N-1 distractors, sampled from Zipf(alpha)
- Decoder: the model

### Relevant Quantities

**Mutual information per demo**: For a single demo d sampled from the Zipf
distribution:
```
I(r*; d | alpha) = H(r*) - H(r* | d)
```
At alpha=0 (uniform over ranks), most demos are rank 4 (uninformative), so
I(r*; d) is small. At alpha→∞, demos are almost always rank 1, so I(r*; d) is
close to H(r*) = log(R).

**Total context information**: With N i.i.d. demos:
```
I(r*; d_1, ..., d_N | alpha) ≤ N · I(r*; d | alpha)  [independence bound]
```
But the actual mutual information saturates at H(r*) = log(R) regardless of N.
The question is how quickly it approaches saturation.

### Architecture as Decoder

Each architecture implements a different decoder for this channel:
- **Transformer**: can access all N demos in parallel via attention, but softmax
  normalization acts as a bottleneck when many demos are uninformative
- **SSM**: processes demos sequentially, accumulating sufficient statistics in
  its state; limited by state capacity but immune to the "denominator explosion"
  of softmax

**Rate-distortion perspective**: The SSM state has capacity C ≈ d_state ×
d_model bits. The minimum description length of r* is log(R) bits. If C >>
log(R), the SSM can extract r* even from very noisy contexts, because it has
ample capacity to store the accumulated evidence. The Transformer has no
analogous capacity bottleneck but suffers from an *extraction* bottleneck (soft
attention over the full context).

### Testable Prediction

Plot the empirical "information extraction rate" = (rollout_success_rate) /
(theoretical max given N, alpha) for each architecture. The SSM should show a
flatter curve as N increases, while the Transformer curve should decay.

## Approach 3: Attention Entropy and Concentration

### Direct Mechanistic Analysis

For trained Transformer models from experiment 13, measure:
1. **Attention entropy** H(attn_i) for each head i at the token positions
   corresponding to the goal/query, as a function of N (eval demo count)
2. **Attention mass on correct demo**: the total attention weight assigned to
   tokens belonging to the rank-1 demo
3. **Attention score gap**: max score on correct demo minus max score on any
   distractor

### Expected Behavior

As N increases with fixed alpha:
- H(attn) increases (attention becomes more diffuse)
- Mass on correct demo decreases roughly as 1/N
- Score gap remains approximately constant (model learned a fixed scoring
  function), but the softmax denominator grows

For the length generalization regime (eval N > train N):
- RoPE positional encodings degrade for positions beyond training range
- This compounds the attention dilution effect
- SSMs, which use no positional encoding, are immune to this failure mode

### Connection to Existing Theory

Hahn (2020) showed that soft attention cannot solve certain star-free languages
that require sharp position-dependent retrieval. The tool retrieval task under
low alpha has a similar character: the model must attend sharply to a specific
position (the correct demo) among many distractors, and soft attention provably
struggles as the number of distractors grows.

## Approach 4: State-Space Compression and Selective Gating

### Formalizing the Gating Advantage

Consider a simplified model where each context token x_t is either "signal"
(probability p ∝ 1/Z(alpha)) or "noise" (probability 1-p). The SSM update is:

```
h_t = (1 - g_t) · h_{t-1} + g_t · x_t
```

where g_t = σ(w · x_t) is a learned gate. If the model learns w such that
g_t ≈ 1 for signal tokens and g_t ≈ 0 for noise tokens, then after processing
N tokens:

```
h_N ≈ Σ_{t: signal} x_t / (# signal tokens)
```

This is an *unweighted average of signal tokens only*. The quality of h_N
depends on the number of signal tokens (which scales as N·p) and the quality of
the gating classifier (which is a fixed function of x, independent of N and
position).

**Key property**: the gating function is position-independent. A gate that works
for N=64 demos also works for N=224 demos. This is the fundamental mechanism
behind SSM length generalization in this task.

### Minimum State Dimension

For the task to be solvable, the state must be able to represent the correct
rule. With R possible rules and d_state dimensions:
- If rules are distinguishable in a d-dimensional space, we need d_state ≥ d
- For arity-0 rules with bounded LHS/RHS size, d = O(log R) suffices with
  appropriate encoding
- Experiment 13 uses d_state=32 with R=256 rules per transition;
  log2(256) = 8, so d_state=32 has ~4x headroom

This predicts that reducing d_state below ~8-16 should cause Mamba2 performance
to degrade, which is testable.

## Approach 5: Formal Language / Automata Perspective

### Task as a Language Recognition Problem

In the arity-0 setting, each rule is a mapping from a set of source-layer atoms
to a set of destination-layer atoms. The task can be encoded as:

```
L = { (F, g, D_1, ..., D_N, r) :
      F is a set of facts,
      g is a goal atom,
      D_i are demo rules,
      r is the unique rule in {D_i} with LHS(r) ⊆ F and g ∈ RHS(r) }
```

This language requires:
1. Set containment checking (LHS ⊆ F) — in TC^0 (threshold circuits)
2. Membership testing (g ∈ RHS) — in TC^0
3. Selecting the unique satisfying rule — requires "argmax" or "winner-take-all"
   across N candidates

Step 3 is the bottleneck. It requires comparing N candidates and selecting one,
which is essentially a MAX/ARGMAX operation. For Transformers, this is
implementable via attention in constant depth. For SSMs, it requires sequential
comparison, but the selective gating mechanism provides a natural implementation:
accumulate the best-matching rule in state, update only when a better match is
found.

### Circuit Complexity

The task is in TC^0 for fixed N (constant-depth threshold circuits), but for
*growing* N, the depth must grow as O(log N) for circuits, or the width must
be polynomial. Transformers have O(1) depth but O(N²) attention, while SSMs
have O(N) sequential steps but O(1) width per step.

The key difference: Transformers pay for growing N through attention dilution
(width is fixed, attention is spread thinner), while SSMs pay through more
sequential steps (but each step's computation is independent of N).

## Approach 6: Related Literature and Comparisons

### Directly Related Theoretical Work

1. **Arora et al. (2024), "Zoology: Measuring and Improving Recall in Efficient
   Language Models"**: Benchmarks SSMs vs Transformers on multi-query associative
   recall. Shows SSMs struggle with exact multi-query recall but can handle
   single-query recall if state is large enough. Our task is single-query
   (one correct rule per step), which is in the SSM-favorable regime.

2. **Jelassi et al. (2024), "Repeat After Me: Transformers are Better than State
   Space Models at Copying"**: Proves fundamental limitations of linear
   recurrences for exact copying. Our task does not require exact copying—it
   requires *identifying* and *reproducing* one item from context, which is
   closer to retrieval than copying. The selective gating mechanism in Mamba2
   may circumvent the copying limitation.

3. **Gu & Dao (2024), "Mamba: Linear-Time Sequence Modeling with Selective State
   Spaces"**: The selective scan mechanism is precisely the input-dependent gating
   analyzed in Approach 4 above. The original paper shows empirically that
   selection is critical for language tasks but does not provide retrieval-
   specific theory.

4. **Bietti et al. (2024), "Birth of a Transformer: A Memory Viewpoint"**:
   Analyzes Transformers as associative memories, deriving capacity bounds.
   Relevant for understanding when Transformer retrieval saturates.

5. **Olsson et al. (2022), "In-context Learning and Induction Heads"**: Induction
   heads perform a simple form of associative recall. Our task is a structured
   generalization where the "key" is not an exact token match but a set-
   containment predicate.

### Related Empirical Comparisons

6. **Waleffe et al. (2024), "An Empirical Study of Mamba-based Language Models"**:
   Systematic comparison on standard benchmarks, noting SSM advantages on long-
   context tasks and Transformer advantages on recall-heavy tasks. Our results
   are consistent: the weak-retrieval (low alpha) regime emphasizes long-context
   processing over exact recall.

7. **Kazemnejad et al. (2023), "The Impact of Positional Encoding on Length
   Generalization in Transformers"**: Shows RoPE degrades outside training
   lengths. Directly relevant to the Transformer's length generalization failure
   in experiment 13.

### Related Tasks

8. **Needle-in-a-haystack** (Kamradt, 2023): Our low-alpha regime is a
   structured version of this benchmark. The "needle" is the correct rule; the
   "haystack" is rank 3-4 distractors.

9. **BFCL / ToolBench** (Patil et al., 2023; Qin et al., 2023): Real-world tool
   selection benchmarks. Our task is a synthetic analog with controlled retriever
   quality. Theoretical results here could inform understanding of tool selection
   in realistic settings.

10. **In-context learning as Bayesian inference** (Xie et al., 2022): Each demo
    provides evidence about the latent "concept" (correct rule). Alpha controls
    the informativeness of each evidence sample. The Bayesian framework gives
    posterior convergence rates that depend on the per-sample information—directly
    analogous to our I(r*; d | alpha) quantity.

## Recommended Simplifications for Tractability

### Already in Place (Experiment 13)

- Arity 0: no unification, rule matching is set containment
- Ground atoms only: no variable binding
- Fixed rule bank structure: 256 rules per transition, known in advance

### Further Simplifications to Consider

1. **Distance 1 (single step)**: Eliminates compounding error in rollouts. Makes
   rollout_success_rate = single_step_accuracy. Currently distance=2, which
   means a model that gets each step right with probability p succeeds at the
   rollout with probability ~p². This conflates retrieval quality with
   multi-step consistency. Single-step is cleaner for isolating the retrieval
   phenomenon.

2. **Binary "tool IDs" instead of rule text**: Replace the autoregressive
   generation of rule text with a classification head over N context positions
   (i.e., "which demo is the correct one?"). This isolates the retrieval
   component from the generation component and makes the task directly comparable
   to standard associative recall setups.

3. **Random feature vectors instead of FOL atoms**: For the pure theoretical
   analysis, replace each rule with a random binary feature vector and the query
   with a target feature vector. The correct rule has maximum overlap with the
   query. This abstracts away all FOL-specific structure while preserving the
   combinatorial matching and noise characteristics.

4. **Fixed number of relevant demos**: Instead of Zipf sampling, fix exactly k
   relevant demos and N-k distractors. This removes the stochasticity in the
   number of relevant demos and makes the analysis cleaner. Alpha can be mapped
   to k/N ratio after the fact.

5. **Single attention head / single SSM layer**: For the tightest theoretical
   bounds, analyze a single attention head vs a single selective SSM layer. This
   is where the sharpest separation results will come from. Multi-layer analysis
   can be layered on top.

### Proposed Minimal Theoretical Task

The cleanest setup for proving a separation result:

- **Input**: query vector q ∈ R^d, followed by N key-value pairs (k_i, v_i)
- **One pair has k_i = q** (the "correct tool"); the rest have k_i drawn
  uniformly from the unit sphere
- **Task**: output v_i for the matching key
- **Alpha analog**: instead of one exact match, the correct key satisfies
  ⟨q, k_correct⟩ > τ, and distractors have ⟨q, k_j⟩ ~ N(0, 1/d). The
  threshold τ plays the role of alpha.

For this task:
- Single-head softmax attention: P(correct) = exp(τ√d) / (exp(τ√d) + (N-1)·c)
  where c = E[exp(⟨q,k⟩)] for random k. This decays as ~1/N for fixed τ, d.
- Single-layer selective SSM: P(correct) ≈ 1 - ε(d_state) for any N, provided
  the gate correctly identifies the matching key (learnable for τ > 0 and
  d_state ≥ d).

This yields a clean **separation theorem**: for any fixed τ > 0 and d, there
exists N* such that for all N > N*, the selective SSM outperforms single-head
softmax attention. The crossover N* ≈ exp(τ√d) / c.

## Experimental Validation Plan

The theoretical predictions above suggest specific experiments using the existing
experiment 13 infrastructure:

1. **Phase diagram**: Plot (eval_max_n_demos, alpha) → winner(Transformer,
   Mamba2) using rollout_success_rate. Compare against the predicted crossover
   boundary.

2. **Attention analysis**: For trained Transformers, extract attention weights at
   the final token and measure attention entropy and mass-on-correct-demo as a
   function of eval_max_n_demos.

3. **d_state ablation**: Train Mamba2 variants with d_state ∈ {4, 8, 16, 32, 64}
   and verify that the threshold for maintaining performance scales as predicted.

4. **Distance-1 ablation**: Repeat with distance=1 to isolate retrieval from
   multi-step effects.

5. **Classification head variant**: Replace autoregressive decoding with a
   pointer/classification head to isolate retrieval from generation.
