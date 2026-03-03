# 2026-03-03 - Maze FOL Rule Bank Design

## Motivation

The current `FOLRuleBank` + `sample_fol_problem` pipeline works well at moderate depths (~24 layers) but fundamentally cannot scale to 1000+ layers. The core issue is that `sample_fol_problem` discovers feasible paths via trial-and-error forward chaining: it picks random initial facts, fires random applicable rules one layer at a time, and hopes to reach the target depth without hitting a dead end. In a sparse rule graph, the probability of a random walk surviving L steps drops exponentially with L. At depth 1000, sampling would essentially never succeed.

This also limits the kinds of experiments we can run. Right now there's no way to directly control the connectivity structure of the predicate graph -- things like bottleneck layers, varying sparsity across depth, or guaranteed long-range paths. These are exactly the properties that would let us study how graph topology influences generalization.

## Key Design Decision: Per-Layer Adjacency + Segment-Tree Reachability

The central idea is to separate the *structural* question ("which predicate at layer i can reach which predicate at layer j?") from the *syntactic* question ("what does the rule look like?") and precompute the structural answer.

**Per-layer adjacency**: Each transition (layer i -> i+1) gets its own sparse directed graph over predicate indices. An edge (src_idx, dst_idx) means there exists a rule that maps a predicate at position `src_idx` in layer i to position `dst_idx` in layer i+1. This is a departure from the current system where rules are sampled independently and the implicit adjacency is an emergent (and uncontrolled) property.

**Segment tree over boolean adjacency matrices**: To answer "can predicate p at layer i reach predicate q at layer j?" we compose the adjacency matrices for transitions i, i+1, ..., j-1. Naively this is O(j - i) matrix multiplications per query. A segment tree precomputes composed matrices for power-of-2 ranges, reducing queries to O(log L) compositions.

The key insight making this practical: with N=64 predicates per layer and branching factor ~3, each adjacency matrix has ~192 nonzero entries out of 4096 possible. Sparse boolean matrix multiplication is O(N * avg_degree), which is ~192 operations. The full segment tree for L=1000 layers has ~4000 nodes, so precomputation is ~768K operations -- negligible.

**Path finding via divide-and-conquer**: Given that we know src can reach dst over [from_layer, to_layer), we find a concrete path by splitting the range at the midpoint, finding a "bridge" predicate reachable from src in the left half that can also reach dst in the right half, then recursing. This is O(L) total.

## Why Not Simpler Approaches

I considered several alternatives:

- **BFS/DFS per query**: Direct graph search from (layer_0, pred_p) to (layer_L, pred_q) over the product graph. This works but is O(N * L) per query with no amortization. The segment tree amortizes across queries and enables O(log L) reachability checks, which matters when sampling thousands of problems from the same bank.

- **Transitive closure**: Precompute the full reachable set for every (layer, predicate) pair. Storage is O(N * L * N) = O(N^2 * L). At N=64, L=1000 this is ~4M entries -- feasible but wasteful. More importantly, we'd need the closure for arbitrary sub-ranges, not just from layer 0. The segment tree is more flexible and more memory-efficient.

- **Random walk with backtracking**: Enhance the current sampler with backtracking when it hits dead ends. This is algorithmically simpler but still has exponential worst-case behavior in sparse graphs with bottlenecks, which are exactly the topologies we want to study.

## Data Structure Sketch

Three main types:

1. **`FOLMazeRuleTemplate`**: A rule that references predicates by *index* (0..N-1) rather than by layered name (e.g., "r5_3"). This template is layer-agnostic -- it specifies variable patterns and predicate positions but not which physical layer it applies to. The `instantiate(src_layer, pred_names_src, pred_names_dst, var_pool)` method produces a concrete `FOLLayerRule`.

2. **`FOLMazeTransition`**: One layer's worth of edges and their associated rule templates, plus precomputed forward/backward adjacency dicts and the sparse boolean adjacency matrix for segment-tree composition.

3. **`FOLMazeBank`**: The top-level container. Holds all transitions, the reachability cache, predicate arities, constants. Provides `to_fol_rule_bank()` to materialize a windowed `FOLRuleBank` for integration with the existing tokenizer and evaluation infrastructure.

The interop with `FOLRuleBank` is important: the maze bank is a *generator* of rule banks, not a replacement. All downstream code (tokenization, `match_rule_completion_fol`, rollout evaluation) continues to work against `FOLRuleBank`. The maze bank just provides a more structured way to build them and a much faster way to sample feasible problems.

## Maze Generation Modes

Three planned modes for the initial implementation:

- **`random_sparse`**: Each transition gets independently sampled sparse adjacency (each predicate gets exactly `branching_factor` random outgoing edges). Dead ends and bottlenecks emerge naturally from the sparsity. This is the primary mode for studying how graph structure interacts with learning.

- **`random_sparse_with_backbone`**: Same as `random_sparse` but first lays down a random Hamiltonian path through all predicates at each transition. This guarantees every predicate participates in at least one long-range path, preventing the graph from fragmenting into disconnected components.

- **`custom`**: User provides explicit edge lists per transition. For manual construction of specific topologies (e.g., known bottleneck positions, tree structures, grids).

The `homogeneous` flag controls whether all transitions share the same adjacency or each gets its own. Homogeneous mode means the graph is the same at every layer (just the predicate names differ); non-homogeneous means the connectivity can vary, creating layer-specific bottlenecks.

## Problem Sampling with Reachability

The new `sample_maze_fol_problem` flow:

1. Pick random `start_layer` and `start_pred`
2. Query `forward_reachable(start_pred, start_layer, start_layer + distance)` to get the set of reachable goal predicates
3. If empty, try another start (with reachability precomputed, this check is O(log L))
4. Pick a random goal predicate from the reachable set
5. Use `find_path` to get the concrete sequence of predicate indices
6. For each step, look up the rule template for that edge, instantiate it with random constants, build ground facts

This eliminates the retry loop entirely for path feasibility. The only remaining source of failure is if the randomly chosen start predicate has no long-range reachability, but this is detectable in O(log L) and recoverable by trying another start.

## Integration with FOLLayerTask

The plan is to add a `maze_bank_path` parameter to `FOLLayerTask.__init__`. When set:
- Load the `FOLMazeBank` from JSON
- Use `sample_maze_fol_problem` for online sampling (no need for the retry-heavy `sample_fol_problem`)
- Materialize a windowed `FOLRuleBank` covering all predicate names/arities for the tokenizer
- Follow the pattern of `depth3_fresh_icl` for integration since it similarly does per-sample generation with a pre-built tokenizer

The maze sampler should be fast enough for synchronous mode initially. Adding prefetch (process pool, server backend) can come later if profiling shows it's needed.

## Open Questions and Future Directions

1. **Distractor rules**: The current plan generates exactly one rule per edge. In the existing system, there are many rules per transition, most of which are distractors (applicable but not on the solution path). For the maze bank, we could add distractor edges/rules to increase difficulty. This is straightforward (just add more edges to each transition) but the right density needs empirical tuning.

2. **Variable complexity**: Starting with k_in=1, k_out=1 (single-atom LHS/RHS). This means the reasoning task reduces to simple predicate-chaining. Multi-atom LHS (k_in > 1) would require the model to identify which subset of available facts to use, which is closer to the full theorem-proving problem. Worth exploring once the basic infrastructure works.

3. **Topology experiments**: Once the maze bank is working, the interesting experiments are:
   - How does generalization change with depth? (depth 10 vs 100 vs 1000 on the same graph)
   - How do bottleneck layers affect learning? (place a single-predicate bottleneck at layer 500)
   - Does the model learn reachability implicitly? (test on start/goal pairs it hasn't seen during training)
   - How does connectivity (branching factor) interact with depth?

4. **Curriculum over depth**: Start training on short paths, gradually increase. The maze bank makes this trivial since `sample_maze_fol_problem` takes `distance` as a parameter and the reachability cache handles any distance efficiently.

5. **Serialization granularity**: The full maze bank (1000 layers, 64 predicates, branching 3) serializes to a moderate JSON file. For very large banks, might want binary serialization. Low priority.

## Next Steps

1. Implement core data structures and sparse matrix utilities in `task/layer_gen/util/fol_maze_bank.py`
2. Implement segment-tree reachability cache with `forward_reachable`, `backward_reachable`, `find_path`
3. Implement maze graph generation (`random_sparse`, `random_sparse_with_backbone`, `custom`)
4. Implement `FOLMazeBank` container with `to_fol_rule_bank()` and `sample_problem()`
5. Implement `build_maze_fol_bank()` builder function
6. Write unit tests (`tests/test_fol_maze_bank.py`) covering correctness of reachability, path validity, rule instantiation, and scale
7. Integrate with `FOLLayerTask` via `maze_bank_path` parameter
8. Run scale benchmarks: build + 1000 samples at L=1000, N=64, branching=3
