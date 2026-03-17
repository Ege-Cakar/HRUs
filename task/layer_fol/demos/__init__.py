"""Demo augmentation utilities for layered FOL tasks."""

# Re-export all public symbols so that ``from task.layer_fol.demos import X``
# continues to work unchanged.

from ._unify import (
    _collect_applicable_demo_schemas,
    _find_demo_schema_instantiation,
    _find_lhs_substitutions_for_facts,
    _find_matching_demo_schema_for_rule,
    _find_oracle_schema_or_raise,
    _is_variable,
    _subst_binds_rhs_variables,
    _unify_template_atom_with_ground,
)

from ._ranking import (
    _classify_rules_by_rank,
    _is_applicable,
    _is_goal_reachable_from_rule_rhs,
    _lhs_predicates,
    _precompute_reachable_sets,
    _reachable_predicates_from_rule,
    _rhs_predicates,
)

from ._sampling import (
    _build_per_rule_pool_and_weights,
    _build_rank_level_pool_and_weights,
    _rank_order_demos,
    _resolve_demo_ranking_beta,
    _sample_demo_schemas,
    _sample_demo_schemas_full_rank,
    _sample_demo_schemas_zipf,
    _sample_demo_schemas_zipf_per_rule,
    _sample_from_weighted_pool,
    _sample_full_rank,
    _sample_uniform,
    _sample_zipf_ranked,
    sample_ranked_demos,
)

from ._clustering import (
    _batch_build_ranking_vectors,
    _batch_cluster_select,
    _build_ranks_matrix,
    _k_medoids,
    _precompute_cluster_candidate_rankings,
    _sample_cluster,
    _sample_cluster_from_precomputed,
    _sample_fresh_query_at_layer,
    _spearman_footrule_distance_matrix,
)

from ._core import (
    FOLDemoAugmentationResult,
    FOLDemoAugmentedAdapter,
    _augment_prompt_with_demos,
    _instantiate_demo_schema_with_random_constants,
    _prepend_demo_statements_to_prompt,
    augment_prompt_with_demos,
)
