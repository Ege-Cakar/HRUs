"""Rule-bank utilities for layered first-order reasoning tasks.

This package re-exports every public (and two private) symbol so that existing
``from task.layer_gen.util.fol_rule_bank import ...`` statements continue to
work unchanged.
"""

from ._types import (
    FOLAtom,
    FOLDepth3ICLSplitBundle,
    FOLLayerRule,
    FOLRuleBank,
    FOLSampledProblem,
    FOLSequent,
    load_fol_depth3_icl_split_bundle,
    load_fol_rule_bank,
    parse_atom_text,
    parse_clause_text,
    parse_conjunction_text,
    parse_sequent_text,
    save_fol_depth3_icl_split_bundle,
    save_fol_rule_bank,
    # Private, but imported by task/layer_fol/task_split_depth3_fresh.py
    _normalize_count_spec,
)

from ._build import (
    build_depth3_icl_split_bundle,
    build_fresh_layer0_bank,
    build_random_fol_rule_bank,
    generate_fresh_predicate_names,
    # Private, but imported by task/layer_fol/common.py
    _FRESH_PREDICATE_CHARSET,
)

from ._sampling import (
    sample_fol_problem,
)

from ._hybrid_icl import (
    HybridICLBank,
    build_hybrid_icl_bank,
    load_hybrid_icl_bank,
    save_hybrid_icl_bank,
)

from ._hybrid_icl_sampling import (
    HybridICLSampledProblem,
    sample_hybrid_icl_problem,
)

__all__ = [
    # Data classes
    "FOLAtom",
    "FOLSequent",
    "FOLLayerRule",
    "FOLRuleBank",
    "FOLDepth3ICLSplitBundle",
    "FOLSampledProblem",
    # Construction
    "build_random_fol_rule_bank",
    "build_fresh_layer0_bank",
    "build_depth3_icl_split_bundle",
    "generate_fresh_predicate_names",
    # Parsing
    "parse_atom_text",
    "parse_conjunction_text",
    "parse_clause_text",
    "parse_sequent_text",
    # I/O
    "save_fol_rule_bank",
    "load_fol_rule_bank",
    "save_fol_depth3_icl_split_bundle",
    "load_fol_depth3_icl_split_bundle",
    # Sampling
    "sample_fol_problem",
    # Hybrid ICL
    "HybridICLBank",
    "HybridICLSampledProblem",
    "build_hybrid_icl_bank",
    "save_hybrid_icl_bank",
    "load_hybrid_icl_bank",
    "sample_hybrid_icl_problem",
    # Private re-exports (used externally)
    "_normalize_count_spec",
    "_FRESH_PREDICATE_CHARSET",
]
