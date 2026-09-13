"""Selection bias: candidate arrivals, density-ratio weights, diagnostics, sensitivity."""

from pcc.selection.ipw import (
    CandidateConfig,
    SelectionModel,
    effective_sample_size,
    generate_candidates,
    matched_subsample,
    negative_control_comparison,
    overlap_diagnostics,
    propensity_strata,
    selection_weights,
    standardised_mean_differences,
    tipping_point_analysis,
)

__all__ = [
    "CandidateConfig", "generate_candidates", "SelectionModel", "selection_weights",
    "effective_sample_size", "overlap_diagnostics", "propensity_strata",
    "negative_control_comparison", "matched_subsample", "standardised_mean_differences",
    "tipping_point_analysis",
]
