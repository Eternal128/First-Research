"""Proper scores, reliability curves, recalibration maps and cluster inference."""

from pcc.calibration.bootstrap import (
    BootstrapResult,
    bootstrap_reliability_band,
    cluster_bootstrap,
    design_effect,
    paired_cluster_bootstrap,
)
from pcc.calibration.metrics import (
    CORPDecomposition,
    adaptive_calibration_error,
    brier_score,
    calibration_slope_intercept,
    corp_decomposition,
    corp_decomposition_cv,
    expected_calibration_error,
    full_report,
    log_loss,
    maximum_calibration_error,
    murphy_decomposition,
    roc_auc,
    sharpness,
)
from pcc.calibration.recalibration import (
    RECALIBRATORS,
    BetaCalibration,
    HierarchicalRecalibrator,
    IdentityRecalibrator,
    IsotonicRecalibrator,
    PlattScaling,
)
from pcc.calibration.reliability import binned_reliability, corp_reliability, reliability_by_group

__all__ = [
    "brier_score", "log_loss", "roc_auc", "sharpness",
    "corp_decomposition", "corp_decomposition_cv", "CORPDecomposition",
    "murphy_decomposition", "expected_calibration_error",
    "maximum_calibration_error", "adaptive_calibration_error",
    "calibration_slope_intercept", "full_report",
    "binned_reliability", "corp_reliability", "reliability_by_group",
    "IdentityRecalibrator", "PlattScaling", "IsotonicRecalibrator",
    "BetaCalibration", "HierarchicalRecalibrator", "RECALIBRATORS",
    "cluster_bootstrap", "paired_cluster_bootstrap", "design_effect", "BootstrapResult",
    "bootstrap_reliability_band",
]
