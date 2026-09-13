"""The calibration estimator must be correct before it is used to judge anything."""

from __future__ import annotations

import numpy as np
import pytest

from pcc.calibration.metrics import (
    adaptive_calibration_error, brier_score, calibration_slope_intercept,
    corp_decomposition, corp_decomposition_cv, expected_calibration_error, full_report,
    log_loss, murphy_decomposition, roc_auc, sharpness, spherical_score,
)


def test_brier_matches_definition():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.2, 0.6, 0.4])
    assert brier_score(y, p) == pytest.approx(np.mean((p - y) ** 2))


def test_brier_is_proper_the_truth_minimises_it():
    """A strictly proper score is minimised in expectation at the true probability."""
    rng = np.random.default_rng(0)
    p_true = 0.3
    y = rng.binomial(1, p_true, 200_000)
    scores = {q: brier_score(y, np.full(y.size, q)) for q in (0.1, 0.2, 0.3, 0.4, 0.5)}
    assert min(scores, key=scores.get) == pytest.approx(0.3)


def test_log_loss_is_proper_too():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.7, 200_000)
    scores = {q: log_loss(y, np.full(y.size, q)) for q in (0.5, 0.6, 0.7, 0.8, 0.9)}
    assert min(scores, key=scores.get) == pytest.approx(0.7)


def test_spherical_score_is_maximised_at_the_truth():
    rng = np.random.default_rng(2)
    y = rng.binomial(1, 0.4, 200_000)
    scores = {q: spherical_score(y, np.full(y.size, q)) for q in (0.2, 0.3, 0.4, 0.5, 0.6)}
    assert max(scores, key=scores.get) == pytest.approx(0.4)


def test_corp_identity_holds(calibrated_sample):
    p, y, _ = calibrated_sample
    d = corp_decomposition(y, p)
    assert d.score == pytest.approx(d.miscalibration - d.discrimination + d.uncertainty, abs=1e-10)


def test_corp_detects_a_calibrated_forecast(calibrated_sample):
    p, y, _ = calibrated_sample
    assert corp_decomposition(y, p).miscalibration < 0.003


def test_corp_detects_injected_miscalibration(calibrated_sample):
    p, y, _ = calibrated_sample
    logit = np.log(p / (1 - p))
    bad = 1 / (1 + np.exp(-2.0 * logit))
    assert corp_decomposition(y, bad).miscalibration > corp_decomposition(y, p).miscalibration + 0.005


def test_in_sample_corp_is_biased_upward(calibrated_sample):
    """In-sample PAV absorbs noise, so it OVERSTATES miscalibration.

    This is the reason the protocol requires the cross-fitted variant for any
    headline claim, and it is why a cross-fitted MCB is allowed to be negative.
    """
    p, y, m = calibrated_sample
    in_sample = corp_decomposition(y, p).miscalibration
    cross = corp_decomposition_cv(y, p, m, n_splits=5).miscalibration
    assert in_sample > cross
    assert in_sample > 0.0            # a calibrated forecast still shows positive in-sample MCB
    assert abs(cross) < abs(in_sample)


def test_in_sample_corp_bias_grows_as_the_sample_shrinks(calibrated_sample):
    p, y, m = calibrated_sample
    small = slice(0, 600)
    big_bias = (corp_decomposition(y, p).miscalibration
                - corp_decomposition_cv(y, p, m, n_splits=5).miscalibration)
    small_bias = (corp_decomposition(y[small], p[small]).miscalibration
                  - corp_decomposition_cv(y[small], p[small], m[small], n_splits=5).miscalibration)
    assert small_bias > big_bias


def test_calibration_slope_recovers_a_known_distortion(calibrated_sample):
    p, y, _ = calibrated_sample
    for factor in (0.7, 1.0, 1.5, 2.0):
        distorted = 1 / (1 + np.exp(-factor * np.log(p / (1 - p))))
        slope = calibration_slope_intercept(y, distorted)["slope"]
        assert slope == pytest.approx(1.0 / factor, abs=0.06), f"factor={factor}"


def test_auc_is_invariant_to_monotone_recalibration(calibrated_sample):
    """The point of the study: discrimination cannot see miscalibration."""
    p, y, _ = calibrated_sample
    squashed = 1 / (1 + np.exp(-3.0 * np.log(p / (1 - p))))
    assert roc_auc(y, p) == pytest.approx(roc_auc(y, squashed), abs=1e-9)


def test_base_rate_forecast_is_calibrated_but_useless(calibrated_sample):
    """Calibration alone is not a sufficient criterion; sharpness must be reported."""
    p, y, _ = calibrated_sample
    flat = np.full(y.size, y.mean())
    d = corp_decomposition(y, flat)
    assert d.miscalibration < 1e-6
    assert d.discrimination < 1e-6
    assert sharpness(flat)["forecast_sd"] == pytest.approx(0.0)
    assert np.isnan(roc_auc(y, flat)) or roc_auc(y, flat) == pytest.approx(0.5, abs=1e-9)


def test_murphy_identity_is_exact_for_discrete_forecasts():
    """The three-term identity holds exactly only when forecasts are bin-valued."""
    rng = np.random.default_rng(4)
    levels = np.array([0.1, 0.35, 0.5, 0.75, 0.9])
    p = rng.choice(levels, 50_000)
    y = rng.binomial(1, p)
    d = murphy_decomposition(y, p, n_bins=10)
    assert d["residual"] == pytest.approx(0.0, abs=1e-9)
    assert brier_score(y, p) == pytest.approx(
        d["reliability"] - d["resolution"] + d["uncertainty"], abs=1e-9
    )


def test_murphy_identity_needs_the_residual_for_continuous_forecasts(calibrated_sample):
    p, y, _ = calibrated_sample
    d = murphy_decomposition(y, p, n_bins=10)
    without = d["reliability"] - d["resolution"] + d["uncertainty"]
    assert brier_score(y, p) != pytest.approx(without, abs=1e-9)
    assert brier_score(y, p) == pytest.approx(without + d["residual"], abs=1e-9)


def test_coarse_bins_can_hide_miscalibration_that_corp_detects():
    """The concrete reason binned ECE is not the study's primary metric.

    ECE averages the *absolute* gap per bin, so errors only cancel when they
    have opposite signs **inside one bin**. Here the forecast alternates between
    over- and under-forecasting four times across the unit interval; with two
    bins each bin contains one positive and one negative region and its mean gap
    is near zero, so the ECE is near zero. With forty bins the same forecast is
    plainly miscalibrated, and the binning-free CORP statistic sees it either
    way.
    """
    rng = np.random.default_rng(5)
    n = 200_000
    p = rng.uniform(0.1, 0.9, n)
    # Sign flips at 0.3 and 0.7 - the midpoints of the two equal-width bins -
    # so each bin holds one +0.1 region and one -0.1 region of equal mass.
    offset = np.where((p < 0.3) | ((p >= 0.5) & (p < 0.7)), 0.1, -0.1)
    truth = np.clip(p + offset, 0.01, 0.99)
    y = rng.binomial(1, truth)

    ece_coarse = expected_calibration_error(y, p, n_bins=2)
    ece_fine = expected_calibration_error(y, p, n_bins=40)
    mcb = corp_decomposition(y, p).miscalibration

    assert ece_coarse < 0.02
    assert ece_fine > 0.08
    assert mcb > 0.005


def test_weighted_metrics_respect_weights():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    w = np.array([10.0, 10.0, 0.0, 0.0])
    assert brier_score(y, p, w) == pytest.approx(0.01)


def test_ace_uses_equal_mass_bins(calibrated_sample):
    p, y, _ = calibrated_sample
    assert 0.0 <= adaptive_calibration_error(y, p, n_bins=10) < 0.05


def test_full_report_has_every_documented_key(calibrated_sample):
    p, y, m = calibrated_sample
    rep = full_report(y, p, groups=m)
    for key in ("brier", "corp_mcb", "corp_dsc", "corp_unc", "calibration_slope",
                "calibration_in_the_large", "auc", "ece_uniform", "ace_quantile",
                "forecast_sd", "base_rate"):
        assert key in rep


def test_rejects_non_binary_outcomes():
    with pytest.raises(ValueError):
        brier_score(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        brier_score(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))
