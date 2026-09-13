"""Recalibration maps and clustered inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcc.calibration import (
    BetaCalibration, HierarchicalRecalibrator, IdentityRecalibrator, IsotonicRecalibrator,
    PlattScaling, RECALIBRATORS, brier_score, bootstrap_reliability_band,
    calibration_slope_intercept, cluster_bootstrap, corp_decomposition, design_effect,
    paired_cluster_bootstrap,
)


@pytest.fixture
def miscalibrated():
    rng = np.random.default_rng(21)
    m = np.repeat(np.arange(24), 500)
    z = rng.normal(0, 1.3, m.size) + rng.normal(0, 0.4, 24)[m]
    p_true = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p_true)
    p_bad = 1 / (1 + np.exp(-1.9 * z))
    return pd.DataFrame({"y": y, "p_true": p_true, "p": p_bad, "match_id": m})


def test_identity_is_a_no_op(miscalibrated):
    p = miscalibrated["p"].to_numpy()
    assert np.allclose(IdentityRecalibrator().fit_transform(p, miscalibrated["y"]), p)


@pytest.mark.parametrize("name", ["platt", "isotonic", "beta"])
def test_recalibration_improves_a_proper_score_out_of_sample(miscalibrated, name):
    df = miscalibrated
    fit = df["match_id"] < 12
    rc = RECALIBRATORS[name]()
    rc.fit(df.loc[fit, "p"], df.loc[fit, "y"])
    held = df.loc[~fit]
    before = brier_score(held["y"], held["p"])
    after = brier_score(held["y"], rc.transform(held["p"]))
    assert after < before


@pytest.mark.parametrize("name", ["platt", "isotonic", "beta"])
def test_recalibration_reduces_miscalibration_out_of_sample(miscalibrated, name):
    df = miscalibrated
    fit = df["match_id"] < 12
    rc = RECALIBRATORS[name]()
    rc.fit(df.loc[fit, "p"], df.loc[fit, "y"])
    held = df.loc[~fit]
    before = corp_decomposition(held["y"], held["p"]).miscalibration
    after = corp_decomposition(held["y"], rc.transform(held["p"])).miscalibration
    assert after < before


def test_platt_slope_equals_the_calibration_slope(miscalibrated):
    """Internal consistency: the two diagnostics must agree by construction."""
    df = miscalibrated
    platt = PlattScaling().fit(df["p"], df["y"])
    slope = calibration_slope_intercept(df["y"], df["p"])["slope"]
    assert platt.b == pytest.approx(slope, rel=1e-6)


def test_recalibration_does_not_change_the_ranking(miscalibrated):
    """Every map here is monotone, so AUC must be untouched."""
    from pcc.calibration import roc_auc

    df = miscalibrated
    base = roc_auc(df["y"], df["p"])
    for name in ("platt", "beta"):
        rc = RECALIBRATORS[name]().fit(df["p"], df["y"])
        assert roc_auc(df["y"], rc.transform(df["p"])) == pytest.approx(base, abs=1e-9)


def test_isotonic_fitted_and_evaluated_in_sample_looks_perfect(miscalibrated):
    """Demonstrates why the protocol forbids it: any model can be made to look calibrated."""
    df = miscalibrated
    rc = IsotonicRecalibrator().fit(df["p"], df["y"])
    in_sample = corp_decomposition(df["y"], rc.transform(df["p"])).miscalibration
    assert in_sample < 1e-9


def test_beta_calibration_has_three_parameters(miscalibrated):
    df = miscalibrated
    b = BetaCalibration().fit(df["p"], df["y"])
    assert np.isfinite([b.a, b.b, b.c]).all()
    assert BetaCalibration.n_parameters == 3


def test_hierarchical_shrinks_small_strata_toward_the_pooled_map(miscalibrated):
    """The shrunk parameter must be exactly the stated convex combination.

    Testing the weight and the combination directly, rather than the distance
    to the pooled estimate, is what makes this a test of the estimator: when the
    unshrunk local fit happens to sit close to the pooled one, a
    distance-ordering assertion passes or fails for reasons unrelated to
    shrinkage.
    """
    df = miscalibrated
    strata = np.where(df["match_id"] < 2, "tiny", "big")
    h = HierarchicalRecalibrator(kappa=5000.0).fit(df["p"], df["y"], strata=strata)

    a_tiny, b_tiny, n_tiny, w_tiny = h.strata["tiny"]
    a_big, b_big, n_big, w_big = h.strata["big"]

    assert n_tiny < n_big
    assert w_tiny < w_big                      # the smaller stratum is shrunk harder
    assert w_tiny == pytest.approx(n_tiny / (n_tiny + h.kappa))

    for key in ("tiny", "big"):
        a, b, n, w = h.strata[key]
        local = PlattScaling().fit(df.loc[strata == key, "p"], df.loc[strata == key, "y"])
        assert b == pytest.approx(w * local.b + (1 - w) * h.global_map.b)
        assert a == pytest.approx(w * local.a + (1 - w) * h.global_map.a)


def test_hierarchical_without_strata_is_just_the_pooled_map(miscalibrated):
    df = miscalibrated
    h = HierarchicalRecalibrator().fit(df["p"], df["y"])
    pooled = PlattScaling().fit(df["p"], df["y"])
    assert np.allclose(h.transform(df["p"]), pooled.transform(df["p"]))


def test_cluster_bootstrap_is_wider_than_the_naive_one(miscalibrated):
    df = miscalibrated.assign(row_id=np.arange(len(miscalibrated)))
    clustered = cluster_bootstrap(lambda d: brier_score(d["y"], d["p"]), df,
                                  cluster_col="match_id", n_boot=200, random_state=0)
    naive = cluster_bootstrap(lambda d: brier_score(d["y"], d["p"]), df,
                              cluster_col="row_id", n_boot=200, random_state=0)
    assert (clustered.hi - clustered.lo) > (naive.hi - naive.lo)


def test_bootstrap_interval_contains_the_point_estimate(miscalibrated):
    res = cluster_bootstrap(lambda d: brier_score(d["y"], d["p"]), miscalibrated,
                            cluster_col="match_id", n_boot=200, random_state=1)
    assert res.lo <= res.point <= res.hi
    assert res.n_clusters == 24


def test_paired_bootstrap_detects_a_real_difference(miscalibrated):
    res = paired_cluster_bootstrap(
        lambda d: brier_score(d["y"], d["p"]), miscalibrated,
        col_a="p", col_b="p_true", cluster_col="match_id", n_boot=200, random_state=2,
    )
    assert res.lo > 0  # the distorted forecast is genuinely worse


def test_design_effect_exceeds_one_for_clustered_data(miscalibrated):
    d = design_effect(miscalibrated, value_col="y", cluster_col="match_id")
    assert d["design_effect"] > 1.0
    assert 0 <= d["icc"] <= 1


def test_reliability_band_brackets_the_curve(miscalibrated):
    band = bootstrap_reliability_band(miscalibrated, prob_col="p", outcome_col="y",
                                      n_bins=5, n_boot=80, random_state=0)
    assert (band["boot_lo"] <= band["obs_freq"] + 1e-9).all()
    assert (band["boot_hi"] >= band["obs_freq"] - 1e-9).all()


def test_bootstrap_rejects_an_unknown_cluster_column(miscalibrated):
    with pytest.raises(KeyError):
        cluster_bootstrap(lambda d: 0.0, miscalibrated, cluster_col="not_a_column", n_boot=5)
