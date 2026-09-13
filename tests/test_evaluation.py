"""Splits must not leak, and the protocol must not fit anything on the test fold."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcc.evaluation import (
    EvaluationConfig, add_subgroups, grouped_holdout, grouped_kfold, leakage_audit,
    leave_one_competition_out, leave_one_source_out, net_benefit, run_protocol,
    spatial_calibration_map, stratified_metrics, temporal_split, threshold_displacement,
)


@pytest.fixture
def panel():
    n_matches, per_match = 12, 200
    m = np.repeat([f"m{i:02d}" for i in range(n_matches)], per_match)
    return pd.DataFrame(
        {
            "match_id": m,
            "possession_id": np.repeat([f"p{i:04d}" for i in range(n_matches * per_match // 5)], 5),
            "competition": np.where(pd.Series(m).str[1:].astype(int) < 6, "A", "B"),
            "tracking_source": np.where(pd.Series(m).str[1:].astype(int) % 2 == 0, "optical", "broadcast"),
            "match_date": pd.to_datetime("2026-01-01") + pd.to_timedelta(
                pd.Series(m).str[1:].astype(int) * 7, unit="D"
            ),
            "dest_x": np.random.default_rng(0).uniform(-50, 50, n_matches * per_match),
            "dest_y": np.random.default_rng(1).uniform(-30, 30, n_matches * per_match),
            "pass_length": np.random.default_rng(2).uniform(3, 50, n_matches * per_match),
            "flight_time": np.random.default_rng(3).uniform(0.2, 2.5, n_matches * per_match),
            "pressure_index": np.random.default_rng(4).integers(0, 4, n_matches * per_match),
            "score_diff": 0,
            "y": np.random.default_rng(5).binomial(1, 0.75, n_matches * per_match),
        }
    )


def test_holdout_keeps_whole_matches_together(panel):
    split = grouped_holdout(panel, random_state=0)
    split.assert_disjoint(panel, "match_id")
    split.assert_disjoint(panel, "possession_id")
    assert split.train.size + split.valid.size + split.test.size == len(panel)


def test_leakage_audit_reports_zero_overlap(panel):
    split = grouped_holdout(panel, random_state=1)
    audit = leakage_audit(panel, split)
    assert (audit[["overlap_train_test", "overlap_train_valid", "overlap_valid_test"]] == 0).all().all()


def test_a_deliberately_leaky_split_is_caught(panel):
    """The guard must actually fire; a guard that cannot fail is not a guard."""
    from pcc.evaluation.splits import Split

    idx = np.arange(len(panel))
    bad = Split(name="leaky", train=idx[::2], valid=np.array([], dtype=int), test=idx[1::2])
    with pytest.raises(AssertionError, match="leaks"):
        bad.assert_disjoint(panel, "match_id")


def test_stratified_holdout_balances_the_strata(panel):
    split = grouped_holdout(panel, stratify_col="competition", random_state=2)
    for fold in (split.train, split.valid, split.test):
        assert panel.iloc[fold]["competition"].nunique() == 2


def test_kfold_covers_every_row_exactly_once(panel):
    tested = []
    for split in grouped_kfold(panel, n_splits=4, random_state=3):
        split.assert_disjoint(panel, "match_id")
        tested.append(split.test)
    combined = np.concatenate(tested)
    assert np.array_equal(np.sort(combined), np.arange(len(panel)))


def test_leave_one_competition_out(panel):
    splits = list(leave_one_competition_out(panel))
    assert len(splits) == 2
    for s in splits:
        assert panel.iloc[s.test]["competition"].nunique() == 1
        assert not set(panel.iloc[s.test]["competition"]) & set(panel.iloc[s.train]["competition"])


def test_leave_one_source_out(panel):
    for s in leave_one_source_out(panel):
        assert panel.iloc[s.test]["tracking_source"].nunique() == 1
        assert not set(panel.iloc[s.test]["tracking_source"]) & set(panel.iloc[s.train]["tracking_source"])


def test_temporal_split_puts_the_future_in_the_test_fold(panel):
    s = temporal_split(panel)
    assert panel.iloc[s.test]["match_date"].min() > panel.iloc[s.train]["match_date"].max()


def test_subgroups_are_attached(panel):
    out = add_subgroups(panel)
    for col in ("zone", "third", "pass_length_band", "pressure_band", "game_state", "flight_band"):
        assert col in out.columns
    assert out["zone"].nunique() > 1


def test_stratified_metrics_suppresses_small_strata(panel):
    out = add_subgroups(panel).assign(p=0.75)
    tab = stratified_metrics(out, prob_col="p", outcome_col="y", group_col="zone",
                             min_n=10_000, n_boot=20)
    assert tab["suppressed"].all()


def test_spatial_map_marks_insufficient_cells(panel):
    cells = spatial_calibration_map(panel.assign(p=0.7), prob_col="p", outcome_col="y", min_n=10_000)
    assert not cells["sufficient"].any()
    assert "n" in cells.columns


def test_net_benefit_reference_policies_are_correct(panel):
    y = panel["y"].to_numpy()
    nb = net_benefit(y, np.full(y.size, 0.8), thresholds=[0.5])
    prevalence = y.mean()
    assert nb["net_benefit_all"].iloc[0] == pytest.approx(prevalence - (1 - prevalence) * 1.0)
    assert nb["net_benefit_none"].iloc[0] == 0.0
    # Forecasting 0.8 everywhere means acting on everything at a 0.5 threshold.
    assert nb["net_benefit_model"].iloc[0] == pytest.approx(nb["net_benefit_all"].iloc[0])


def test_threshold_displacement_counts_flips():
    y = np.array([1, 1, 0, 0])
    raw = np.array([0.9, 0.55, 0.45, 0.1])
    cal = np.array([0.9, 0.45, 0.55, 0.1])
    out = threshold_displacement(y, raw, cal, thresholds=[0.5])
    assert out["frac_decisions_changed"].iloc[0] == pytest.approx(0.5)


def test_protocol_refuses_to_recalibrate_without_a_validation_fold(sim_corpus):
    from pcc.evaluation.protocol import apply_recalibration, fit_predict
    from pcc.evaluation.splits import Split
    from pcc.models import LogisticControl

    frames, df = sim_corpus
    y = df["y_control"].to_numpy(dtype=int)
    idx = np.arange(len(frames))
    split = Split(name="novalid", train=idx[:800], valid=np.array([], dtype=int), test=idx[800:])
    model = LogisticControl()
    preds = fit_predict(model, frames, y, split)
    with pytest.raises(ValueError, match="validation fold"):
        apply_recalibration(preds, model, frames, y, split)


def test_full_protocol_runs_and_returns_the_documented_tables(sim_corpus):
    from pcc.models import MarginalBaseline, PhysicalControl, VoronoiControl

    frames, df = sim_corpus
    from pcc.evaluation import add_subgroups

    meta = add_subgroups(df).reset_index(drop=True)
    y = meta["y_control"].to_numpy(dtype=int)
    split = grouped_holdout(meta, random_state=0)
    out = run_protocol(
        [MarginalBaseline(), VoronoiControl(), PhysicalControl()],
        frames, y, meta, split,
        EvaluationConfig(n_boot=25, recalibrators=("identity", "platt")),
    )
    for key in ("metrics", "comparisons", "bootstrap", "leakage", "predictions"):
        assert key in out
    assert set(out["metrics"]["variant"]) == {"raw", "identity", "platt"}
    assert len(out["predictions"]) == split.test.size
