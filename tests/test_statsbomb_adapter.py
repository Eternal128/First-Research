"""StatsBomb adapter tests, including regressions for two real bugs.

The two bugs, both found only because a *second* provider was added:

1. **Period-mixing.** The outcome lookup sorted all frames by timestamp alone.
   StatsBomb's clock restarts each period, so a first-half arrival could be
   labelled from a second-half frame with a similar timestamp. Metrica has a
   continuous match clock and was unaffected, which is precisely why a
   single-provider test suite could not have caught it. The effect was large:
   the corpus base rate read 0.59 instead of 0.91.
2. **NaN truthiness.** ``event["pass_outcome"] or "COMPLETE"`` never fires for
   a missing outcome, because ``float('nan')`` is truthy in Python. Completed
   passes were left labelled ``NaN``.

The polygon and coordinate tests run without any data; the rest skip when the
corpus is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pcc.data.labels import LabelConfig, label_from_possession_track
from pcc.data.statsbomb import SB_LENGTH, SB_WIDTH, point_in_polygon, to_metric

REPO = Path(__file__).resolve().parents[1]
SB_ROOT = REPO / "data" / "raw" / "statsbomb_open"
needs_data = pytest.mark.skipif(
    not (SB_ROOT / "events").is_dir(),
    reason="StatsBomb data not present; fetch with scripts/fetch_data.py",
)


# --------------------------------------------------------------------------
# Pure functions - no data needed
# --------------------------------------------------------------------------
def test_coordinate_mapping_centres_and_flips_y():
    """StatsBomb's origin is top-left, so metric y must be negated."""
    out = to_metric([[SB_LENGTH / 2, SB_WIDTH / 2], [0, 0], [SB_LENGTH, SB_WIDTH]])
    assert out[0] == pytest.approx([0.0, 0.0])
    assert out[1][1] > 0 and out[2][1] < 0        # y flipped
    assert out[1][0] == pytest.approx(-52.5)
    assert out[2][0] == pytest.approx(52.5)


def test_point_in_polygon_on_a_square():
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert point_in_polygon((5, 5), square)
    assert not point_in_polygon((15, 5), square)
    assert not point_in_polygon((5, -1), square)
    assert not point_in_polygon((-0.1, 5), square)


def test_point_in_polygon_on_a_concave_shape():
    """An L-shape: the notch must read as outside."""
    L = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], dtype=float)
    assert point_in_polygon((2, 2), L)
    assert point_in_polygon((8, 2), L)
    assert point_in_polygon((2, 8), L)
    assert not point_in_polygon((8, 8), L)        # the notch


# --------------------------------------------------------------------------
# Regression: period-aware labelling
# --------------------------------------------------------------------------
def test_label_lookup_respects_period_when_the_clock_restarts():
    """The bug: a period-1 arrival labelled from a period-2 frame.

    Both periods run 0-100 s. Team A holds the ball throughout period 1 and
    team B throughout period 2. Without period filtering the lookup sorts by
    timestamp and picks whichever frame happens to sort first at t = 50.
    """
    times = np.concatenate([np.arange(0, 100, 1.0), np.arange(0, 100, 1.0)])
    periods = np.concatenate([np.ones(100, dtype=int), np.full(100, 2, dtype=int)])
    poss = pd.Series(["A"] * 100 + ["B"] * 100)
    cfg = LabelConfig(horizon=1.0)

    first = label_from_possession_track(poss, times, 50.0, "A", cfg, periods=periods, arrival_period=1)
    second = label_from_possession_track(poss, times, 50.0, "A", cfg, periods=periods, arrival_period=2)
    assert first["y"] == 1
    assert second["y"] == 0

    # Without the period arguments the two are indistinguishable - the bug.
    ambiguous = label_from_possession_track(poss, times, 50.0, "A", cfg)
    assert ambiguous["y"] in (0, 1)


def test_label_censors_an_arrival_in_a_period_with_no_frames():
    times = np.arange(0, 100, 1.0)
    periods = np.ones(100, dtype=int)
    poss = pd.Series(["A"] * 100)
    out = label_from_possession_track(
        poss, times, 50.0, "A", LabelConfig(horizon=1.0), periods=periods, arrival_period=2
    )
    assert out["censored"] is True


def test_continuous_clock_corpora_are_unaffected_by_the_fix():
    """Metrica-style single continuous clock: period arguments change nothing."""
    times = np.arange(0, 200, 1.0)
    periods = np.where(times < 100, 1, 2)
    poss = pd.Series(["A"] * 200)
    cfg = LabelConfig(horizon=1.0)
    with_p = label_from_possession_track(poss, times, 150.0, "A", cfg, periods=periods, arrival_period=2)
    without = label_from_possession_track(poss, times, 150.0, "A", cfg)
    assert with_p["y"] == without["y"] == 1


# --------------------------------------------------------------------------
# Corpus tests
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def loaded():
    from pcc.data import load_source

    return load_source("statsbomb_open", root=str(SB_ROOT), max_matches=4)


@needs_data
def test_corpus_loads_and_conforms_to_the_schema(loaded):
    from pcc.data import validate_arrivals

    frames, df = loaded
    assert len(frames) == len(df) > 1000
    validate_arrivals(df, strict=True)


@needs_data
def test_pass_outcome_is_not_left_as_nan(loaded):
    """Regression: NaN is truthy, so `x or "COMPLETE"` never fired."""
    _frames, df = loaded
    assert df["pass_outcome"].notna().all()
    assert (df["pass_outcome"] == "COMPLETE").mean() > 0.5


@needs_data
def test_derived_label_agrees_with_completed_passes(loaded):
    """The derived construct must track the provider's own outcome where they agree.

    Agreement on completed passes is the check that the possession derivation is
    sound. Divergence on *incomplete* passes is expected and is not an error: an
    incomplete pass is often still recovered by the same team within the horizon,
    which is exactly the distinction between "the pass completed" and "the team
    controls the ball", i.e. between the event label and this study's construct.
    """
    _frames, df = loaded
    complete = df[df["pass_outcome"] == "COMPLETE"]["y_control"].mean()
    incomplete = df[df["pass_outcome"] == "Incomplete"]["y_control"].mean()
    assert complete > 0.90
    assert incomplete < complete
    assert 0.2 < incomplete < 0.8, "incomplete passes are neither always nor never recovered"


@needs_data
def test_base_rate_is_football_plausible(loaded):
    """Guards the period-mixing regression, which read 0.59 instead of 0.91."""
    _frames, df = loaded
    assert 0.80 < df["y_control"].mean() < 0.97
    assert df["outcome_censored"].mean() < 0.05


@needs_data
def test_events_are_already_attack_normalised(loaded):
    _frames, df = loaded
    assert (df["dest_x"] - df["origin_x"]).mean() > 0.5


@needs_data
def test_flight_time_is_measured_not_imputed(loaded):
    """StatsBomb supplies a duration for every pass, so nothing is imputed."""
    _frames, df = loaded
    assert not df["flight_time_imputed"].any()
    assert 5.0 < df["ball_speed"].median() < 25.0


@needs_data
def test_no_frame_carries_velocity(loaded):
    """The defining limitation of the fallback design, asserted rather than assumed."""
    frames, _df = loaded
    for f in frames[:300]:
        assert np.allclose(f.att_v, 0.0)
        assert np.allclose(f.def_v, 0.0)


@needs_data
def test_visibility_is_recorded_and_incomplete(loaded):
    """Roughly one destination in six is outside the camera's covered region."""
    _frames, df = loaded
    assert "dest_visible" in df.columns
    rate = df["dest_visible"].mean()
    assert 0.6 < rate < 0.95, f"unexpected visibility rate {rate:.3f}"
    hidden = df[~df["dest_visible"]]
    assert (hidden["quality_flag"] == "destination_not_visible").all()


@needs_data
def test_frames_almost_never_show_all_22_players(loaded):
    """Freeze frames are camera-limited; completeness is a first-class covariate.

    This test previously asserted that *no* frame shows all 22 players, which was
    measured on a single match and failed as soon as the corpus grew: at scale it
    happens at roughly 1 arrival in 2,500. The assertion is now about the rate,
    not about impossibility - a categorical claim from one match was exactly the
    kind of over-reach this study exists to criticise.
    """
    _frames, df = loaded
    complete = float((df["frame_completeness"] >= 1.0).mean())
    assert complete < 0.01, f"{complete:.4f} of frames fully observed: unexpectedly high"
    assert 0.6 < df["frame_completeness"].median() < 0.95


@needs_data
def test_pass_height_separates_aerial_from_ground(loaded):
    """A construct check: aerial arrivals must be more contested than ground ones."""
    _frames, df = loaded
    by_height = df.groupby("pass_height")["y_control"].mean()
    assert by_height["high"] < by_height["ground"]


@needs_data
def test_models_run_on_freeze_frame_states(loaded):
    from pcc.models import PhysicalControl, VoronoiControl

    frames, _df = loaded
    for model in (VoronoiControl(), PhysicalControl()):
        p = model.predict(frames[:200])
        assert p.shape == (200,)
        assert np.all((p >= 0) & (p <= 1))


@needs_data
def test_physics_model_discriminates_but_is_not_calibrated(loaded):
    """The fallback design's defining property, asserted rather than assumed.

    On camera-limited freeze frames the zero-velocity physics model **ranks**
    situations well (AUC comfortably above chance) while its **values** are
    badly miscalibrated: it under-forecasts control, because the camera centres
    on the ball and a long pass destination frequently has no visible player of
    either team, leaving the competing-risks integration near 0.5 where reality
    is near 0.9.

    The consequence is the study's central thesis in one line: the model's Brier
    score is **worse than forecasting the base rate everywhere**, while its AUC
    says it is a good model. Any evaluation resting on discrimination alone
    would report this model as working.

    This test would fail if the adapter were ever changed to silently impute
    unseen players, which would be the natural "fix" and would destroy the very
    effect the fallback design exists to measure.
    """
    from pcc.calibration import brier_score, calibration_slope_intercept, roc_auc
    from pcc.models import PhysicalControl

    frames, df = loaded
    y = df["y_control"].to_numpy()
    p = PhysicalControl().predict(frames)
    constant = np.full(y.size, y.mean())

    assert roc_auc(y, p) > 0.65, "the model should still rank situations well"
    assert brier_score(y, p) > brier_score(y, constant), (
        "on this corpus the raw forecast is worse than a constant; if this ever "
        "reverses, check whether unseen players are being imputed"
    )
    assert calibration_slope_intercept(y, p)["slope"] < 0.6, "expected severe overconfidence in scale"
    assert p.mean() < y.mean() - 0.05, "expected systematic under-forecasting"


@needs_data
def test_recalibration_rescues_the_freeze_frame_forecast(loaded):
    """If the ordering is sound, recalibration should recover the score.

    Fitted on some matches and evaluated on the others, so the recovery is out
    of sample and grouped by match. This is the pre-registered "deflating"
    outcome: the physics model's *ranking* of situations is sound and only its
    scale was wrong, which means the defect is fixable downstream rather than
    fatal.
    """
    from pcc.calibration import IsotonicRecalibrator, PlattScaling, brier_score
    from pcc.models import PhysicalControl

    frames, df = loaded
    y = df["y_control"].to_numpy()
    p = PhysicalControl().predict(frames)
    matches = df["match_id"].to_numpy()
    unique = np.unique(matches)
    assert unique.size >= 4, "this test needs enough matches to split by match"
    fit = np.isin(matches, unique[: unique.size // 2])
    held = ~fit

    before = brier_score(y[held], p[held])
    constant = brier_score(y[held], np.full(int(held.sum()), y[fit].mean()))

    scores = {}
    for mapping in (PlattScaling(), IsotonicRecalibrator()):
        mapping.fit(p[fit], y[fit])
        scores[mapping.name] = brier_score(y[held], mapping.transform(p[held]))
        assert scores[mapping.name] < before, f"{mapping.name} should improve a badly-scaled forecast"

    # The non-parametric map reaches past the constant forecast; the
    # two-parameter one only just reaches it. That gap is the signature of
    # distortion that is NOT logit-linear - hypothesis H5b - and is the
    # reportable nuance here, so it is asserted rather than smoothed over.
    assert scores["isotonic"] < constant, "isotonic should recover enough to beat a constant"
    assert scores["isotonic"] <= scores["platt"] + 1e-9, (
        "isotonic should be at least as good as Platt; if Platt ever matches it, "
        "the distortion has become logit-linear and H5b should be revisited"
    )
