"""Metrica adapter tests.

These run only when the data is present, so the suite stays green on a clean
checkout. They are the tests that would have caught the two real bugs found
while writing the adapter:

* the attacking-direction inference using the team mean rather than the
  goalkeeper, which mirrors the pitch for the team taking the kick-off;
* a filter loop evaluating its masks against a stale frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
METRICA_ROOT = REPO / "data" / "raw" / "metrica_sample"

pytestmark = pytest.mark.skipif(
    not (METRICA_ROOT / "data").is_dir(),
    reason="Metrica sample data not present; fetch with scripts/fetch_data.py",
)


@pytest.fixture(scope="module")
def game_dir():
    from pcc.data.metrica import available_games

    games = available_games(METRICA_ROOT)
    assert games, "no CSV-layout games found"
    return games[0]


@pytest.fixture(scope="module")
def tracking(game_dir):
    from pcc.data.metrica import load_tracking

    return load_tracking(game_dir)


@pytest.fixture(scope="module")
def loaded():
    from pcc.data import load_source

    return load_source("metrica_sample", root=str(METRICA_ROOT))


def test_only_csv_layout_games_are_offered():
    from pcc.data.metrica import available_games

    names = [g.name for g in available_games(METRICA_ROOT)]
    assert "Sample_Game_1" in names
    # Game 3 is EPTS/FIFA format and must not be silently half-read.
    assert "Sample_Game_3" not in names


def test_tracking_shape_and_frame_rate(tracking):
    assert tracking.fps == 25.0
    assert tracking.n_frames > 100_000
    assert tracking.n_players >= 22
    tracking.validate()


def test_coordinates_are_metric_and_on_the_pitch(tracking):
    xy = tracking.xy.reshape(-1, 2)
    finite = np.isfinite(xy).all(axis=1)
    inside = tracking.pitch.contains(xy[finite], tol=5.0)
    assert inside.mean() > 0.98


def test_y_axis_was_negated(game_dir):
    """Metrica's origin is top-left, so metric y must be the negation of raw y."""
    from pcc.data.metrica import to_metric

    out = to_metric(np.array([[0.5, 0.0], [0.5, 1.0]]))
    assert out[0, 1] > 0 and out[1, 1] < 0
    assert out[0, 1] == pytest.approx(34.0)


def test_goalkeepers_are_identified_one_per_team(tracking):
    for team in ("Home", "Away"):
        cols = np.flatnonzero(tracking.teams == team)
        assert tracking.is_gk[cols].sum() == 1


def test_goalkeepers_are_the_deepest_players(tracking):
    """The inference is only credible if the keeper really is an outlier in x."""
    for team in ("Home", "Away"):
        cols = np.flatnonzero(tracking.teams == team)
        first = tracking.period == tracking.period[0]
        # Substitutes are all-NaN in the first period; guard rather than warn.
        means = np.array([
            np.nanmean(tracking.xy[first, c, 0]) if np.isfinite(tracking.xy[first, c, 0]).any()
            else np.nan
            for c in cols
        ])
        gk = tracking.is_gk[cols]
        outfield = np.abs(means[~gk])
        assert np.abs(means[gk][0]) > np.nanmax(outfield), team


def test_attack_signs_are_opposed_and_flip_at_half_time(tracking):
    """The bug this catches mirrors the pitch while leaving aggregates plausible."""
    signs = tracking.attack_sign
    for period in (1, 2):
        assert signs[("Home", period)] == -signs[("Away", period)], period
    assert signs[("Home", 1)] == -signs[("Home", 2)]


def test_arrivals_load_and_conform_to_the_schema(loaded):
    from pcc.data import validate_arrivals

    frames, df = loaded
    assert len(frames) == len(df) > 1500
    validate_arrivals(df, strict=True)


def test_passes_progress_forward_on_average(loaded):
    """Positive mean progression confirms the canonical frame is not mirrored."""
    _frames, df = loaded
    assert (df["dest_x"] - df["origin_x"]).mean() > 1.0


def test_outcome_rates_are_football_plausible(loaded):
    """Derived labels must order the way football says they should.

    Open passes are mostly retained; interceptions almost never are. If this
    ordering ever breaks, the possession derivation is wrong, and no calibration
    result computed on it would mean anything.
    """
    _frames, df = loaded
    by_type = df.groupby("arrival_type")["y_control"].mean()
    assert 0.85 < by_type["open_pass"] < 1.0
    assert by_type["interception"] < 0.35
    assert by_type["interception"] < by_type["open_pass"]
    assert 0.7 < df["y_control"].mean() < 0.95


def test_both_teams_appear_as_the_team_in_possession(loaded):
    _frames, df = loaded
    assert df["team_a"].nunique() == 2
    assert (df["team_a"] != df["team_b"]).all()


def test_state_is_taken_at_release_not_arrival(loaded):
    """Every frame's origin must sit near a team-A player at release.

    If the state were taken at arrival, the ball's origin would no longer
    coincide with a passer. This is the check that the outcome is not leaking
    into the model's inputs.
    """
    frames, df = loaded
    rng = np.random.default_rng(0)
    sample = rng.choice(len(frames), size=200, replace=False)
    distances = []
    for i in sample:
        f = frames[i]
        origin = np.asarray(f.meta["origin"], dtype=float)
        distances.append(float(np.min(np.linalg.norm(f.att_xy - origin[None, :], axis=1))))
    assert float(np.median(distances)) < 3.0


def test_models_run_on_real_frames(loaded):
    from pcc.models import PhysicalControl, VoronoiControl

    frames, _df = loaded
    for model in (VoronoiControl(), PhysicalControl()):
        p = model.predict(frames[:200])
        assert p.shape == (200,)
        assert np.all((p >= 0) & (p <= 1))


def test_physics_model_has_signal_on_real_data(loaded):
    """A weak but real check: the model must beat the base rate on real football."""
    from pcc.calibration import brier_score, roc_auc
    from pcc.models import PhysicalControl

    frames, df = loaded
    y = df["y_control"].to_numpy()[:1200]
    p = PhysicalControl().predict(frames[:1200])
    assert roc_auc(y, p) > 0.6
    assert brier_score(y, p) < brier_score(y, np.full(y.size, y.mean()))
