"""SkillCorner adapter tests, including regressions for two coordinate bugs.

Both bugs were mirroring bugs, and both are close to invisible in aggregate:
they leave base rates, pass lengths, flight times and outcome orderings
completely intact, and show up only as a mean pass progression that is not
clearly forward.

1. **Double attack-normalisation.** SkillCorner's event coordinates are already
   expressed with the acting team playing toward +x, while its tracking is in a
   fixed match frame. Applying the canonical rotation to already-normalised
   event coordinates mirrors one team every period. Progression read -0.59 m.
2. **Wrong frame for the destination.** After a failed pass the next possession
   belongs to the *opponent*, whose coordinates are normalised to the opposite
   direction. Converting the destination with the passer's sign mirrors every
   unsuccessful arrival. Progression read +0.71 m instead of +4.16 m.

The guard that should have caught the first one was set at -2.0 m and did not
fire; it is now +0.5 m, and `test_direction_guard_fires_on_a_mirrored_corpus`
holds it to that.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SC_ROOT = REPO / "data" / "raw" / "skillcorner_open"


def _resolved_matches():
    try:
        from pcc.data.skillcorner import available_matches

        return available_matches(SC_ROOT)
    except Exception:
        return []


needs_data = pytest.mark.skipif(
    not _resolved_matches(),
    reason="SkillCorner tracking not present or unresolved; "
           "fetch with scripts/fetch_data.py --source skillcorner_open",
)


# --------------------------------------------------------------------------
# Guards that need no data
# --------------------------------------------------------------------------
def test_lfs_pointer_is_rejected_with_an_actionable_message(tmp_path):
    from pcc.data.skillcorner import LFSPointerError, _check_not_lfs_pointer

    stub = tmp_path / "x_tracking_extrapolated.jsonl"
    stub.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 90729279\n"
    )
    with pytest.raises(LFSPointerError) as exc:
        _check_not_lfs_pointer(stub)
    assert "fetch_data.py" in str(exc.value)


def _warnings_from(fn, *args, **kwargs) -> list[str]:
    """Collect warnings from one call, immune to the warning registry.

    ``pytest.warns`` relies on the default "once per location" filter, and the
    guard under test warns with ``stacklevel=3`` - which points at its caller's
    caller, so repeated calls from one test line are deduplicated and the
    assertion fails for reasons unrelated to the guard.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(*args, **kwargs)
    return [str(w.message) for w in caught]


def test_direction_guard_fires_on_a_mirrored_corpus():
    """The guard must fire at the magnitudes the two real bugs produced."""
    from pcc.data.tracking import _check_direction_of_play

    n = 500

    def table(progression: float) -> pd.DataFrame:
        return pd.DataFrame({"origin_x": np.zeros(n), "dest_x": np.full(n, progression)})

    clean = [m for m in _warnings_from(_check_direction_of_play, table(4.0)) if "not clearly" in m]
    assert clean == [], "a clearly forward corpus must not be flagged"

    # -0.59 was the double-normalisation bug; +0.71 the wrong-destination-frame
    # bug. The previous -2.0 m threshold caught neither.
    for observed in (-0.59, 0.0, 0.71):
        messages = _warnings_from(_check_direction_of_play, table(observed))
        assert any("not clearly" in m for m in messages), f"guard missed progression={observed}"


def test_direction_guard_stays_quiet_below_the_row_threshold():
    """Too few arrivals to judge is not the same as evidence of mirroring."""
    from pcc.data.tracking import _check_direction_of_play

    tiny = pd.DataFrame({"origin_x": np.zeros(10), "dest_x": np.full(10, -5.0)})
    flagged = [m for m in _warnings_from(_check_direction_of_play, tiny) if "not clearly" in m]
    assert flagged == []


def test_only_structural_event_columns_are_permitted():
    """SkillCorner's own fitted metrics must never become model inputs."""
    from pcc.data.skillcorner import STRUCTURAL_COLUMNS

    contaminating = [
        "xpass_completion", "xthreat", "possession_epv_for_start", "pass_epv_total",
        "reception_difficulty_start", "overall_pressure_start", "passing_option_ease_start",
        "xloss_player_possession_start", "xshot_player_possession_start",
        "player_targeted_xpass_completion", "player_targeted_xthreat",
    ]
    for column in contaminating:
        assert column not in STRUCTURAL_COLUMNS, f"{column} is a fitted metric, not an observation"


# --------------------------------------------------------------------------
# Corpus tests
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def match_dir():
    return _resolved_matches()[0]


@pytest.fixture(scope="module")
def tracking(match_dir):
    from pcc.data.skillcorner import load_match_meta, load_tracking

    meta = load_match_meta(match_dir)
    table, possession = load_tracking(match_dir, meta)
    return meta, table, possession


@pytest.fixture(scope="module")
def loaded():
    from pcc.data import load_source

    return load_source("skillcorner_open", root=str(SC_ROOT), max_matches=2)


@needs_data
def test_metadata_supplies_true_pitch_and_stated_direction(tracking):
    """Neither has to be inferred here - unusually among the three providers."""
    meta, table, _poss = tracking
    assert 95.0 <= meta["pitch"].length <= 115.0
    assert 60.0 <= meta["pitch"].width <= 75.0
    assert len(meta["home_side"]) >= 2
    assert set(meta["home_side"]) <= {"left_to_right", "right_to_left"}
    table.validate()


@needs_data
def test_frame_rate_is_ten_hz(tracking):
    _meta, table, _poss = tracking
    assert table.fps == 10.0
    assert table.n_frames > 40_000


@needs_data
def test_attack_signs_agree_with_goalkeeper_positions(tracking):
    """Cross-check the stated direction against where the keepers actually stand."""
    meta, table, _poss = tracking
    for period in (1, 2):
        mask = np.flatnonzero(table.period == period)
        for team in (meta["home"], meta["away"]):
            cols = np.flatnonzero((table.teams == team) & table.is_gk)
            if cols.size == 0:
                continue
            gk_x = float(np.nanmean(table.xy[np.ix_(mask, cols)][..., 0]))
            sign = table.attack_sign[(team, period)]
            # A keeper defends the goal opposite the direction their team attacks.
            assert np.sign(gk_x) == -np.sign(sign), (team, period, gk_x, sign)


@needs_data
def test_observed_mask_records_detection_not_interpolation(tracking):
    """The corpus extrapolates every player every frame; `observed` must not.

    If this ever reads as fully observed, the adapter has started trusting
    extrapolated positions, and the RQ4 contrast this corpus exists for is gone.
    """
    _meta, table, _poss = tracking
    on_pitch = np.isfinite(table.xy).all(axis=2)
    assert on_pitch.mean() > 0.5, "players should be filled in on most frames"
    detected_given_on_pitch = table.observed[on_pitch].mean()
    assert 0.2 < detected_given_on_pitch < 0.95, (
        f"detection rate {detected_given_on_pitch:.3f}: expected partial detection"
    )
    assert table.observed.sum() < on_pitch.sum()


@needs_data
def test_possession_label_is_provider_supplied_and_sparse(tracking):
    _meta, _table, possession = tracking
    assert possession.notna().any()
    assert possession.isna().mean() > 0.1, "provider possession is known to be sparse"


@needs_data
def test_arrivals_conform_to_the_schema(loaded):
    from pcc.data import validate_arrivals

    frames, df = loaded
    assert len(frames) == len(df) > 1000
    validate_arrivals(df, strict=True)
    assert (df["tracking_source"] == "broadcast").all()


@needs_data
def test_passes_progress_clearly_forward(loaded):
    """The regression for both mirroring bugs.

    Before the fixes this read -0.59 m and then +0.71 m, with every other
    summary statistic unchanged.
    """
    _frames, df = loaded
    progression = (df["dest_x"] - df["origin_x"]).mean()
    assert progression > 2.0, f"mean progression {progression:.2f} m suggests a mirrored pitch"


@needs_data
def test_lost_passes_are_longer_and_more_forward_than_retained_ones(loaded):
    """A football check that the destination frame is right for BOTH outcomes.

    The second bug mirrored only the unsuccessful arrivals, so a test that
    looked at the pooled mean alone could miss it.
    """
    _frames, df = loaded
    prog = df.assign(p=df["dest_x"] - df["origin_x"]).groupby("y_control")["p"].mean()
    assert prog.loc[0] > prog.loc[1], "lost passes should be the more ambitious ones"
    assert prog.loc[0] > 0


@needs_data
def test_outcome_rates_order_as_football_requires(loaded):
    _frames, df = loaded
    by_type = df.groupby("arrival_type")["y_control"].mean()
    assert by_type["open_pass"] > 0.8
    if "cross" in by_type:
        assert by_type["cross"] < by_type["open_pass"]
    assert 0.7 < df["y_control"].mean() < 0.92


@needs_data
def test_frame_completeness_is_never_full(loaded):
    """Broadcast tracking cannot see all 22; completeness is a real covariate."""
    _frames, df = loaded
    assert df["frame_completeness"].max() < 1.0
    assert 0.4 < df["frame_completeness"].median() < 0.95
    assert df["frame_completeness"].std() > 0.02, "completeness must actually vary"


@needs_data
def test_velocity_is_present_unlike_the_freeze_frame_corpus(loaded):
    """The whole point of the RQ4 arm: broadcast tracking is still tracking."""
    frames, _df = loaded
    speeds = np.concatenate([np.linalg.norm(f.att_v, axis=1) for f in frames[:300]])
    assert np.isfinite(speeds).all()
    assert speeds.max() > 1.0
    assert speeds.max() <= 12.0 + 1e-6, "speed cap should bound derived velocity"


@needs_data
def test_models_run_on_broadcast_frames(loaded):
    from pcc.models import PhysicalControl, VoronoiControl

    frames, _df = loaded
    for model in (VoronoiControl(), PhysicalControl()):
        p = model.predict(frames[:200])
        assert p.shape == (200,)
        assert np.all((p >= 0) & (p <= 1))
