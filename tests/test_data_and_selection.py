"""Schema, preprocessing, simulator honesty, and the selection machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcc.data import (
    ARRIVAL_COLUMNS, DataNotAvailable, LabelConfig, PreprocessConfig, Provenance,
    SOURCES, classify_arrival, filter_arrivals, load_source, summarise_sources,
    validate_arrivals,
)
from pcc.data.labels import label_from_possession_track
from pcc.geometry import Pitch, to_canonical, zone_index
from pcc.kinematics import LocomotionParams, cap_speed, derivative, gap_mask, smooth_positions


# --- geometry ---------------------------------------------------------------
def test_canonical_mapping_centres_the_pitch():
    out = to_canonical(np.array([[0, 0], [120, 80], [60, 40]]), source_extent=(0, 120, 0, 80))
    assert out[2] == pytest.approx([0.0, 0.0])
    assert out[0] == pytest.approx([-52.5, -34.0])
    assert out[1] == pytest.approx([52.5, 34.0])


def test_attacking_direction_flip_is_a_rotation():
    xy = np.array([[30.0, 10.0]])
    a = to_canonical(xy, source_extent=(-52.5, 52.5, -34, 34), attacking_left_to_right=True)
    b = to_canonical(xy, source_extent=(-52.5, 52.5, -34, 34), attacking_left_to_right=False)
    assert a == pytest.approx(-b)


def test_degenerate_extent_is_rejected():
    with pytest.raises(ValueError):
        to_canonical(np.zeros((1, 2)), source_extent=(0, 0, 0, 80))


def test_zones_partition_the_pitch():
    rng = np.random.default_rng(0)
    xy = np.column_stack([rng.uniform(-52.5, 52.5, 5000), rng.uniform(-34, 34, 5000)])
    z = zone_index(xy, n_x=5, n_y=3)
    assert z.min() >= 0 and z.max() <= 14
    assert len(np.unique(z)) == 15


# --- kinematics -------------------------------------------------------------
def test_smoothing_preserves_occlusion_gaps():
    t = np.linspace(0, 10, 250)
    track = np.column_stack([t, np.sin(t)])
    track[50:80] = np.nan
    out = smooth_positions(track, fps=25)
    assert np.isnan(out[50:80]).all()
    assert np.isfinite(out[:50]).all()


def test_derivative_recovers_a_known_velocity():
    fps = 25.0
    t = np.arange(0, 8, 1 / fps)
    track = np.column_stack([3.0 * t, np.zeros_like(t)])   # 3 m/s along x
    v = derivative(track, fps=fps)
    assert v[50:-50, 0] == pytest.approx(3.0, abs=1e-6)
    assert v[50:-50, 1] == pytest.approx(0.0, abs=1e-6)


def test_gap_mask_flags_only_long_gaps():
    track = np.zeros((100, 2))
    track[10:13] = np.nan       # 0.12 s at 25 fps: short
    track[40:60] = np.nan       # 0.8 s: long
    mask = gap_mask(track, fps=25.0, max_gap_seconds=0.5)
    assert not mask[10:13].any()
    assert mask[40:60].all()


def test_speed_cap_limits_magnitude_and_preserves_direction():
    v = np.array([[30.0, 40.0]])            # 50 m/s, impossible
    out = cap_speed(v, v_cap=12.0)
    assert np.linalg.norm(out) == pytest.approx(12.0)
    assert out[0, 0] / out[0, 1] == pytest.approx(30.0 / 40.0)


# --- schema and preprocessing ----------------------------------------------
def test_validate_rejects_an_inconsistent_table(sim_corpus):
    _frames, df = sim_corpus
    broken = df.copy()
    broken.loc[0, "flight_time"] = broken.loc[0, "flight_time"] + 1.0
    with pytest.raises(ValueError, match="t_arrival"):
        validate_arrivals(broken, strict=True)


def test_validate_rejects_duplicate_ids(sim_corpus):
    _frames, df = sim_corpus
    broken = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="not unique"):
        validate_arrivals(broken, strict=True)


def test_filtering_records_every_exclusion(sim_corpus):
    _frames, df = sim_corpus
    df = df.copy()
    df.loc[df.index[:20], "pass_length"] = 1.0       # too short
    df.loc[df.index[20:35], "flight_time"] = 9.0     # too long
    df.loc[df.index[20:35], "t_arrival"] = df.loc[df.index[20:35], "t_release"] + 9.0
    prov = Provenance()
    out = filter_arrivals(df, PreprocessConfig(), provenance=prov)
    steps = prov.to_frame().set_index("step")
    assert steps.loc["drop_pass_too_short", "dropped"] == 20
    assert steps.loc["drop_flight_too_long", "dropped"] == 15
    assert len(out) == len(df) - 35
    assert prov.total_dropped() >= 35


def test_low_completeness_is_flagged_not_dropped(sim_corpus):
    """RQ4 depends on these rows surviving as a stratum rather than being deleted."""
    _frames, df = sim_corpus
    df = df.copy()
    df["frame_completeness"] = 1.0            # start clean: the simulator's broadcast
    df.loc[df.index[:50], "frame_completeness"] = 0.4   # matches are already low
    out = filter_arrivals(df, PreprocessConfig(min_frame_completeness=0.75), provenance=Provenance())
    assert (out["quality_flag"] == "low_completeness").sum() == 50
    assert out["low_completeness"].sum() == 50
    assert len(out) == len(df)                # flagged, not dropped


def test_every_required_schema_column_is_produced(sim_corpus):
    _frames, df = sim_corpus
    required = [c for c, _t, req, _d in ARRIVAL_COLUMNS if req]
    assert not set(required) - set(df.columns)


# --- labelling --------------------------------------------------------------
def test_label_at_horizon():
    """Possession flips to B at t = 2.0 s, so the horizon decides the label."""
    times = np.arange(0, 5, 0.1)
    poss = pd.Series(["A"] * 20 + ["B"] * 30)     # A holds for t < 2.0

    early = label_from_possession_track(poss, times, t_arrival=0.5, team_a="A",
                                        config=LabelConfig(horizon=1.0))
    assert early["y"] == 1                         # evaluated at t = 1.5: A
    assert not early["censored"]

    late = label_from_possession_track(poss, times, t_arrival=0.5, team_a="A",
                                       config=LabelConfig(horizon=2.0))
    assert late["y"] == 0                          # evaluated at t = 2.5: B

    # The horizon is a substantive choice, not a detail: the same arrival is
    # labelled both ways depending on it. This is why the study pre-registers a
    # horizon sensitivity sweep rather than fixing one silently.
    assert early["y"] != late["y"]


def test_label_is_censored_by_a_stoppage():
    times = np.arange(0, 5, 0.1)
    poss = pd.Series(["A"] * 50)
    stop = pd.Series([False] * 10 + [True] * 5 + [False] * 35)
    out = label_from_possession_track(poss, times, t_arrival=0.5, team_a="A",
                                      config=LabelConfig(horizon=1.0), stoppage=stop)
    assert out["censored"] is True


def test_arrival_classification_marks_exogenous_types():
    for label in ("clearance", "deflection", "block", "rebound", "duel"):
        _type, endogenous = classify_arrival(label)
        assert endogenous is False
    assert classify_arrival("pass") == ("open_pass", True)
    assert classify_arrival("corner")[0] == "set_piece"


def test_label_config_rejects_an_unknown_definition():
    with pytest.raises(ValueError):
        LabelConfig(definition="whatever")


# --- source registry --------------------------------------------------------
def test_verified_status_is_backed_by_recorded_evidence():
    """A source may claim verified contents only if it records what was checked.

    This is the invariant that keeps the registry honest as sources are
    verified one by one: the claim and the evidence move together, so a status
    cannot be upgraded by editing one field.
    """
    for key, src in SOURCES.items():
        if src.status == "verified_contents" and key != "simulated":
            assert src.verified_fields, f"{key} claims verified contents with no verified_fields"
            assert "checked_on" in src.verified_fields, f"{key} records no verification date"
        if src.status == "unverified":
            assert "not verified" in src.licence_note.lower(), (
                f"{key} is unverified but its licence_note does not say so"
            )


def test_every_external_source_carries_caveats_and_a_licence_note():
    for key, src in SOURCES.items():
        if key != "simulated":
            assert src.caveats, f"{key} has no recorded caveats"
            assert src.licence_note.strip(), f"{key} has no licence note"


def test_verified_caveats_are_marked_as_verified():
    """Caveats established from the files are distinguishable from assumed ones."""
    for key, src in SOURCES.items():
        if src.status == "verified_contents" and key != "simulated":
            assert any(c.startswith("VERIFIED") for c in src.caveats), (
                f"{key} claims verified contents but no caveat is marked VERIFIED"
            )


def test_missing_provider_data_raises_an_actionable_error():
    with pytest.raises((DataNotAvailable, NotImplementedError)) as exc:
        load_source("metrica_sample", root="/definitely/not/here")
    assert "data/raw" in str(exc.value) or "scaffold" in str(exc.value)


def test_source_summary_flags_the_fallback_only_sources():
    rows = {r["key"]: r for r in summarise_sources()}
    assert rows["statsbomb_open"]["meets_primary_requirements"] is False
    assert rows["statsbomb_open"]["freeze_frames"] is True
    assert rows["statsbomb_open"]["player_tracking"] is False


# --- simulator honesty ------------------------------------------------------
def test_simulated_rows_are_labelled_as_simulated(sim_corpus):
    _frames, df = sim_corpus
    assert (df["tracking_source"] == "simulated").all()


def test_simulator_reproduces_the_selection_effect(sim_corpus):
    """Chosen destinations must succeed more often than unchosen ones.

    If this fails, the simulator no longer exercises the selection machinery and
    scripts/05 is testing nothing.
    """
    _frames, df = sim_corpus
    chosen = df[df["is_endogenous"]]["y_control"].mean()
    unchosen = df[~df["is_endogenous"]]["y_control"].mean()
    assert chosen > unchosen + 0.05


def test_simulator_is_reproducible():
    from pcc.data.synthetic import SimulationConfig, simulate_dataset

    cfg = SimulationConfig(n_matches=2, arrivals_per_match=50, random_state=99)
    _f1, a = simulate_dataset(cfg)
    _f2, b = simulate_dataset(cfg)
    pd.testing.assert_frame_equal(a, b)


# --- selection --------------------------------------------------------------
def test_candidates_stay_on_the_pitch(sim_corpus):
    from pcc.selection import CandidateConfig, generate_candidates

    frames, _df = sim_corpus
    cands, parent = generate_candidates(frames[:40], CandidateConfig(n_per_arrival=5))
    assert len(cands) == 200
    assert parent.max() == 39
    pitch = Pitch()
    targets = np.array([c.target for c in cands])
    assert pitch.contains(targets, tol=0.01).all()


def test_selection_model_detects_that_passers_choose(sim_corpus):
    from pcc.models.features import build_feature_matrix
    from pcc.selection import CandidateConfig, SelectionModel, generate_candidates

    frames, _df = sim_corpus
    cands, _ = generate_candidates(frames[:300], CandidateConfig(n_per_arrival=4, random_state=1))
    sel = SelectionModel().fit(build_feature_matrix(frames[:300]), build_feature_matrix(cands))
    auc = sel.discrimination(build_feature_matrix(frames[:300]), build_feature_matrix(cands))
    assert auc > 0.6, "destinations should be clearly non-random"


def test_weights_are_stabilised_and_ess_is_reported(sim_corpus):
    from pcc.models.features import build_feature_matrix
    from pcc.selection import (
        CandidateConfig, SelectionModel, effective_sample_size, generate_candidates,
        overlap_diagnostics, selection_weights,
    )

    frames, _df = sim_corpus
    real_X = build_feature_matrix(frames[:300])
    cands, _ = generate_candidates(frames[:300], CandidateConfig(n_per_arrival=4, random_state=2))
    cand_X = build_feature_matrix(cands)
    sel = SelectionModel().fit(real_X, cand_X)
    w = selection_weights(sel, real_X)
    assert w.mean() == pytest.approx(1.0, abs=1e-6)
    ess = effective_sample_size(w)
    assert 0 < ess <= len(w)
    diag = overlap_diagnostics(sel, real_X, cand_X)
    assert 0 <= diag["ess_ratio"] <= 1


def test_tipping_point_is_flat_at_gamma_one(sim_corpus):
    from pcc.selection import tipping_point_analysis

    _frames, df = sim_corpus
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(4, 2, len(df)), 0.01, 0.99)
    tab = tipping_point_analysis(df["y_control"].to_numpy(), p, gamma_grid=[1.0, 2.0])
    at_one = tab[tab["gamma"] == 1.0]
    assert at_one["corp_mcb"].nunique() == 1      # deflated == inflated at Gamma = 1


def test_negative_control_splits_chosen_from_unchosen(sim_corpus):
    from pcc.selection import negative_control_comparison

    _frames, df = sim_corpus
    rng = np.random.default_rng(3)
    work = df.assign(p=np.clip(rng.beta(5, 2, len(df)), 0.01, 0.99))
    out = negative_control_comparison(work, prob_col="p", outcome_col="y_control", n_boot=40)
    assert set(out["subsample"]) == {"chosen", "unchosen"}


# --- frame serialisation (the corpus cache) ---------------------------------
def test_frame_roundtrip_preserves_everything_a_model_reads(sim_corpus, tmp_path):
    """A cache that changes the data is worse than no cache at all."""
    import numpy as np

    from pcc.data.tracking import load_frames, save_frames
    from pcc.models import PhysicalControl, VoronoiControl

    frames, _df = sim_corpus
    path = tmp_path / "frames.npz"
    save_frames(frames[:400], path)
    restored = load_frames(path)

    assert len(restored) == 400
    for a, b in zip(frames[:400], restored):
        assert np.allclose(a.att_xy, b.att_xy, atol=1e-3)
        assert np.allclose(a.att_v, b.att_v, atol=1e-3)
        assert np.allclose(a.def_xy, b.def_xy, atol=1e-3)
        assert np.allclose(a.target, b.target, atol=1e-3)
        assert abs(a.flight_time - b.flight_time) < 1e-3
        assert (a.att_is_gk == b.att_is_gk).all()
        assert (a.def_observed == b.def_observed).all()
        assert np.allclose(a.meta["origin"], b.meta["origin"], atol=1e-3)

    # The property that actually matters: models must not notice the roundtrip.
    for model in (VoronoiControl(), PhysicalControl()):
        assert np.allclose(model.predict(frames[:200]), model.predict(restored[:200]), atol=1e-4)


def test_frame_roundtrip_handles_an_empty_corpus(tmp_path):
    from pcc.data.tracking import load_frames, save_frames

    path = tmp_path / "empty.npz"
    save_frames([], path)
    assert load_frames(path) == []


def test_frame_roundtrip_handles_ragged_player_counts(tmp_path):
    """Substitutions, occlusions and freeze frames all give uneven team sizes."""
    import numpy as np

    from pcc.data.schema import ArrivalFrame
    from pcc.data.tracking import load_frames, save_frames

    rng = np.random.default_rng(0)
    frames = [
        ArrivalFrame(
            att_xy=rng.uniform(-40, 40, (n_a, 2)), att_v=rng.normal(0, 2, (n_a, 2)),
            def_xy=rng.uniform(-40, 40, (n_b, 2)), def_v=rng.normal(0, 2, (n_b, 2)),
            target=np.array([1.0, 2.0]), flight_time=1.0,
            meta={"origin": np.array([-3.0, 4.0])},
        )
        for n_a, n_b in [(11, 11), (8, 10), (3, 4), (11, 7)]
    ]
    path = tmp_path / "ragged.npz"
    save_frames(frames, path)
    restored = load_frames(path)
    assert [f.n_att for f in restored] == [11, 8, 3, 11]
    assert [f.n_def for f in restored] == [11, 10, 4, 7]
