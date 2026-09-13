"""Downstream propagation, figures, and an end-to-end run of the scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from pcc.downstream import (  # noqa: E402
    PositionalValueSurface, downstream_sensitivity, expected_possession_value,
    off_ball_run_value, pass_value_comparison, space_metrics,
)
from pcc.models import PhysicalControl  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def surface(sim_corpus):
    _frames, df = sim_corpus
    return PositionalValueSurface.fit_from_outcomes(
        df[["dest_x", "dest_y"]].to_numpy(), df["y_control"].to_numpy(),
        provenance="fitted in tests",
    )


def test_placeholder_surface_refuses_silent_use():
    """A fabricated value surface must not be constructible by accident."""
    with pytest.raises(ValueError, match="acknowledge_placeholder"):
        PositionalValueSurface.distance_placeholder()
    ok = PositionalValueSurface.distance_placeholder(acknowledge_placeholder=True)
    assert "PLACEHOLDER" in ok.provenance


def test_value_surface_records_its_provenance(surface):
    assert surface.provenance
    assert np.isfinite(surface.grid).all()


def test_value_lookup_returns_nan_for_missing_coordinates(surface):
    out = surface.value_at(np.array([[0.0, 0.0], [np.nan, 3.0]]))
    assert np.isfinite(out[0])
    assert np.isnan(out[1])


def test_epv_is_monotone_in_control(surface):
    dest = np.tile(np.array([30.0, 5.0]), (5, 1))
    control = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    epv = expected_possession_value(control, dest, surface)
    assert np.all(np.diff(epv) > 0)


def test_epv_is_linear_in_control(surface):
    """Linearity is the property that makes calibration error propagate additively."""
    dest = np.tile(np.array([20.0, -10.0]), (3, 1))
    epv = expected_possession_value(np.array([0.2, 0.5, 0.8]), dest, surface)
    assert (epv[1] - epv[0]) == pytest.approx(epv[2] - epv[1])


def test_a_constant_calibration_shift_biases_total_epv_in_one_direction(surface, sim_corpus):
    """The accumulation argument: systematic error does not average away."""
    _frames, df = sim_corpus
    dest = df[["dest_x", "dest_y"]].to_numpy()
    honest = np.clip(np.full(len(df), 0.70), 0, 1)
    overconfident = np.clip(honest + 0.05, 0, 1)
    total_honest = expected_possession_value(honest, dest, surface).sum()
    total_over = expected_possession_value(overconfident, dest, surface).sum()
    assert total_over > total_honest
    assert (total_over - total_honest) / len(df) > 0.01


def test_space_metrics_are_plausible(arrival_frame):
    m = space_metrics(PhysicalControl(), arrival_frame)
    pitch_area = 105.0 * 68.0
    assert 0 <= m["space_owned_m2"] <= pitch_area
    assert 0 <= m["soft_space_owned_m2"] <= pitch_area
    assert 0 <= m["control_entropy"] <= 1.0


def test_hard_and_soft_space_differ_when_the_field_is_contested(arrival_frame):
    """The gap between thresholded and probability-weighted space is a calibration effect."""
    m = space_metrics(PhysicalControl(), arrival_frame)
    assert m["hard_minus_soft_m2"] != 0.0


def test_off_ball_run_value_responds_to_movement(arrival_frame, surface):
    still = off_ball_run_value(PhysicalControl(), arrival_frame, player_index=8,
                               displacement=np.array([0.0, 0.0]), surface=surface)
    moved = off_ball_run_value(PhysicalControl(), arrival_frame, player_index=8,
                               displacement=np.array([15.0, 5.0]), surface=surface)
    assert still["delta_soft_space_m2"] == pytest.approx(0.0, abs=1e-9)
    assert abs(moved["delta_soft_space_m2"]) > 1.0


def test_pass_value_comparison_and_sensitivity(sim_corpus, surface):
    _frames, df = sim_corpus
    rng = np.random.default_rng(0)
    work = df.assign(
        p_raw=np.clip(rng.beta(5, 2, len(df)), 0.01, 0.99),
    )
    work = work.assign(p_cal=np.clip(work["p_raw"] * 0.8 + 0.08, 0.01, 0.99))
    tab = pass_value_comparison(work, surface, control_cols={"raw": "p_raw", "isotonic": "p_cal"})
    assert set(tab["variant"]) == {"raw", "isotonic"}
    assert (tab.loc[tab["variant"] == "isotonic", "mean_abs_diff_vs_reference"] == 0).all()

    sens = downstream_sensitivity(work, raw_col="p_raw", calibrated_col="p_cal", surface=surface)
    assert (sens["frac_decisions_changed"] >= 0).all()
    assert (sens["frac_decisions_changed"] <= 1).all()
    assert sens["value_surface_provenance"].iloc[0] == surface.provenance


# --- figures ----------------------------------------------------------------
def test_every_figure_renders(sim_corpus, arrival_frame):
    from pcc.evaluation import net_benefit, spatial_calibration_map
    from pcc.viz import (
        calibration_gap_by_group, control_field, decision_curve, reliability_diagram,
        spatial_gap_map,
    )
    import pandas as pd

    _frames, df = sim_corpus
    rng = np.random.default_rng(1)
    y = df["y_control"].to_numpy()
    p = np.clip(rng.beta(5, 2, len(y)), 0.01, 0.99)

    assert reliability_diagram(y, {"a": p, "b": np.clip(p * 0.9, 0.01, 0.99)}) is not None
    assert control_field(PhysicalControl(), arrival_frame, n_x=20, n_y=14) is not None
    cells = spatial_calibration_map(df.assign(p=p), prob_col="p", outcome_col="y_control", min_n=5)
    assert spatial_gap_map(cells) is not None
    assert decision_curve(net_benefit(y, p)) is not None
    forest = pd.DataFrame({"level": ["a", "b"], "corp_mcb": [0.01, 0.02],
                           "mcb_lo": [0.0, 0.01], "mcb_hi": [0.02, 0.03], "suppressed": [False, False]})
    assert calibration_gap_by_group(forest, group_col="level") is not None


def test_spatial_map_with_no_sufficient_cells_still_renders(sim_corpus):
    from pcc.evaluation import spatial_calibration_map
    from pcc.viz import spatial_gap_map

    _frames, df = sim_corpus
    cells = spatial_calibration_map(df.assign(p=0.5), prob_col="p", outcome_col="y_control",
                                    min_n=10**9)
    fig = spatial_gap_map(cells)
    assert "no cell met" in fig.axes[0].get_title()


# --- scripts ----------------------------------------------------------------
def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        cwd=REPO / "scripts", capture_output=True, text=True, timeout=900,
    )


@pytest.mark.slow
def test_environment_and_availability_scripts_run():
    for script in ("00_environment_report.py", "01_check_data_availability.py"):
        res = _run(script)
        assert res.returncode == 0, res.stderr[-2000:]


@pytest.mark.slow
def test_instrument_validation_script_passes_every_check():
    """If this fails, no result from this repository is trustworthy."""
    res = _run("09_validate_instrument.py")
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    assert "checks passed" in res.stdout
    assert "[FAIL]" not in res.stdout
