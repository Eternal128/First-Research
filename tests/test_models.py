"""The control models must satisfy the invariants a probability field implies."""

from __future__ import annotations

import numpy as np
import pytest

from pcc.data.schema import ArrivalFrame
from pcc.kinematics import (
    LocomotionParams, arrival_probability, time_to_point_bounded_accel,
    time_to_point_constant_speed,
)
from pcc.models import (
    GBMControl, LogisticControl, MarginalBaseline, PhysicalControl,
    ReachabilitySigmoid, VoronoiControl, build_model, default_model_suite,
)

ALL_GEOMETRIC = [VoronoiControl, PhysicalControl, ReachabilitySigmoid]


@pytest.mark.parametrize("cls", ALL_GEOMETRIC)
def test_output_is_in_the_unit_interval(cls, arrival_frame):
    out = cls().control(arrival_frame)
    assert out.shape == (1,)
    assert 0.0 <= out[0] <= 1.0


@pytest.mark.parametrize("cls", ALL_GEOMETRIC)
def test_grid_evaluation_shape(cls, arrival_frame):
    grid = np.column_stack([np.linspace(-50, 50, 23), np.linspace(-30, 30, 23)])
    out = cls().control(arrival_frame, grid)
    assert out.shape == (23,)
    assert np.all((out >= 0) & (out <= 1))


@pytest.mark.parametrize("cls", ALL_GEOMETRIC)
def test_team_swap_gives_the_complementary_probability(cls, arrival_frame):
    """C_A + C_B = 1 must hold, or the output is not a distribution over two outcomes."""
    f = arrival_frame
    mirrored = ArrivalFrame(
        att_xy=f.def_xy, att_v=f.def_v, def_xy=f.att_xy, def_v=f.att_v,
        target=f.target, flight_time=f.flight_time,
        att_is_gk=f.def_is_gk, def_is_gk=f.att_is_gk, meta=f.meta,
    )
    a = float(cls().control(f)[0])
    b = float(cls().control(mirrored)[0])
    assert a + b == pytest.approx(1.0, abs=0.02)


def test_control_increases_when_the_attacker_gets_closer(arrival_frame):
    """A monotonicity the physics genuinely implies; a violation is a bug."""
    f = arrival_frame
    model = PhysicalControl()
    before = float(model.control(f)[0])
    moved = f.att_xy.copy()
    moved[5] = f.target + np.array([0.5, 0.0])
    after = float(model.control(
        ArrivalFrame(att_xy=moved, att_v=f.att_v, def_xy=f.def_xy, def_v=f.def_v,
                     target=f.target, flight_time=f.flight_time,
                     att_is_gk=f.att_is_gk, def_is_gk=f.def_is_gk, meta=f.meta)
    )[0])
    assert after >= before - 1e-9


def test_voronoi_is_hard_thresholded(arrival_frame):
    """M1 can only say (almost) 0 or 1 - which is why it is the sharpness reference."""
    grid = np.column_stack([np.linspace(-50, 50, 40), np.zeros(40)])
    out = VoronoiControl(epsilon=1e-3).control(arrival_frame, grid)
    assert np.all((out < 0.01) | (out > 0.99))


def test_physical_model_leaves_little_unassigned_mass(arrival_frame):
    """A field that does not sum to one across teams is not a probability."""
    unassigned = PhysicalControl().unassigned_mass(arrival_frame)
    assert float(unassigned[0]) < 0.02


def test_integration_step_converges(arrival_frame):
    """The Euler step must be small enough that the answer is about the model."""
    fine = float(PhysicalControl(dt=0.01).control(arrival_frame)[0])
    default = float(PhysicalControl(dt=0.04).control(arrival_frame)[0])
    assert default == pytest.approx(fine, abs=0.02)


def test_bounded_accel_is_never_faster_than_constant_speed():
    """Adding an acceleration limit cannot make a player arrive sooner."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(-40, 40, (30, 2))
    vel = rng.normal(0, 3, (30, 2))
    target = np.array([5.0, 5.0])
    params = LocomotionParams()
    const = time_to_point_constant_speed(pos, vel, target, params)
    bounded = time_to_point_bounded_accel(pos, vel, target, params)
    assert np.all(bounded >= const - 1e-9)


def test_arrival_probability_is_one_half_at_tau():
    params = LocomotionParams()
    assert float(arrival_probability(np.array([2.0]), 2.0, params)[0]) == pytest.approx(0.5)


def test_arrival_probability_is_monotone_in_time():
    params = LocomotionParams()
    ts = np.linspace(0, 6, 50)
    vals = np.array([float(arrival_probability(np.array([3.0]), t, params)[0]) for t in ts])
    assert np.all(np.diff(vals) >= -1e-12)


def test_fitted_models_require_fitting(arrival_frame):
    for model in (LogisticControl(), GBMControl(), MarginalBaseline()):
        with pytest.raises(RuntimeError):
            model.predict([arrival_frame])


def test_fitted_models_learn_and_predict(sim_corpus):
    frames, df = sim_corpus
    y = df["y_control"].to_numpy(dtype=int)
    for model in (LogisticControl(), GBMControl(), MarginalBaseline(), ReachabilitySigmoid()):
        model.fit(frames[:800], y[:800])
        p = model.predict(frames[800:1000])
        assert p.shape == (200,)
        assert np.all((p >= 0) & (p <= 1))


def test_marginal_baseline_returns_the_training_base_rate(sim_corpus):
    frames, df = sim_corpus
    y = df["y_control"].to_numpy(dtype=int)
    m = MarginalBaseline().fit(frames[:500], y[:500])
    assert m.predict(frames[500:600])[0] == pytest.approx(y[:500].mean())


def test_reachability_sigmoid_fit_improves_the_score(sim_corpus):
    from pcc.calibration import brier_score

    frames, df = sim_corpus
    y = df["y_control"].to_numpy(dtype=int)
    model = ReachabilitySigmoid(scale=10.0)
    before = brier_score(y[:800], model.predict(frames[:800]))
    model.fit(frames[:800], y[:800])
    after = brier_score(y[:800], model.predict(frames[:800]))
    assert after <= before


def test_registry_builds_every_named_model():
    for name in ("M0_marginal", "M1_voronoi", "M2_physical", "M2a_reach_sigmoid",
                 "M3_logistic", "M4_gbm"):
        assert build_model(name).name == name
    with pytest.raises(KeyError):
        build_model("not_a_model")


def test_default_suite_has_no_duplicate_names():
    names = [m.name for m in default_model_suite()]
    assert len(names) == len(set(names))


def test_frame_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        ArrivalFrame(att_xy=np.zeros((3, 2)), att_v=np.zeros((2, 2)),
                     def_xy=np.zeros((3, 2)), def_v=np.zeros((3, 2)),
                     target=np.zeros(2), flight_time=1.0)
    with pytest.raises(ValueError):
        ArrivalFrame(att_xy=np.zeros((3, 2)), att_v=np.zeros((3, 2)),
                     def_xy=np.zeros((3, 2)), def_v=np.zeros((3, 2)),
                     target=np.zeros(2), flight_time=-1.0)
