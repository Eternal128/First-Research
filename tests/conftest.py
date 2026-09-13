"""Shared fixtures. The whole suite runs on simulated data and needs no downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def sim_corpus():
    """A small simulated corpus: frames plus the arrivals table."""
    from pcc.data.synthetic import SimulationConfig, simulate_dataset

    cfg = SimulationConfig(n_matches=8, arrivals_per_match=150, random_state=7)
    return simulate_dataset(cfg)


@pytest.fixture(scope="session")
def calibrated_sample():
    """A perfectly calibrated (forecast, outcome) pair with match clustering."""
    rng = np.random.default_rng(11)
    match = np.repeat(np.arange(20), 400)
    z = rng.normal(0, 1.3, match.size) + rng.normal(0, 0.4, 20)[match]
    p = 1 / (1 + np.exp(-z))
    return p, rng.binomial(1, p), match


@pytest.fixture
def arrival_frame():
    from pcc.data.schema import ArrivalFrame

    rng = np.random.default_rng(3)
    return ArrivalFrame(
        att_xy=rng.uniform(-40, 40, (11, 2)), att_v=rng.normal(0, 2, (11, 2)),
        def_xy=rng.uniform(-40, 40, (11, 2)), def_v=rng.normal(0, 2, (11, 2)),
        target=np.array([12.0, -4.0]), flight_time=1.1,
        att_is_gk=[True] + [False] * 10, def_is_gk=[True] + [False] * 10,
        meta={"origin": np.array([-8.0, 3.0])},
    )
