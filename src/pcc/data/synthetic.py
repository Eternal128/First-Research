"""A simulator for exercising the pipeline. **Not a source of football findings.**

Purpose and limits
------------------
This module generates *fabricated* tracking-like states and ball arrivals from a
known data-generating process. It exists for three reasons, all methodological:

1. **End-to-end testing.** The whole pipeline - frames, models, splits,
   calibration, reweighting, bootstrap - runs without any provider data, so a
   reviewer or a new collaborator can reproduce the machinery on day one while
   data access is being arranged.
2. **Validating the instrument.** Because the true conditional probability is
   known, the tests can check that the calibration machinery reports near-zero
   miscalibration for the oracle forecast and detects a *deliberately injected*
   distortion. A calibration study whose measuring instrument has never been
   calibrated is not a serious study.
3. **Power analysis.** Sample sizes needed to detect a calibration slope of,
   say, 0.85 at a given number of matches can be estimated by simulation before
   committing to a data source (``scripts/09_power_analysis.py``).

The simulated football is crude - formations are Gaussian blobs, there is no
possession structure, no offside, no fatigue. **No number produced from this
module may appear in a results section as a claim about real football**, and
every artefact it writes is tagged ``tracking_source = "simulated"`` so that it
cannot be silently mixed with real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pcc.data.schema import ArrivalFrame
from pcc.geometry import Pitch
from pcc.kinematics import LocomotionParams

#: A 4-4-2-ish set of mean positions in the canonical frame (attacking +x),
#: expressed as fractions of half-length and half-width.
FORMATION = np.array(
    [
        [-0.92, 0.00],   # GK
        [-0.55, -0.55], [-0.60, -0.18], [-0.60, 0.18], [-0.55, 0.55],
        [-0.10, -0.60], [-0.15, -0.20], [-0.15, 0.20], [-0.10, 0.60],
        [0.30, -0.20], [0.30, 0.20],
    ]
)


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the fabricated data-generating process."""

    n_matches: int = 12
    arrivals_per_match: int = 400
    frac_exogenous: float = 0.18      # deflections/clearances/second balls
    formation_jitter: float = 6.0     # m
    speed_sd: float = 1.8             # m/s
    line_height_sd: float = 8.0       # m, match-level defensive-line variation
    control_horizon: float = 1.0      # s
    selection_temperature: float = 1.2  # lower = passers choose safer destinations
    n_pass_options: int = 12
    broadcast_fraction: float = 0.3   # share of matches with degraded tracking
    broadcast_occlusion: float = 0.25  # share of players unobserved in those matches
    broadcast_noise: float = 1.2      # m, positional noise in those matches
    # --- the injected truth --------------------------------------------------
    truth_intercept: float = 0.15
    truth_tau_gap: float = 1.30       # log-odds per second of reachability advantage
    truth_flight: float = -0.18       # longer flights favour the defence
    truth_zone_x: float = -0.012      # per metre upfield: control is harder near goal
    truth_congestion: float = -0.22   # per extra opponent within 10 m
    truth_match_sd: float = 0.25      # match-level random intercept
    random_state: int = 0


def _true_logodds(
    tau_gap: np.ndarray,
    flight: np.ndarray,
    dest_x: np.ndarray,
    n_def_close: np.ndarray,
    match_effect: float,
    cfg: SimulationConfig,
) -> np.ndarray:
    """The ground-truth conditional log-odds of control.

    Deliberately *not* the functional form of any model in
    :mod:`pcc.models`. The reachability gap enters linearly in log-odds, which
    the physics model approximates but does not reproduce, and two further terms
    (pitch position and local congestion) are invisible to a pure two-player
    reachability comparison. A physics-based model evaluated against this truth
    should therefore be miscalibrated *by construction*, and the test suite
    asserts that the machinery detects it.
    """
    return (
        cfg.truth_intercept
        + cfg.truth_tau_gap * tau_gap
        + cfg.truth_flight * flight
        + cfg.truth_zone_x * dest_x
        + cfg.truth_congestion * n_def_close
        + match_effect
    )


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))



def _sample_in_pitch(
    origin: np.ndarray, n: int, rng: np.random.Generator, pitch: Pitch,
    *, r_min: float = 5.0, r_max: float = 45.0, max_tries: int = 40,
) -> np.ndarray:
    """Draw ``n`` destinations in an annulus about ``origin``, inside the pitch."""
    out = np.empty((n, 2))
    filled = 0
    for _ in range(max_tries):
        need = n - filled
        r = rng.uniform(r_min, r_max, need * 3)
        th = rng.uniform(0, 2 * np.pi, need * 3)
        pts = origin[None, :] + r[:, None] * np.column_stack([np.cos(th), np.sin(th)])
        ok = pts[pitch.contains(pts, tol=-1.0)]
        take = min(need, ok.shape[0])
        if take:
            out[filled : filled + take] = ok[:take]
            filled += take
        if filled >= n:
            return out
    # Degenerate fallback (origin in a corner): fill the remainder with jitter
    # about the origin, still inside the pitch.
    while filled < n:
        out[filled] = pitch.clip((origin + rng.normal(0, 5.0, 2))[None, :])[0]
        filled += 1
    return out


def simulate_match(
    match_id: str,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    *,
    pitch: Pitch = Pitch(),
    params: LocomotionParams | None = None,
    tracking_source: str = "optical",
    competition: str = "SIM-A",
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """Simulate one match's worth of ball arrivals."""
    from pcc.kinematics import time_to_point_bounded_accel

    params = params or LocomotionParams()
    match_effect = float(rng.normal(0, cfg.truth_match_sd))
    line_shift = float(rng.normal(0, cfg.line_height_sd))

    frames: list[ArrivalFrame] = []
    rows: list[dict] = []

    n_exo = int(round(cfg.frac_exogenous * cfg.arrivals_per_match))
    exogenous_flags = np.zeros(cfg.arrivals_per_match, dtype=bool)
    exogenous_flags[:n_exo] = True
    rng.shuffle(exogenous_flags)

    for k in range(cfg.arrivals_per_match):
        # --- team shapes ----------------------------------------------------
        base_a = FORMATION * np.array([pitch.half_length, pitch.half_width])
        base_b = -base_a.copy()
        base_b[:, 0] += line_shift  # match-level defensive-line height

        att_xy = base_a + rng.normal(0, cfg.formation_jitter, base_a.shape)
        def_xy = base_b + rng.normal(0, cfg.formation_jitter, base_b.shape)
        att_xy = pitch.clip(att_xy)
        def_xy = pitch.clip(def_xy)
        att_v = rng.normal(0, cfg.speed_sd, att_xy.shape)
        def_v = rng.normal(0, cfg.speed_sd, def_xy.shape)

        att_gk = np.zeros(11, dtype=bool); att_gk[0] = True
        def_gk = np.zeros(11, dtype=bool); def_gk[0] = True

        # --- broadcast-style degradation ------------------------------------
        att_obs = np.ones(11, dtype=bool)
        def_obs = np.ones(11, dtype=bool)
        if tracking_source == "broadcast":
            att_xy = att_xy + rng.normal(0, cfg.broadcast_noise, att_xy.shape)
            def_xy = def_xy + rng.normal(0, cfg.broadcast_noise, def_xy.shape)
            att_obs = rng.uniform(size=11) > cfg.broadcast_occlusion
            def_obs = rng.uniform(size=11) > cfg.broadcast_occlusion
            att_obs[0] = True
            def_obs[0] = True

        # --- release point --------------------------------------------------
        passer = int(rng.integers(1, 11))
        origin = att_xy[passer]

        # --- candidate destinations and the passer's choice ------------------
        # Candidates are drawn by rejection inside the playing area rather than
        # clipped onto it: clipping would pile a spurious mass of destinations
        # onto the touchlines and goal lines and would distort both the
        # selection model and every spatial calibration statistic.
        cands = _sample_in_pitch(origin, cfg.n_pass_options, rng, pitch)
        n_opt = cands.shape[0]

        speed = float(rng.uniform(10, 22))
        flights = np.maximum(np.linalg.norm(cands - origin[None, :], axis=1) / speed, 0.05)

        tau_a = time_to_point_bounded_accel(att_xy, att_v, cands, params)   # (n_opt, 11)
        tau_b = time_to_point_bounded_accel(def_xy, def_v, cands, params)
        gap = np.nanmin(tau_b, axis=1) - np.nanmin(tau_a, axis=1)
        n_close = (np.linalg.norm(cands[:, None, :] - def_xy[None, :, :], axis=2) <= 10).sum(axis=1)

        if exogenous_flags[k]:
            # A deflection/clearance: destination is not chosen. Uniform over the
            # candidate set, which is the simulator's way of shutting the
            # selection channel - the analysis must be able to recover this.
            choice = int(rng.integers(0, n_opt))
            arrival_type = str(rng.choice(["deflection", "clearance", "second_ball"]))
            is_endogenous = False
        else:
            # The passer's *private* perception of control includes a term the
            # recorded state does not contain: this is the unobservable that
            # makes selection bias irreducible in real data, reproduced here so
            # that the sensitivity analysis has something to detect.
            private = rng.normal(0, 0.6, n_opt)
            perceived = _true_logodds(gap, flights, cands[:, 0], n_close, match_effect, cfg) + private
            progression = 0.02 * (cands[:, 0] - origin[0])
            utility = perceived + progression
            w = _sigmoid((utility - utility.max()) / cfg.selection_temperature)
            w = w / w.sum()
            choice = int(rng.choice(n_opt, p=w))
            arrival_type = "cross" if abs(cands[choice, 1]) > 0.6 * pitch.half_width else "open_pass"
            is_endogenous = True
            # The private term also shifts the *true* outcome probability, which
            # is exactly the Y-not-independent-of-D-given-X violation.
            match_effect_k = match_effect + 0.5 * private[choice]

        dest = cands[choice]
        flight = float(flights[choice])
        me = match_effect if not is_endogenous else match_effect_k

        p_true = float(
            _sigmoid(
                _true_logodds(
                    np.array([gap[choice]]), np.array([flight]), np.array([dest[0]]),
                    np.array([n_close[choice]]), me, cfg,
                )
            )[0]
        )
        y = int(rng.uniform() < p_true)

        t_release = float(k * (5400.0 / cfg.arrivals_per_match))
        frames.append(
            ArrivalFrame(
                att_xy=att_xy, att_v=att_v, def_xy=def_xy, def_v=def_v,
                target=dest, flight_time=flight,
                att_is_gk=att_gk, def_is_gk=def_gk,
                att_observed=att_obs, def_observed=def_obs,
                meta={"origin": origin, "match_id": match_id, "p_true": p_true},
            )
        )
        rows.append(
            {
                "arrival_id": f"{match_id}_{k:05d}",
                "match_id": match_id,
                "competition": competition,
                "possession_id": f"{match_id}_poss{k // 4:04d}",
                "period": 1 if t_release < 2700 else 2,
                "team_a": f"{match_id}_A",
                "team_b": f"{match_id}_B",
                "t_release": t_release,
                "t_arrival": t_release + flight,
                "flight_time": flight,
                "flight_time_imputed": False,
                "origin_x": float(origin[0]), "origin_y": float(origin[1]),
                "dest_x": float(dest[0]), "dest_y": float(dest[1]),
                "y_control": y,
                "y_first_touch": y,
                "control_horizon": cfg.control_horizon,
                "outcome_censored": False,
                "arrival_type": arrival_type,
                "is_endogenous": is_endogenous,
                "pass_length": float(np.linalg.norm(dest - origin)),
                "pass_height": "ground",
                "ball_speed": speed,
                "score_diff": 0,
                "minute": t_release / 60.0,
                "n_players_a": 11, "n_players_b": 11,
                "pressure_index": float(
                    (np.linalg.norm(def_xy - origin[None, :], axis=1) <= 5).sum()
                ),
                "set_piece": False,
                "tracking_source": "simulated",
                "provider": f"simulator/{tracking_source}",
                "frame_completeness": float((att_obs.sum() + def_obs.sum()) / 22.0),
                "sync_offset": 0.0,
                "quality_flag": "ok" if tracking_source == "optical" else "interpolated",
                "sim_tracking_mode": tracking_source,
                "p_true": p_true,
            }
        )

    return frames, pd.DataFrame(rows)


def simulate_dataset(
    cfg: SimulationConfig | None = None, *, pitch: Pitch = Pitch()
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """Simulate a multi-match corpus with a mix of tracking qualities.

    The returned table carries a ``p_true`` column - the oracle probability.
    Real data never has this; it is present so the tests can compare a model's
    forecast against the truth rather than only against the realised outcome.
    Any analysis script that reads ``p_true`` is by definition a validation
    script, not an empirical one, and is named accordingly.
    """
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.random_state)

    frames: list[ArrivalFrame] = []
    tables: list[pd.DataFrame] = []
    n_broadcast = int(round(cfg.broadcast_fraction * cfg.n_matches))
    modes = ["broadcast"] * n_broadcast + ["optical"] * (cfg.n_matches - n_broadcast)
    rng.shuffle(modes)

    for i, mode in enumerate(modes):
        mid = f"SIM{i:03d}"
        comp = "SIM-A" if i < cfg.n_matches // 2 else "SIM-B"
        f, t = simulate_match(mid, cfg, rng, pitch=pitch, tracking_source=mode, competition=comp)
        frames.extend(f)
        tables.append(t)

    table = pd.concat(tables, ignore_index=True)
    table["match_date"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        table["match_id"].str.replace("SIM", "").astype(int) * 7, unit="D"
    )
    return frames, table
