"""The canonical data contract.

Every provider loader must produce (a) a table of *ball arrivals* conforming to
:data:`ARRIVAL_COLUMNS` and (b) for each arrival, an :class:`ArrivalFrame`
holding the positional state used by the control models. Keeping the contract
explicit is what makes the optical-versus-broadcast comparison (RQ4) a
controlled one: the two sources differ only in the quality of the fields, not
in their meaning.

Terminology follows the proposal's conceptual framework (Section 7). In
particular ``y_control`` is the *realised* outcome at the control horizon and
is the estimand's target; it is deliberately distinct from ``pass_completed``,
which is a provider event label about the pass, not about control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Column contract for the arrival-level analysis table.
#: ``(name, dtype, required, description)``
ARRIVAL_COLUMNS: list[tuple[str, str, bool, str]] = [
    # --- identity and grouping (used for leakage-safe splitting) -------------
    ("arrival_id", "string", True, "Unique id for the ball arrival."),
    ("match_id", "string", True, "Match identifier; the primary clustering unit."),
    ("competition", "string", True, "Competition/season label; cross-competition split unit."),
    ("possession_id", "string", True, "Possession-sequence id; nested clustering unit."),
    ("period", "int16", True, "Match period (1, 2, 3, 4)."),
    ("team_a", "string", True, "Team whose control probability is modelled (the passing team)."),
    ("team_b", "string", True, "Opposing team."),
    # --- when and where -----------------------------------------------------
    ("t_release", "float64", True, "Time of ball release, seconds from period start."),
    ("t_arrival", "float64", True, "Time the ball reaches the destination, seconds from period start."),
    ("flight_time", "float64", True, "t_arrival - t_release, seconds."),
    ("flight_time_imputed", "bool", True, "True if flight time came from a speed model, not ball tracking."),
    ("origin_x", "float64", True, "Release x in the canonical frame (m)."),
    ("origin_y", "float64", True, "Release y in the canonical frame (m)."),
    ("dest_x", "float64", True, "Arrival x in the canonical frame (m)."),
    ("dest_y", "float64", True, "Arrival y in the canonical frame (m)."),
    # --- outcome ------------------------------------------------------------
    ("y_control", "int8", True, "1 if team A is in control at t_arrival + horizon, else 0."),
    ("y_first_touch", "int8", False, "1 if a team-A player makes the first touch after arrival."),
    ("y_retained_3s", "int8", False, "1 if team A still has possession 3 s after arrival."),
    ("control_horizon", "float64", True, "Horizon h (s) used to define y_control."),
    ("outcome_censored", "bool", True, "True if the horizon window was cut short (whistle, period end)."),
    # --- arrival typology (defines the estimation subpopulations) -----------
    ("arrival_type", "string", True, "open_pass | deflection | clearance | second_ball | set_piece | cross"),
    ("is_endogenous", "bool", True, "True if the destination was chosen by a player (selection-prone)."),
    ("pass_length", "float64", True, "Straight-line release-to-arrival distance (m)."),
    ("pass_height", "string", False, "ground | low | high | unknown."),
    ("ball_speed", "float64", False, "Mean ball speed over the flight (m/s)."),
    # --- context ------------------------------------------------------------
    ("score_diff", "int16", False, "Team A goals minus team B goals at t_release."),
    ("minute", "float64", False, "Match minute at t_release."),
    ("n_players_a", "int16", True, "Team A players on the pitch and tracked."),
    ("n_players_b", "int16", True, "Team B players on the pitch and tracked."),
    ("pressure_index", "float64", False, "Opponents within 5 m of the release point."),
    ("set_piece", "bool", True, "True if the arrival follows a dead-ball restart."),
    # --- provenance and quality --------------------------------------------
    ("tracking_source", "string", True, "optical | broadcast | simulated."),
    ("provider", "string", True, "Data provider label."),
    ("frame_completeness", "float64", True, "Fraction of the 22 players observed at t_arrival."),
    ("sync_offset", "float64", False, "Estimated event-to-tracking offset applied (s)."),
    ("quality_flag", "string", True, "ok | interpolated | low_completeness | suspect_sync."),
]

REQUIRED_ARRIVAL_COLUMNS = [c for c, _, req, _ in ARRIVAL_COLUMNS if req]

#: Arrival types whose destination is *not* selected by the team in possession.
#: These form the quasi-exogenous subsample central to the selection-bias
#: strategy (proposal Section 14).
EXOGENOUS_ARRIVAL_TYPES = ("deflection", "clearance", "second_ball")


@dataclass
class ArrivalFrame:
    """Positional state of both teams at the moment a ball arrival is evaluated.

    Attributes
    ----------
    att_xy, att_v
        ``(n_a, 2)`` positions and velocities of team A (the team whose control
        probability is being modelled) at ``t_release``, in the canonical frame.
    def_xy, def_v
        ``(n_b, 2)`` the same for team B.
    target
        ``(2,)`` destination of the ball.
    flight_time
        Seconds from ``t_release`` until the ball reaches ``target``.
    att_is_gk, def_is_gk
        Boolean masks; goalkeepers get a different ball-control rate.
    att_vmax, def_vmax
        Optional per-player sprint-speed estimates. ``NaN`` falls back to the
        squad-level assumption in :class:`~pcc.kinematics.LocomotionParams`.
    att_observed, def_observed
        Boolean masks marking players actually observed (as opposed to
        interpolated across an occlusion). Models may ignore these; the
        evaluation uses them to build the low-completeness stratum.
    meta
        Free-form provenance carried through for diagnostics.

    Note
    ----
    The state is taken at ``t_release``, not at ``t_arrival``. Using the state
    at arrival would leak the outcome: defenders converge on the ball *because*
    it is arriving, so a model conditioned on arrival-time positions is partly
    reading the answer. All models in this study therefore forecast forward
    over the flight, which is the operationally meaningful question too - a
    passer decides at release.
    """

    att_xy: np.ndarray
    att_v: np.ndarray
    def_xy: np.ndarray
    def_v: np.ndarray
    target: np.ndarray
    flight_time: float
    att_is_gk: np.ndarray | None = None
    def_is_gk: np.ndarray | None = None
    att_vmax: np.ndarray | None = None
    def_vmax: np.ndarray | None = None
    att_observed: np.ndarray | None = None
    def_observed: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.att_xy = np.atleast_2d(np.asarray(self.att_xy, dtype=float))
        self.att_v = np.atleast_2d(np.asarray(self.att_v, dtype=float))
        self.def_xy = np.atleast_2d(np.asarray(self.def_xy, dtype=float))
        self.def_v = np.atleast_2d(np.asarray(self.def_v, dtype=float))
        self.target = np.asarray(self.target, dtype=float).reshape(-1)

        if self.att_xy.shape != self.att_v.shape:
            raise ValueError("att_xy and att_v must have the same shape")
        if self.def_xy.shape != self.def_v.shape:
            raise ValueError("def_xy and def_v must have the same shape")
        if self.target.shape != (2,):
            raise ValueError("target must be a 2-vector")
        if not np.isfinite(self.flight_time) or self.flight_time < 0:
            raise ValueError(f"flight_time must be finite and non-negative, got {self.flight_time}")

        self.att_is_gk = self._as_mask(self.att_is_gk, self.n_att)
        self.def_is_gk = self._as_mask(self.def_is_gk, self.n_def)
        self.att_observed = self._as_mask(self.att_observed, self.n_att, default=True)
        self.def_observed = self._as_mask(self.def_observed, self.n_def, default=True)

    @staticmethod
    def _as_mask(mask: np.ndarray | None, n: int, default: bool = False) -> np.ndarray:
        if mask is None:
            return np.full(n, default, dtype=bool)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != n:
            raise ValueError(f"mask length {mask.size} does not match {n} players")
        return mask

    @property
    def n_att(self) -> int:
        return int(self.att_xy.shape[0])

    @property
    def n_def(self) -> int:
        return int(self.def_xy.shape[0])

    @property
    def completeness(self) -> float:
        """Fraction of a nominal 22-player frame that is actually observed."""
        observed = int(self.att_observed.sum() + self.def_observed.sum())
        return observed / 22.0


def validate_arrivals(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Check an arrivals table against :data:`ARRIVAL_COLUMNS`.

    Raises on missing required columns. Reports, rather than silently fixes,
    rows that violate the internal consistency conditions - a silent fix here
    would hide exactly the preprocessing errors the study needs to quantify.
    """
    missing = [c for c in REQUIRED_ARRIVAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"arrivals table is missing required columns: {missing}")

    problems: list[str] = []
    if not df["arrival_id"].is_unique:
        problems.append("arrival_id is not unique")
    if (df["flight_time"] < 0).any():
        problems.append("negative flight_time")
    bad_ft = (df["t_arrival"] - df["t_release"] - df["flight_time"]).abs() > 1e-6
    if bad_ft.any():
        problems.append(f"{int(bad_ft.sum())} rows where t_arrival - t_release != flight_time")
    if not df["y_control"].isin([0, 1]).all():
        problems.append("y_control contains values other than 0/1")
    if (df["frame_completeness"] > 1.0).any() or (df["frame_completeness"] < 0).any():
        problems.append("frame_completeness outside [0, 1]")

    if problems:
        message = "arrivals table failed validation:\n  - " + "\n  - ".join(problems)
        if strict:
            raise ValueError(message)
        import warnings

        warnings.warn(message, stacklevel=2)
    return df
