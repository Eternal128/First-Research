"""Feature extraction for the statistical baselines (Models 3-4).

Design rule: **every feature must be computable from exactly the information
the geometric models receive** - positions and velocities at release, the
destination, and the flight time. No event-stream annotations (pass type,
outcome-adjacent labels), no post-arrival information. Otherwise a win for the
statistical baseline would be uninformative: it would show that extra data
helps, not that the physics-based functional form is wrong.

The feature set is intentionally small and interpretable. A large automatically
generated feature bank would improve discrimination and obscure the mechanism,
and the study is about whether a number means what it claims, not about
squeezing out AUC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from pcc.data.schema import ArrivalFrame
from pcc.geometry import Pitch, distance_to_goal, distance_to_touchline
from pcc.kinematics import TTI_MODELS, LocomotionParams

FEATURE_NAMES: list[str] = [
    "tau_a_min",            # fastest team-A arrival time (s)
    "tau_b_min",            # fastest team-B arrival time (s)
    "tau_gap",              # tau_b_min - tau_a_min (s); the core discriminator
    "tau_a_second",         # second-fastest team-A arrival (s)
    "tau_b_second",         # second-fastest team-B arrival (s)
    "tau_a_slack",          # tau_a_min - flight_time; negative = A arrives early
    "tau_b_slack",          # tau_b_min - flight_time
    "dist_a_min",           # distance from destination to nearest team-A player (m)
    "dist_b_min",           # distance to nearest team-B player (m)
    "dist_gap",             # dist_b_min - dist_a_min (m)
    "n_a_within_5",         # team-A players within 5 m of the destination
    "n_b_within_5",
    "n_a_within_10",
    "n_b_within_10",
    "local_numerical_adv",  # n_a_within_10 - n_b_within_10
    "closing_speed_a",      # speed of nearest A player projected onto the destination (m/s)
    "closing_speed_b",
    "flight_time",          # s
    "pass_length",          # m
    "ball_speed",           # m/s
    "dest_x",               # canonical metres
    "dest_y",
    "abs_dest_y",
    "dist_to_goal",         # m
    "dist_to_touchline",    # m
    "origin_dist_to_goal",  # m
    "pass_forwardness",     # (dest_x - origin_x) / pass_length in [-1, 1]
    "gk_b_dist",            # distance from destination to team-B goalkeeper (m)
    "frame_completeness",   # fraction of 22 players observed
]


@dataclass(frozen=True)
class FeatureConfig:
    """Knobs that the ablation study (Section 17) toggles."""

    params: LocomotionParams = LocomotionParams()
    tti_model: str = "bounded_accel"
    pitch: Pitch = Pitch()
    use_velocity: bool = True
    use_acceleration: bool = False  # acceleration enters only via the TTI model
    use_reaction_time: bool = True
    nearest_distance_only: bool = False  # ablation: collapse to dist_a_min, dist_b_min


def _safe_sorted(values: np.ndarray, k: int, fill: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fill
    if values.size <= k:
        return float(np.max(values))
    return float(np.partition(values, k)[k])


def extract_features(frame: ArrivalFrame, config: FeatureConfig | None = None) -> dict[str, float]:
    """Engineered features for a single arrival."""
    cfg = config or FeatureConfig()
    params = cfg.params if cfg.use_reaction_time else cfg.params.replace(reaction_time=0.0)
    tti = TTI_MODELS[cfg.tti_model]

    att_v = frame.att_v if cfg.use_velocity else np.zeros_like(frame.att_v)
    def_v = frame.def_v if cfg.use_velocity else np.zeros_like(frame.def_v)

    target = frame.target
    tau_a = np.asarray(tti(frame.att_xy, att_v, target, params, v_max=frame.att_vmax), dtype=float)
    tau_b = np.asarray(tti(frame.def_xy, def_v, target, params, v_max=frame.def_vmax), dtype=float)

    d_a = np.linalg.norm(frame.att_xy - target[None, :], axis=1)
    d_b = np.linalg.norm(frame.def_xy - target[None, :], axis=1)

    origin = np.asarray(frame.meta.get("origin", target), dtype=float).reshape(2)
    pass_length = float(np.linalg.norm(target - origin))
    ft = float(max(frame.flight_time, 1e-6))

    def _closing(xy: np.ndarray, v: np.ndarray, d: np.ndarray) -> float:
        if xy.shape[0] == 0:
            return 0.0
        j = int(np.argmin(d))
        direction = target - xy[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return float(np.linalg.norm(v[j]))
        return float(np.dot(v[j], direction / norm))

    gk_b = frame.def_xy[frame.def_is_gk]
    gk_b_dist = float(np.linalg.norm(gk_b[0] - target)) if gk_b.shape[0] else float("nan")

    feats: dict[str, float] = {
        "tau_a_min": float(np.min(tau_a)) if tau_a.size else 99.0,
        "tau_b_min": float(np.min(tau_b)) if tau_b.size else 99.0,
        "tau_a_second": _safe_sorted(tau_a, 1, 99.0),
        "tau_b_second": _safe_sorted(tau_b, 1, 99.0),
        "dist_a_min": float(np.min(d_a)) if d_a.size else 99.0,
        "dist_b_min": float(np.min(d_b)) if d_b.size else 99.0,
        "n_a_within_5": float((d_a <= 5).sum()),
        "n_b_within_5": float((d_b <= 5).sum()),
        "n_a_within_10": float((d_a <= 10).sum()),
        "n_b_within_10": float((d_b <= 10).sum()),
        "closing_speed_a": _closing(frame.att_xy, att_v, d_a),
        "closing_speed_b": _closing(frame.def_xy, def_v, d_b),
        "flight_time": ft,
        "pass_length": pass_length,
        "ball_speed": pass_length / ft,
        "dest_x": float(target[0]),
        "dest_y": float(target[1]),
        "abs_dest_y": float(abs(target[1])),
        "dist_to_goal": float(distance_to_goal(target[None, :], pitch=cfg.pitch)[0]),
        "dist_to_touchline": float(distance_to_touchline(target[None, :], pitch=cfg.pitch)[0]),
        "origin_dist_to_goal": float(distance_to_goal(origin[None, :], pitch=cfg.pitch)[0]),
        "pass_forwardness": float((target[0] - origin[0]) / pass_length) if pass_length > 1e-6 else 0.0,
        "gk_b_dist": gk_b_dist,
        "frame_completeness": float(frame.completeness),
    }
    feats["tau_gap"] = feats["tau_b_min"] - feats["tau_a_min"]
    feats["tau_a_slack"] = feats["tau_a_min"] - ft
    feats["tau_b_slack"] = feats["tau_b_min"] - ft
    feats["dist_gap"] = feats["dist_b_min"] - feats["dist_a_min"]
    feats["local_numerical_adv"] = feats["n_a_within_10"] - feats["n_b_within_10"]

    if cfg.nearest_distance_only:
        keep = {"dist_a_min", "dist_b_min", "dist_gap", "flight_time"}
        feats = {k: (v if k in keep else 0.0) for k, v in feats.items()}

    return {k: feats[k] for k in FEATURE_NAMES}


def build_feature_matrix(
    frames: Sequence[ArrivalFrame], config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Feature matrix for a sequence of arrivals, columns ordered as ``FEATURE_NAMES``."""
    rows = [extract_features(f, config) for f in frames]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
