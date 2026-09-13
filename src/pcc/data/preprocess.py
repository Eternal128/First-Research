"""The preprocessing pipeline, written so its uncertainty can be reported.

Preprocessing is usually the invisible part of a tracking-data study. Here it
cannot be, for a specific reason: every quantity a pitch-control model consumes
- velocity, arrival time, the destination itself - is a *processed* quantity,
and the processing choices plausibly move the calibration statistics by more
than the difference between the models being compared. The pipeline therefore

* records a provenance entry for each step;
* propagates a per-arrival quality flag rather than dropping rows silently;
* exposes each choice as a parameter so the ablation study can sweep it;
* estimates preprocessing uncertainty by re-running the pipeline under
  perturbed settings and reporting the spread of the final statistic
  (:func:`preprocessing_uncertainty`).

The last point is the honest alternative to pretending the pipeline is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from pcc.geometry import Pitch, to_canonical
from pcc.kinematics import cap_speed, derivative, gap_mask, smooth_positions


@dataclass
class PreprocessConfig:
    """Every preprocessing choice, in one auditable place."""

    fps: float = 25.0
    smoothing_window_s: float = 0.4
    smoothing_polyorder: int = 2
    speed_cap: float = 12.0
    max_occlusion_gap_s: float = 0.5
    min_frame_completeness: float = 0.75
    max_sync_offset_s: float = 0.5
    min_pass_length_m: float = 3.0
    max_pass_length_m: float = 80.0
    min_flight_time_s: float = 0.1
    max_flight_time_s: float = 5.0
    exclude_set_pieces: bool = True
    source_extent: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    pitch: Pitch = field(default_factory=Pitch)

    def perturbed(self, **kwargs) -> "PreprocessConfig":
        from dataclasses import replace

        return replace(self, **kwargs)


@dataclass
class Provenance:
    """An audit trail of what the pipeline did and how much it dropped."""

    steps: list[dict] = field(default_factory=list)

    def record(self, step: str, **details) -> None:
        self.steps.append({"step": step, **details})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def total_dropped(self) -> int:
        return int(sum(s.get("dropped", 0) for s in self.steps))


# ---------------------------------------------------------------------------
# Tracking-side preprocessing
# ---------------------------------------------------------------------------
def prepare_tracking(
    tracking: pd.DataFrame,
    config: PreprocessConfig,
    *,
    player_col: str = "player_id",
    time_col: str = "timestamp",
    x_col: str = "x",
    y_col: str = "y",
    attacking_left_to_right: Callable[[pd.DataFrame], bool] | None = None,
    provenance: Provenance | None = None,
) -> pd.DataFrame:
    """Normalise coordinates, smooth tracks, and derive velocity per player.

    Returns a long frame with ``x, y, vx, vy, speed, occluded`` in the canonical
    metric frame. Velocity is a *derived* quantity throughout; the columns are
    named to make that impossible to forget.
    """
    prov = provenance or Provenance()
    out = []
    n_capped = 0
    for pid, g in tracking.sort_values(time_col).groupby(player_col, sort=False):
        xy = g[[x_col, y_col]].to_numpy(dtype=float)
        ltr = True if attacking_left_to_right is None else bool(attacking_left_to_right(g))
        xy = to_canonical(
            xy, source_extent=config.source_extent, pitch=config.pitch,
            attacking_left_to_right=ltr,
        )
        occluded = gap_mask(xy, fps=config.fps, max_gap_seconds=config.max_occlusion_gap_s)
        smoothed = smooth_positions(
            xy, fps=config.fps,
            window_seconds=config.smoothing_window_s, polyorder=config.smoothing_polyorder,
        )
        vel = derivative(
            xy, fps=config.fps,
            window_seconds=config.smoothing_window_s, polyorder=config.smoothing_polyorder,
        )
        capped = cap_speed(vel, v_cap=config.speed_cap)
        n_capped += int(np.sum(np.linalg.norm(vel, axis=1) > config.speed_cap))

        out.append(
            g.assign(
                x=smoothed[:, 0], y=smoothed[:, 1],
                vx=capped[:, 0], vy=capped[:, 1],
                speed=np.linalg.norm(capped, axis=1),
                occluded=occluded,
            )
        )
    result = pd.concat(out, ignore_index=True) if out else tracking.iloc[0:0].copy()
    prov.record(
        "prepare_tracking",
        n_players=int(tracking[player_col].nunique()),
        n_frames=int(len(tracking)),
        frames_capped=n_capped,
        smoothing_window_s=config.smoothing_window_s,
        fps=config.fps,
    )
    return result


def estimate_sync_offset(
    event_times: np.ndarray,
    ball_speed_series: pd.Series,
    frame_times: np.ndarray,
    *,
    search_s: float = 1.0,
    step_s: float = 0.04,
) -> float:
    """Estimate a constant event-to-tracking clock offset by cross-correlation.

    Kicks produce a sharp acceleration of the ball. Aligning event timestamps
    with peaks in the ball-speed derivative gives a per-match offset. This is
    necessary because event and tracking streams are frequently logged against
    different clocks, and a 0.3 s offset is enough to move a defender several
    metres - which would be charged to the *model* as miscalibration when it is
    really a synchronisation error.

    Returns the offset in seconds to add to event times. Residual
    synchronisation error after correction is *not* zero, and the per-match
    residual is retained as a quality covariate so that a subgroup analysis by
    sync quality can rule this explanation in or out.
    """
    frame_times = np.asarray(frame_times, dtype=float)
    accel = np.abs(np.gradient(np.asarray(ball_speed_series, dtype=float)))
    accel = np.nan_to_num(accel)
    if accel.size == 0 or len(event_times) == 0:
        return 0.0

    offsets = np.arange(-search_s, search_s + step_s, step_s)
    scores = []
    for off in offsets:
        idx = np.searchsorted(frame_times, np.asarray(event_times, dtype=float) + off)
        idx = np.clip(idx, 0, accel.size - 1)
        scores.append(float(np.mean(accel[idx])))
    return float(offsets[int(np.argmax(scores))])


# ---------------------------------------------------------------------------
# Event-side filtering
# ---------------------------------------------------------------------------
def filter_arrivals(
    arrivals: pd.DataFrame, config: PreprocessConfig, *, provenance: Provenance | None = None
) -> pd.DataFrame:
    """Apply the pre-registered inclusion criteria, recording every exclusion.

    Each filter has a stated reason, because an unstated exclusion is an
    uncontrolled researcher degree of freedom:

    * **duplicate ids** - provider artefacts; keeping both double-counts.
    * **very short passes** - under ~3 m the "arrival" is barely distinguishable
      from the release and the control question is degenerate.
    * **very long passes / long flights** - the state at release is a poor
      predictor several seconds later for reasons unrelated to control, and
      these arrivals would dominate the miscalibration statistics.
    * **low frame completeness** - a control model missing six players is not
      being tested on the same task. These are *retained as a separate
      stratum*, not deleted, because their existence is the point of RQ4.
    * **set pieces** - the defensive organisation is qualitatively different
      (a wall, a zonal block) and every physics model's assumptions fail;
      analysed separately rather than mixed in.
    * **censored outcomes** - stoppages inside the horizon window.
    """
    prov = provenance or Provenance()
    df = arrivals.copy()
    n0 = len(df)

    dup = df["arrival_id"].duplicated()
    df = df[~dup]
    prov.record("drop_duplicate_arrival_id", dropped=int(dup.sum()), remaining=len(df))

    # Predicates are callables, not pre-computed masks. A list of masks built
    # from the original frame would be evaluated once and then applied to the
    # progressively filtered frame, whose index no longer matches; pandas would
    # reindex, and the recorded exclusion counts - the audit trail this function
    # exists to produce - would be wrong.
    predicates = [
        ("pass_too_short", lambda d: d["pass_length"] < config.min_pass_length_m,
         "degenerate control question"),
        ("pass_too_long", lambda d: d["pass_length"] > config.max_pass_length_m,
         "outside the modelled regime"),
        ("flight_too_short", lambda d: d["flight_time"] < config.min_flight_time_s,
         "sub-frame flight time"),
        ("flight_too_long", lambda d: d["flight_time"] > config.max_flight_time_s,
         "state at release uninformative"),
        ("censored", lambda d: d["outcome_censored"].astype(bool),
         "outcome window interrupted"),
    ]
    for name, predicate, reason in predicates:
        mask = predicate(df).to_numpy()
        n_drop = int(mask.sum())
        df = df[~mask]
        prov.record(f"drop_{name}", dropped=n_drop, remaining=len(df), reason=reason)

    if config.exclude_set_pieces:
        mask = df["set_piece"].astype(bool).to_numpy()
        prov.record("separate_set_pieces", dropped=int(mask.sum()), remaining=int((~mask).sum()),
                    reason="analysed as a separate stratum, not discarded")
        df = df[~mask]

    low = (df["frame_completeness"] < config.min_frame_completeness).to_numpy()
    df = df.assign(
        low_completeness=low,
        quality_flag=np.where(low, "low_completeness", df["quality_flag"].fillna("ok")),
    )
    prov.record("flag_low_completeness", flagged=int(low.sum()), remaining=len(df),
                reason="retained as a stratum for RQ4, not dropped")

    if "sync_offset" in df.columns:
        suspect = (df["sync_offset"].abs() > config.max_sync_offset_s).to_numpy()
        df = df.assign(quality_flag=np.where(suspect, "suspect_sync", df["quality_flag"]))
        prov.record("flag_suspect_sync", flagged=int(suspect.sum()), remaining=len(df))

    prov.record("summary", n_in=n0, n_out=len(df), retention=len(df) / max(n0, 1))
    return df.reset_index(drop=True)


def preprocessing_uncertainty(
    run_pipeline: Callable[[PreprocessConfig], float],
    base_config: PreprocessConfig,
    *,
    perturbations: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Re-run a statistic under perturbed preprocessing and report the spread.

    This is the study's answer to "how much of your result is preprocessing?"
    The default perturbation grid covers the smoothing bandwidth, the occlusion
    tolerance and the completeness threshold - the three choices most likely to
    matter. If the range of the headline calibration statistic across this grid
    is comparable to the difference between models, then the model comparison
    is not identified by the data and the paper must report it that way rather
    than picking the configuration that separates them.
    """
    if perturbations is None:
        perturbations = {
            "smoothing_window_s": [0.2, 0.4, 0.8],
            "max_occlusion_gap_s": [0.25, 0.5, 1.0],
            "min_frame_completeness": [0.5, 0.75, 0.9],
        }
    rows = [{"parameter": "baseline", "value": None, "statistic": float(run_pipeline(base_config))}]
    for param, values in perturbations.items():
        for v in values:
            cfg = base_config.perturbed(**{param: v})
            try:
                stat = float(run_pipeline(cfg))
            except Exception as exc:  # pragma: no cover - reported, not swallowed
                stat = float("nan")
                rows.append({"parameter": param, "value": v, "statistic": stat, "error": str(exc)})
                continue
            rows.append({"parameter": param, "value": v, "statistic": stat})
    out = pd.DataFrame(rows)
    base = out.loc[out["parameter"] == "baseline", "statistic"].iloc[0]
    out["delta_vs_baseline"] = out["statistic"] - base
    return out
