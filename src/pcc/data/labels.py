"""Operational definitions of the outcome ``Y``.

Everything in this study rests on one choice: *what counts as team A having
controlled the ball?* The pitch-control literature leaves this implicit, which
is part of the measurement-validity problem the study identifies. Four
candidate definitions are implemented; they are not interchangeable, and the
proposal recommends ``controlled_at_horizon`` as primary with the others as
pre-registered robustness checks.

======================  =====================================================
Definition              What it means and what it assumes
======================  =====================================================
``first_touch``         The first player to touch the ball after arrival is a
                        team-A player. Closest to the literal reading of a
                        pitch-control field. Assumes touch attribution is
                        reliable, which for contested aerial balls it is not,
                        and counts a touch that immediately concedes
                        possession as control - which no coach would.
``controlled_at_horizon``  Team A is in possession ``h`` seconds after arrival.
                        Primary. Requires a horizon; ``h = 1.0 s`` is the
                        default because it is long enough to exclude a
                        deflection off a shin and short enough that the
                        subsequent action has not yet dominated the outcome.
``retained_sequence``   Team A completes at least one further controlled action
                        before losing the ball. The most football-meaningful and
                        the furthest from what a pitch-control field computes;
                        it mixes control with the quality of the next decision.
``possession_at_5s``    Team A holds the ball 5 s after arrival. A deliberately
                        long horizon included to show how quickly the construct
                        drifts away from "control" toward "sustained
                        possession".
======================  =====================================================

The sensitivity of every headline result to this choice is reported: if the
conclusion about calibration flips between ``h = 0.5`` and ``h = 2.0``, then
the claim "pitch control is (mis)calibrated" is under-specified and the paper
must say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONTROL_DEFINITIONS = ("first_touch", "controlled_at_horizon", "retained_sequence", "possession_at_5s")
DEFAULT_HORIZON = 1.0


@dataclass(frozen=True)
class LabelConfig:
    definition: str = "controlled_at_horizon"
    horizon: float = DEFAULT_HORIZON
    #: Minimum seconds of uninterrupted possession to count as "controlled".
    min_possession_seconds: float = 0.4
    #: Treat a stoppage inside the horizon window as censored rather than as a
    #: loss of control. Silently coding whistles as failures would bias every
    #: model downward in exactly the zones where fouls concentrate.
    censor_on_stoppage: bool = True

    def __post_init__(self) -> None:
        if self.definition not in CONTROL_DEFINITIONS:
            raise ValueError(f"definition must be one of {CONTROL_DEFINITIONS}")
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative")


def label_from_possession_track(
    possession_team: pd.Series,
    times: np.ndarray,
    t_arrival: float,
    team_a: str,
    config: LabelConfig | None = None,
    *,
    stoppage: pd.Series | None = None,
) -> dict[str, object]:
    """Label one arrival from a per-frame possession-team track.

    Parameters
    ----------
    possession_team
        Per-frame team in possession (or ``NA`` when the ball is loose).
    times
        Frame timestamps aligned with ``possession_team``.
    t_arrival
        Arrival time of the ball.
    team_a
        The team whose control probability is being modelled.
    stoppage
        Optional per-frame boolean marking dead-ball frames.

    Returns
    -------
    dict
        ``{"y": int, "censored": bool, "definition": str, "horizon": float}``.

    Notes
    -----
    Providers differ in how they define "in possession", and some do not supply
    it at all, in which case it must be derived from touch events. Whichever
    route is used must be recorded per match: a corpus whose outcome variable
    is defined differently in different matches has an outcome-definition
    confound that no amount of modelling will remove.
    """
    cfg = config or LabelConfig()
    times = np.asarray(times, dtype=float)
    order = np.argsort(times)
    times = times[order]
    poss = possession_team.to_numpy()[order]
    stop = None if stoppage is None else np.asarray(stoppage)[order]

    window = (times >= t_arrival) & (times <= t_arrival + cfg.horizon)
    if not window.any():
        return {"y": 0, "censored": True, "definition": cfg.definition, "horizon": cfg.horizon}

    if cfg.censor_on_stoppage and stop is not None and stop[window].any():
        return {"y": 0, "censored": True, "definition": cfg.definition, "horizon": cfg.horizon}

    if cfg.definition == "first_touch":
        after = poss[times >= t_arrival]
        first = next((t for t in after if pd.notna(t)), None)
        y = int(first == team_a)
    elif cfg.definition in ("controlled_at_horizon", "possession_at_5s"):
        h = 5.0 if cfg.definition == "possession_at_5s" else cfg.horizon
        idx = np.searchsorted(times, t_arrival + h)
        if idx >= times.size:
            return {"y": 0, "censored": True, "definition": cfg.definition, "horizon": h}
        y = int(poss[idx] == team_a)
    elif cfg.definition == "retained_sequence":
        seg = poss[window]
        held = np.sum(seg == team_a) / max(seg.size, 1) * cfg.horizon
        y = int(held >= cfg.min_possession_seconds and seg[-1] == team_a)
    else:  # pragma: no cover - guarded by LabelConfig
        raise ValueError(cfg.definition)

    return {"y": int(y), "censored": False, "definition": cfg.definition,
            "horizon": 5.0 if cfg.definition == "possession_at_5s" else cfg.horizon}


def classify_arrival(event_type: str, next_event_type: str | None = None, *, aerial: bool = False) -> tuple[str, bool]:
    """Map provider event labels onto the study's arrival typology.

    Returns ``(arrival_type, is_endogenous)``. The ``is_endogenous`` flag is the
    one that matters for the selection analysis: it marks arrivals whose
    destination was chosen by a player who may have known something the state
    vector does not.

    The mapping below is a *template*: every provider spells these labels
    differently and the concordance must be re-derived and documented per
    provider rather than assumed. Getting this wrong silently contaminates the
    quasi-exogenous subsample, which is the study's strongest identification
    argument, so the concordance is treated as a first-class artefact
    (``docs/data_sources.md``) and spot-checked against video where possible.
    """
    e = (event_type or "").strip().lower()
    exogenous = {
        "clearance": "clearance",
        "deflection": "deflection",
        "block": "deflection",
        "blocked_pass": "deflection",
        "rebound": "second_ball",
        "loose_ball": "second_ball",
        "duel": "second_ball",
        "aerial": "second_ball",
        "goalkeeper_punch": "clearance",
    }
    if e in exogenous:
        return exogenous[e], False
    if e in {"corner", "free_kick", "throw_in", "goal_kick", "kick_off", "set_piece"}:
        return "set_piece", True
    if e == "cross" or (e == "pass" and aerial):
        return "cross", True
    return "open_pass", True


def horizon_sensitivity_grid(base: float = DEFAULT_HORIZON) -> list[float]:
    """Pre-registered horizons for the sensitivity analysis (Section 15)."""
    return sorted({0.25, 0.5, base, 1.5, 2.0, 3.0})
