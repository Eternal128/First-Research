"""Provider-independent tracking container and arrival assembly.

This is the half of every loader that is *not* provider-specific. A provider
adapter's job is to produce a :class:`TrackingTable` and a list of
:class:`ArrivalSpec`; everything from there - attacking-direction
normalisation, state extraction at release, labelling, and the arrivals table -
happens here, identically for every provider.

That factoring is what makes the optical-versus-broadcast comparison (RQ4) a
controlled contrast rather than a comparison of two codebases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pcc.data.labels import LabelConfig, label_from_possession_track
from pcc.data.schema import ArrivalFrame
from pcc.geometry import Pitch
from pcc.kinematics import cap_speed, derivative, gap_mask


@dataclass
class TrackingTable:
    """Wide-format tracking for one match, in a fixed centred metric frame.

    Coordinates are metres with the origin at the centre spot, ``+y`` toward
    one touchline, and **no attacking-direction normalisation applied**. The
    per-arrival rotation into the canonical "team A attacks ``+x``" frame is
    done by :func:`assemble_arrivals` using :attr:`attack_sign`, because the
    correct rotation depends on which team is in possession and in which
    period.

    Attributes
    ----------
    time
        ``(T,)`` seconds from the start of the match's own clock.
    period
        ``(T,)`` period index per frame.
    xy, vel
        ``(T, P, 2)`` player positions and derived velocities.
    observed
        ``(T, P)`` False where the player is off the pitch or inside an
        occlusion gap longer than the tolerance. Derived velocity is not
        trustworthy there either.
    ball_xy
        ``(T, 2)``; ``NaN`` where the ball is not tracked.
    player_ids, teams, is_gk
        ``(P,)`` per-player metadata.
    attack_sign
        ``{(team, period): +1 | -1}``. ``+1`` means that team attacks ``+x`` in
        the stored frame during that period.
    fps
        Frame rate. Asserted against the timestamps by :meth:`validate`.
    """

    time: np.ndarray
    period: np.ndarray
    xy: np.ndarray
    vel: np.ndarray
    observed: np.ndarray
    ball_xy: np.ndarray
    player_ids: list[str]
    teams: np.ndarray
    is_gk: np.ndarray
    attack_sign: dict
    fps: float
    pitch: Pitch = field(default_factory=Pitch)
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.xy.shape[0])

    @property
    def n_players(self) -> int:
        return int(self.xy.shape[1])

    def frame_index(self, t: float, period: int | None = None) -> int:
        """Index of the tracking frame nearest to time ``t``."""
        if period is None:
            candidates = np.arange(self.n_frames)
        else:
            candidates = np.flatnonzero(self.period == period)
            if candidates.size == 0:
                candidates = np.arange(self.n_frames)
        j = int(np.argmin(np.abs(self.time[candidates] - t)))
        return int(candidates[j])

    def validate(self) -> None:
        """Check internal consistency. Raises rather than warns."""
        T, P = self.xy.shape[0], self.xy.shape[1]
        for name, arr, shape in (
            ("vel", self.vel, (T, P, 2)),
            ("observed", self.observed, (T, P)),
            ("ball_xy", self.ball_xy, (T, 2)),
            ("time", self.time, (T,)),
            ("period", self.period, (T,)),
        ):
            if arr.shape != shape:
                raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
        if len(self.player_ids) != P or self.teams.shape[0] != P or self.is_gk.shape[0] != P:
            raise ValueError("player metadata length does not match the tracking array")
        if T > 1:
            dt = float(np.median(np.diff(self.time)))
            if dt <= 0 or abs(1.0 / dt - self.fps) > 0.51 * max(1.0, self.fps / 25.0):
                raise ValueError(
                    f"declared fps={self.fps} disagrees with the median timestep {dt:.4f}s "
                    f"(implies {1/dt:.2f} fps)"
                )
        inside = self.pitch.contains(self.xy.reshape(-1, 2), tol=10.0)
        finite = np.isfinite(self.xy.reshape(-1, 2)).all(axis=1)
        bad = float(np.mean(~inside[finite])) if finite.any() else 0.0
        if bad > 0.02:
            raise ValueError(
                f"{bad:.1%} of finite player positions lie more than 10 m outside the pitch; "
                "the coordinate mapping is probably wrong"
            )


@dataclass
class ArrivalSpec:
    """One candidate ball arrival, in provider-neutral terms.

    ``t_release`` and ``t_arrival`` are on the tracking clock. ``origin`` and
    ``destination`` are in the same *unrotated* metric frame as
    :class:`TrackingTable`.
    """

    arrival_id: str
    period: int
    team_a: str
    team_b: str
    t_release: float
    t_arrival: float
    origin: np.ndarray
    destination: np.ndarray
    arrival_type: str
    is_endogenous: bool
    set_piece: bool = False
    pass_height: str = "unknown"
    possession_id: str = ""
    source_event: str = ""


def compute_velocities(
    xy: np.ndarray, *, fps: float, window_seconds: float = 0.4, polyorder: int = 2,
    speed_cap: float = 12.0, max_gap_seconds: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Derive per-player velocity and an observation mask from positions.

    Returns ``(vel, observed, diagnostics)``. Velocity is differentiated
    per player with Savitzky-Golay; frames inside a long occlusion gap are
    marked unobserved, because an interpolated position yields a velocity that
    is an artefact of the filter rather than a measurement.
    """
    T, P, _ = xy.shape
    vel = np.full_like(xy, np.nan)
    observed = np.zeros((T, P), dtype=bool)
    n_capped = 0

    for p in range(P):
        track = xy[:, p, :]
        present = np.isfinite(track).all(axis=1)
        if present.sum() < 3:
            continue
        v = derivative(track, fps=fps, window_seconds=window_seconds, polyorder=polyorder)
        n_capped += int(np.sum(np.linalg.norm(v, axis=1) > speed_cap))
        vel[:, p, :] = cap_speed(v, v_cap=speed_cap)
        observed[:, p] = present & ~gap_mask(track, fps=fps, max_gap_seconds=max_gap_seconds)

    return vel, observed, {
        "frames_speed_capped": n_capped,
        "velocity_window_seconds": window_seconds,
        "velocity_polyorder": polyorder,
        "speed_cap": speed_cap,
    }


def infer_attack_sign(
    tracking_xy: np.ndarray, teams: np.ndarray, periods: np.ndarray,
    *, is_gk: np.ndarray | None = None,
) -> dict:
    """Infer each team's attacking direction per period.

    **The goalkeeper is the signal.** A keeper stays in front of the goal their
    team defends for the whole period, so the sign of their mean ``x``
    identifies that goal and the team attacks the other one. The whole period
    is averaged, not just its opening, because the keeper's position is stable
    throughout.

    The obvious alternative - the sign of the team's mean ``x`` in the opening
    frames, on the reasoning that both teams start in their own half - is
    **not reliable**, and this is worth recording because it is the natural
    first implementation. Within a few seconds of kick-off the kicking team has
    already advanced, so its outfield mean can sit on the wrong side of the
    halfway line while its keeper has not moved. On Metrica's Sample Game 1 that
    rule misclassifies the team taking the kick-off. The team-mean rule is
    retained only as a fallback for corpora with no identified goalkeeper.

    Getting this wrong mirrors the pitch: every aggregate still looks plausible
    while the half-spaces are swapped and progression runs backwards. The
    result is therefore cross-checked in :func:`assemble_arrivals` against the
    direction of play the arrivals themselves imply.
    """
    signs: dict = {}
    for period in np.unique(periods):
        idx = np.flatnonzero(periods == period)
        if idx.size == 0:
            continue
        for team in np.unique(teams):
            cols = np.flatnonzero(teams == team)
            gk_cols = cols[is_gk[cols]] if is_gk is not None else np.array([], dtype=int)

            reference = np.nan
            if gk_cols.size:
                block = tracking_xy[np.ix_(idx, gk_cols)][..., 0]
                if np.isfinite(block).any():
                    reference = float(np.nanmean(block))
            if not np.isfinite(reference):
                block = tracking_xy[np.ix_(idx, cols)][..., 0][:250]
                reference = float(np.nanmean(block)) if np.isfinite(block).any() else 0.0
                # Team-mean fallback: own half is negative x -> attacks +x.
                signs[(str(team), int(period))] = 1.0 if reference < 0 else -1.0
                continue

            # Keeper in the negative half -> that team defends -x, attacks +x.
            signs[(str(team), int(period))] = 1.0 if reference < 0 else -1.0
    return signs


def build_possession_track(
    events: pd.DataFrame, tracking: TrackingTable, *,
    team_col: str = "team", time_col: str = "t", period_col: str = "period",
    stoppage_col: str | None = None,
) -> tuple[pd.Series, np.ndarray]:
    """Derive a per-frame possession label from a sparse on-ball event stream.

    Providers that do not supply a frame-level possession label still supply an
    ordered sequence of on-ball events. The team in possession at a frame is
    taken to be the team of the most recent on-ball event at or before it,
    which is the weakest assumption that yields a usable label.

    **This is a derived, not a measured, quantity** and the study must say so
    for any provider where this path is taken. Its principal weakness is at
    transitions: between a loss and the opponent's recovery the ball is loose,
    and this construction attributes it to whichever event came last. Arrivals
    whose horizon window straddles a transition are therefore the ones most
    exposed to label noise, which is why the horizon sensitivity sweep matters.

    Returns ``(possession_by_frame, stoppage_by_frame)``.
    """
    ev = events.dropna(subset=[time_col]).sort_values([period_col, time_col])
    poss = np.full(tracking.n_frames, None, dtype=object)
    stop = np.zeros(tracking.n_frames, dtype=bool)

    for period in np.unique(tracking.period):
        mask = tracking.period == period
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        t = tracking.time[idx]
        sub = ev[ev[period_col] == period]
        if sub.empty:
            continue
        pos = np.searchsorted(sub[time_col].to_numpy(), t, side="right") - 1
        valid = pos >= 0
        teams = sub[team_col].to_numpy()
        assigned = np.full(idx.size, None, dtype=object)
        assigned[valid] = teams[pos[valid]]
        poss[idx] = assigned
        if stoppage_col is not None and stoppage_col in sub.columns:
            flags = sub[stoppage_col].to_numpy(dtype=bool)
            s = np.zeros(idx.size, dtype=bool)
            s[valid] = flags[pos[valid]]
            stop[idx] = s

    return pd.Series(poss), stop



def arrival_row(
    spec: "ArrivalSpec",
    *,
    origin: np.ndarray,
    destination: np.ndarray,
    flight_time: float,
    label: dict,
    horizon: float,
    match_id: str,
    competition: str,
    n_players_a: int,
    n_players_b: int,
    pressure_index: float,
    frame_completeness: float,
    tracking_source: str,
    provider: str,
    sync_offset: float = 0.0,
    flight_time_imputed: bool = False,
    quality_flag: str = "ok",
    row_ordinal: int = 0,
    extra: dict | None = None,
) -> dict:
    """Build one row of the arrivals table.

    Shared by the continuous-tracking path (:func:`assemble_arrivals`) and by
    adapters working from per-event freeze frames, which have no continuous
    tracking to index into. Keeping the schema in one function is what stops
    the two paths drifting apart: a column added for one provider and forgotten
    for another would silently produce two incompatible corpora that the
    evaluation code would happily concatenate.
    """
    origin = np.asarray(origin, dtype=float).reshape(2)
    destination = np.asarray(destination, dtype=float).reshape(2)
    flight_time = float(max(flight_time, 1e-3))
    length = float(np.linalg.norm(destination - origin))

    row = {
        "arrival_id": spec.arrival_id,
        "match_id": match_id,
        "competition": competition,
        "possession_id": spec.possession_id or f"{match_id}_p{row_ordinal // 5:05d}",
        "period": int(spec.period),
        "team_a": spec.team_a,
        "team_b": spec.team_b,
        "t_release": float(spec.t_release),
        "t_arrival": float(spec.t_release + flight_time),
        "flight_time": flight_time,
        "flight_time_imputed": bool(flight_time_imputed),
        "origin_x": float(origin[0]), "origin_y": float(origin[1]),
        "dest_x": float(destination[0]), "dest_y": float(destination[1]),
        "y_control": int(label["y"]),
        "y_first_touch": int(label["y"]),
        "control_horizon": float(horizon),
        "outcome_censored": bool(label["censored"]),
        "arrival_type": spec.arrival_type,
        "is_endogenous": bool(spec.is_endogenous),
        "pass_length": length,
        "pass_height": spec.pass_height,
        "ball_speed": length / flight_time,
        "score_diff": 0,
        "minute": float(spec.t_release / 60.0),
        "n_players_a": int(n_players_a),
        "n_players_b": int(n_players_b),
        "pressure_index": float(pressure_index),
        "set_piece": bool(spec.set_piece),
        "tracking_source": tracking_source,
        "provider": provider,
        "frame_completeness": float(frame_completeness),
        "sync_offset": float(sync_offset),
        "quality_flag": quality_flag,
        "source_event": spec.source_event,
    }
    if extra:
        row.update(extra)
    return row


def assemble_arrivals(
    tracking: TrackingTable,
    specs: list[ArrivalSpec],
    possession: pd.Series,
    stoppage: np.ndarray,
    *,
    match_id: str,
    competition: str,
    tracking_source: str,
    provider: str,
    label_config: LabelConfig | None = None,
    match_date=None,
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """Turn a tracking table and arrival specs into frames and the arrivals table.

    The provider-independent assembly step. Two invariants it enforces, both of
    which would otherwise corrupt the study silently:

    1. **State is taken at release, never at arrival.** Defenders converge on
       the ball *because* it is arriving, so conditioning on arrival-time
       positions would leak the outcome into the forecast.
    2. **Coordinates are rotated per arrival** so that the team in possession
       always attacks ``+x``. Without this, every spatial statistic averages
       two mirror-image populations.
    """
    cfg = label_config or LabelConfig()
    frames: list[ArrivalFrame] = []
    rows: list[dict] = []

    for spec in specs:
        i_rel = tracking.frame_index(spec.t_release, spec.period)
        sign = float(tracking.attack_sign.get((spec.team_a, int(spec.period)), 1.0))

        xy = tracking.xy[i_rel] * sign
        vel = tracking.vel[i_rel] * sign
        obs = tracking.observed[i_rel].copy()

        is_a = tracking.teams == spec.team_a
        is_b = tracking.teams == spec.team_b
        on_pitch = np.isfinite(xy).all(axis=1)
        sel_a = is_a & on_pitch
        sel_b = is_b & on_pitch
        if sel_a.sum() < 6 or sel_b.sum() < 6:
            # Fewer than six tracked players a side is not the modelled problem;
            # recorded as a rejection rather than silently skipped.
            continue

        att_v = np.nan_to_num(vel[sel_a], nan=0.0)
        def_v = np.nan_to_num(vel[sel_b], nan=0.0)
        origin = spec.origin * sign
        dest = spec.destination * sign
        flight = float(max(spec.t_arrival - spec.t_release, 1e-3))

        label = label_from_possession_track(
            possession, tracking.time, spec.t_arrival, spec.team_a, cfg,
            stoppage=pd.Series(stoppage),
            periods=tracking.period, arrival_period=int(spec.period),
        )

        completeness = float((obs[sel_a].sum() + obs[sel_b].sum()) / 22.0)
        n_opp_near_origin = int(np.sum(np.linalg.norm(xy[sel_b] - origin[None, :], axis=1) <= 5.0))

        frames.append(
            ArrivalFrame(
                att_xy=xy[sel_a], att_v=att_v, def_xy=xy[sel_b], def_v=def_v,
                target=dest, flight_time=flight,
                att_is_gk=tracking.is_gk[sel_a], def_is_gk=tracking.is_gk[sel_b],
                att_observed=obs[sel_a], def_observed=obs[sel_b],
                meta={"origin": origin, "match_id": match_id, "arrival_id": spec.arrival_id},
            )
        )
        rows.append(
            arrival_row(
                spec, origin=origin, destination=dest, flight_time=flight, label=label,
                horizon=cfg.horizon, match_id=match_id, competition=competition,
                n_players_a=int(sel_a.sum()), n_players_b=int(sel_b.sum()),
                pressure_index=n_opp_near_origin, frame_completeness=completeness,
                tracking_source=tracking_source, provider=provider,
                sync_offset=float(tracking.meta.get("sync_offset", 0.0)),
                row_ordinal=len(rows),
            )
        )

    table = pd.DataFrame(rows)
    if not table.empty and match_date is not None:
        table["match_date"] = pd.to_datetime(match_date)

    _check_direction_of_play(table)
    return frames, table


def _check_direction_of_play(table: pd.DataFrame, *, min_rows: int = 200) -> None:
    """Sanity-check the attacking-direction inference against the arrivals.

    In the canonical frame the team in possession attacks ``+x``, so passes
    should be forward on average and arrivals should skew toward positive
    ``x`` relative to origins. A negative mean progression over a whole match
    is the signature of a mirrored pitch - a failure mode that produces
    entirely plausible-looking but wrong spatial results, so it is checked
    rather than assumed.
    """
    import warnings

    if len(table) < min_rows:
        return
    progression = float((table["dest_x"] - table["origin_x"]).mean())
    # Threshold at +1.5 m, not a negative number. Teams pass forward on net, and
    # the three corpora checked here all sit between +2.0 and +4.2 m, so a match
    # averaging below +1.5 m is already anomalous. The bar is set from observed
    # failures rather than intuition: two real mirroring bugs produced -0.59 m
    # and +0.71 m, and an earlier -2.0 m threshold caught neither. Mirroring is
    # otherwise close to invisible - base rates, pass lengths, flight times and
    # outcome orderings all survive it intact.
    #
    # A false positive here costs one investigation; a false negative silently
    # swaps the half-spaces in every spatial result. Hence a warning, with the
    # likely causes named, rather than an exception.
    if progression < 1.5:
        warnings.warn(
            f"mean pass progression is {progression:+.2f} m, which is not clearly "
            "forward. The corpora checked for this study sit between +2.0 and +4.2 m. "
            "The most likely cause is a "
            "mirrored pitch: either the attacking-direction inference is wrong, or "
            "the provider's event coordinates are ALREADY attack-normalised and the "
            "rotation has been applied twice.",
            stacklevel=3,
        )
