"""StatsBomb Open Data adapter — the fallback (freeze-frame) design.

**Verified against the real files** (repository `statsbomb/open-data`, checked
2026-09-15 on FIFA World Cup 2022, competition 43 / season 106). What the data
actually contains, as opposed to what planning assumed:

* **Coordinates** are a 120 x 80 *template*, with (0, 0) at the top left, so the
  y-axis increases downward and must be negated. The template is not metres: a
  real pitch may be 100-110 m long, so mapping it onto 105 x 68 m introduces a
  metric error into every time-to-point calculation. That error is systematic
  per match and unmeasurable without the true dimensions.
* **Events are already attack-normalised**: the team in possession always plays
  toward x = 120. Mean pass progression is positive, so no attacking-direction
  inference is needed - unlike the Metrica adapter, where getting that wrong
  mirrors the pitch.
* **Flight time is measured, not imputed.** Every pass carries a ``duration``,
  and the implied ball speeds (5th-95th percentile 6.9-22.1 m/s, median 13.1)
  are physically plausible. This corrects the study plan's assumption that
  StatsBomb arrival times would have to come from a ball-speed model. The
  caveat: ``duration`` runs to the related event (the receipt), which is the
  flight time for a completed pass and an approximation of it otherwise.
* **``pass.height``** distinguishes Ground / Low / High passes, so aerial and
  ground arrivals *can* be separated here - which continuous-tracking corpora
  without a ball z-coordinate cannot do.
* **Possession is labelled at event level** (``possession`` sequence id and
  ``possession_team``), so the outcome does not have to be reconstructed from a
  transfer log as it does for Metrica.

The two findings that constrain the design
------------------------------------------
1. **No velocity, at all.** A freeze frame is a single snapshot. Every model
   runs in its zero-velocity form, which is exactly the "remove velocity"
   ablation - so the fallback *measures* what that ablation simulates.
2. **Freeze frames show only what the camera saw.** On the matches checked,
   360 frames cover 86.6% of passes; the median frame shows **17 of 22
   players** and **no frame shows all 22**. Worse, **16.4% of pass destinations
   fall outside the ``visible_area`` polygon entirely.** For those arrivals,
   "no defender near the destination" means "no defender *visible*", and a
   control model will confidently report that the attacking team owns the space
   while being uninformed rather than right.

   This adapter therefore computes ``dest_visible`` per arrival and marks
   invisible-destination arrivals with a ``quality_flag`` so they can be
   excluded or analysed as their own stratum. Treating them as ordinary
   arrivals would bias control upward precisely where the study is looking.

Licence: governed by StatsBomb's own user agreement, held in the repository.
Read it before using the data or publishing anything derived from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pcc.data.labels import LabelConfig, label_from_possession_track
from pcc.data.schema import ArrivalFrame
from pcc.data.tracking import ArrivalSpec, arrival_row, build_possession_track
from pcc.geometry import Pitch

#: StatsBomb's coordinate template. NOT metres - see the module docstring.
SB_LENGTH, SB_WIDTH = 120.0, 80.0

#: Event types that interrupt play; arrivals whose horizon window straddles one
#: are censored rather than counted as a loss of control.
STOPPAGE_EVENTS = {
    "Foul Committed", "Foul Won", "Injury Stoppage", "Substitution",
    "Half Start", "Half End", "Offside", "Player Off", "Player On",
    "Tactical Shift", "Referee Ball-Drop", "Bad Behaviour",
}

#: Pass types that are dead-ball restarts.
SET_PIECE_PASS_TYPES = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off", "Penalty"}

#: The frequency of the pseudo-frame grid on which possession is evaluated.
#: 10 Hz is finer than the event stream and coarse enough to stay cheap; it only
#: has to resolve the control horizon, which is of order a second.
POSSESSION_GRID_HZ = 10.0


def to_metric(xy, pitch: Pitch = Pitch()) -> np.ndarray:
    """Map StatsBomb's 120 x 80 template onto centred metres.

    The y-axis is negated because StatsBomb's origin is the top-left corner.
    The rescaling assumes a 105 x 68 m pitch; the true dimensions are not
    supplied, so this is an approximation whose error is systematic per match
    (see the module docstring, and the pitch-dimension ablation).
    """
    arr = np.atleast_2d(np.asarray(xy, dtype=float))[:, :2]
    out = np.empty_like(arr)
    out[:, 0] = (arr[:, 0] / SB_LENGTH - 0.5) * pitch.length
    out[:, 1] = -(arr[:, 1] / SB_WIDTH - 0.5) * pitch.width
    return out


def _timestamp_seconds(ts: str) -> float:
    """Convert ``HH:MM:SS.mmm`` to seconds from the start of the period."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_events(path: Path) -> pd.DataFrame:
    """Read one match's events into a flat frame with the fields this study needs."""
    raw = json.loads(Path(path).read_text())
    rows = []
    for e in raw:
        p = e.get("pass") or {}
        rows.append(
            {
                "id": e["id"],
                "index": e.get("index"),
                "period": int(e.get("period", 1)),
                "t": _timestamp_seconds(e["timestamp"]),
                "minute": e.get("minute", 0),
                "type": e["type"]["name"],
                "team": (e.get("team") or {}).get("name"),
                "possession": e.get("possession"),
                "possession_team": (e.get("possession_team") or {}).get("name"),
                "play_pattern": (e.get("play_pattern") or {}).get("name"),
                "duration": e.get("duration"),
                "loc": e.get("location"),
                "pass_end": p.get("end_location"),
                "pass_height": (p.get("height") or {}).get("name"),
                "pass_type": (p.get("type") or {}).get("name"),
                "pass_outcome": (p.get("outcome") or {}).get("name"),
                "pass_cross": bool(p.get("cross", False)),
                "pass_length": p.get("length"),
            }
        )
    ev = pd.DataFrame(rows).sort_values(["period", "t"]).reset_index(drop=True)
    ev["is_stoppage"] = ev["type"].isin(STOPPAGE_EVENTS)
    return ev


def load_frames(path: Path) -> dict:
    """Read one match's 360 frames, keyed by ``event_uuid``."""
    return {f["event_uuid"]: f for f in json.loads(Path(path).read_text())}


def visible_area_polygon(frame: dict) -> np.ndarray | None:
    """The camera's covered region as an ``(n, 2)`` polygon, or ``None``."""
    va = frame.get("visible_area")
    if not va or len(va) < 6:
        return None
    return np.asarray(va, dtype=float).reshape(-1, 2)


def point_in_polygon(point, polygon: np.ndarray) -> bool:
    """Ray-casting containment test.

    Implemented here rather than pulled from matplotlib so that the data layer
    carries no plotting dependency; the algorithm is standard and is covered by
    a unit test against known-inside and known-outside points.
    """
    x, y = float(point[0]), float(point[1])
    n = polygon.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def freeze_frame_arrays(frame: dict, pitch: Pitch = Pitch()):
    """Split a freeze frame into team-A and team-B positions in canonical metres.

    ``teammate`` is relative to the *actor* - the player performing the event -
    so team A is the passing team by construction. The actor itself is included
    in team A: they are on the pitch and can in principle recover their own
    pass.

    Returns ``(att_xy, def_xy, att_is_gk, def_is_gk, n_visible)``. **There is no
    velocity**, and callers must supply zeros rather than inventing one.
    """
    att, deff, att_gk, def_gk = [], [], [], []
    for pl in frame.get("freeze_frame", []):
        loc = pl.get("location")
        if not loc or len(loc) < 2:
            continue
        (att if pl.get("teammate") else deff).append(loc[:2])
        (att_gk if pl.get("teammate") else def_gk).append(bool(pl.get("keeper")))

    att_xy = to_metric(att, pitch) if att else np.zeros((0, 2))
    def_xy = to_metric(deff, pitch) if deff else np.zeros((0, 2))
    return att_xy, def_xy, np.array(att_gk, dtype=bool), np.array(def_gk, dtype=bool), len(att) + len(deff)


def build_possession_grid(events: pd.DataFrame, *, hz: float = POSSESSION_GRID_HZ):
    """A pseudo-frame possession track from the event stream.

    StatsBomb labels possession at event level, so a regular time grid is
    interpolated from it in order to reuse the same horizon-based outcome
    definition the tracking corpora use. Using the same construct across
    providers is what keeps the fallback design comparable to the primary one
    rather than answering a subtly different question.
    """
    class _Grid:
        pass

    times, periods = [], []
    for period, sub in events.groupby("period"):
        t0, t1 = float(sub["t"].min()), float(sub["t"].max())
        grid = np.arange(t0, t1 + 1.0 / hz, 1.0 / hz)
        times.append(grid)
        periods.append(np.full(grid.size, int(period)))

    grid = _Grid()
    grid.time = np.concatenate(times) if times else np.zeros(0)
    grid.period = np.concatenate(periods) if periods else np.zeros(0, dtype=int)
    grid.n_frames = grid.time.size

    possession, stoppage = build_possession_track(
        events, grid, team_col="possession_team", time_col="t",
        period_col="period", stoppage_col="is_stoppage",
    )
    return grid, possession, stoppage


def _classify(row) -> tuple[str, bool, bool]:
    """``(arrival_type, is_endogenous, set_piece)`` for one pass event."""
    pass_type = row["pass_type"]
    set_piece = bool(pass_type in SET_PIECE_PASS_TYPES)
    if set_piece:
        return "set_piece", True, True
    if row["pass_cross"]:
        return "cross", True, False
    return "open_pass", True, False


def load_match(
    events_path: Path,
    frames_path: Path | None,
    *,
    match_id: str,
    competition: str,
    teams: tuple[str, str] | None = None,
    label_config: LabelConfig | None = None,
    pitch: Pitch = Pitch(),
    require_360: bool = True,
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """Load one StatsBomb match into frames and an arrivals table."""
    cfg = label_config or LabelConfig()
    events = load_events(events_path)
    frames_by_uuid = load_frames(frames_path) if frames_path and Path(frames_path).exists() else {}
    grid, possession, stoppage = build_possession_grid(events)

    all_teams = [t for t in events["team"].dropna().unique()]
    if teams is None and len(all_teams) >= 2:
        teams = (all_teams[0], all_teams[1])

    out_frames: list[ArrivalFrame] = []
    rows: list[dict] = []

    passes = events[(events["type"] == "Pass") & events["loc"].notna() & events["pass_end"].notna()]
    for _, ev in passes.iterrows():
        ff = frames_by_uuid.get(ev["id"])
        if ff is None:
            if require_360:
                continue
            ff = {"freeze_frame": [], "visible_area": None}

        att_xy, def_xy, att_gk, def_gk, n_visible = freeze_frame_arrays(ff, pitch)
        if att_xy.shape[0] < 3 or def_xy.shape[0] < 3:
            continue

        duration = ev["duration"]
        flight_imputed = not (isinstance(duration, (int, float)) and np.isfinite(duration) and duration > 0)
        origin_sb = np.asarray(ev["loc"][:2], dtype=float)
        dest_sb = np.asarray(ev["pass_end"][:2], dtype=float)
        origin = to_metric(origin_sb, pitch)[0]
        dest = to_metric(dest_sb, pitch)[0]
        if flight_imputed:
            duration = float(np.linalg.norm(dest - origin) / 13.0)  # median implied ball speed
        flight = float(max(duration, 1e-3))

        poly = visible_area_polygon(ff)
        dest_visible = bool(poly is not None and point_in_polygon(dest_sb, poly))

        team_a = str(ev["team"])
        team_b = next((t for t in all_teams if t != team_a), "opponent")

        label = label_from_possession_track(
            possession, grid.time, float(ev["t"]) + flight, team_a, cfg,
            stoppage=pd.Series(stoppage),
            periods=grid.period, arrival_period=int(ev["period"]),
        )

        arrival_type, endogenous, set_piece = _classify(ev)
        spec = ArrivalSpec(
            arrival_id=str(ev["id"]),
            period=int(ev["period"]), team_a=team_a, team_b=team_b,
            t_release=float(ev["t"]), t_arrival=float(ev["t"]) + flight,
            origin=origin, destination=dest,
            arrival_type=arrival_type, is_endogenous=endogenous, set_piece=set_piece,
            pass_height={"Ground Pass": "ground", "Low Pass": "low", "High Pass": "high"}.get(
                ev["pass_height"], "unknown"
            ),
            possession_id=f"{match_id}_poss{int(ev['possession']):05d}" if pd.notna(ev["possession"]) else "",
            source_event=(
                f"Pass|{'open_play' if pd.isna(ev['pass_type']) else ev['pass_type']}"
                f"|{'COMPLETE' if pd.isna(ev['pass_outcome']) else ev['pass_outcome']}"
            ),
        )

        out_frames.append(
            ArrivalFrame(
                att_xy=att_xy,
                att_v=np.zeros_like(att_xy),   # no velocity: a freeze frame is one snapshot
                def_xy=def_xy,
                def_v=np.zeros_like(def_xy),
                target=dest, flight_time=flight,
                att_is_gk=att_gk, def_is_gk=def_gk,
                meta={"origin": origin, "match_id": match_id, "arrival_id": spec.arrival_id,
                      "dest_visible": dest_visible},
            )
        )
        rows.append(
            arrival_row(
                spec, origin=origin, destination=dest, flight_time=flight, label=label,
                horizon=cfg.horizon, match_id=match_id, competition=competition,
                n_players_a=att_xy.shape[0], n_players_b=def_xy.shape[0],
                pressure_index=float(np.sum(np.linalg.norm(def_xy - origin[None, :], axis=1) <= 5.0)),
                frame_completeness=n_visible / 22.0,
                tracking_source="freeze_frame",
                provider="statsbomb_open (360 freeze frames; NO velocity)",
                flight_time_imputed=flight_imputed,
                quality_flag="ok" if dest_visible else "destination_not_visible",
                row_ordinal=len(rows),
                # pd.isna, not `or`: a NaN outcome is truthy in Python, so
                # `ev["pass_outcome"] or "COMPLETE"` silently keeps the NaN.
                extra={
                    "dest_visible": dest_visible,
                    "pass_outcome": "COMPLETE" if pd.isna(ev["pass_outcome"]) else ev["pass_outcome"],
                },
            )
        )

    return out_frames, pd.DataFrame(rows)


def available_matches(root: Path) -> list[tuple[str, Path, Path | None]]:
    """Matches with an events file, paired with their 360 file where present."""
    root = Path(root)
    events_dir = root / "events"
    frames_dir = root / "three-sixty"
    if not events_dir.is_dir():
        return []
    out = []
    for ep in sorted(events_dir.glob("*.json")):
        fp = frames_dir / ep.name
        out.append((ep.stem, ep, fp if fp.exists() else None))
    return out


def match_metadata(root: Path) -> dict:
    """``{match_id: {competition, date, home, away}}`` from the matches index."""
    meta: dict = {}
    for mp in (Path(root) / "matches").rglob("*.json"):
        try:
            for m in json.loads(mp.read_text()):
                meta[str(m["match_id"])] = {
                    "competition": f"{m['competition']['competition_name']} "
                                   f"{m['season']['season_name']}",
                    "date": m.get("match_date"),
                    "home": m["home_team"]["home_team_name"],
                    "away": m["away_team"]["away_team_name"],
                }
        except Exception:
            continue
    return meta
