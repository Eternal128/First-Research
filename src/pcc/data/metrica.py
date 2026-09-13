"""Metrica Sports sample-data adapter.

**Verified against the real files** (repository `metrica-sports/sample-data`,
commit `e706dd5`, checked 2026-09-13). The facts below were read from the data,
not from documentation:

* Three matches. Games 1 and 2 are in the CSV layout this module parses; Game 3
  is in the EPTS/FIFA format with JSON events and is **not** handled here - it
  needs a different deserialiser (``kloppy`` is the sensible route).
* Tracking is 25 fps (timestamps step by 0.04 s). ~145,000 frames per match.
* Coordinates are normalised to ``[0, 1]`` with **(0, 0) at the top left and
  (1, 1) at the bottom right**, so the y-axis increases *downward* and must be
  flipped. Pitch dimensions are 105 x 68 m for both matches.
* Tracking and event data are already synchronised (the event file carries
  frame numbers), so no clock-offset estimation is needed. This is unusual and
  worth stating: it removes one of the larger error sources in the pipeline.
* Each tracking CSV has **three header rows**: team labels, jersey numbers, then
  column names. Each player occupies two columns (x, y) with the name on the
  first only; the last pair is the ball.
* Players not on the pitch are ``NaN``.
* **There is no frame-level possession label.** The outcome must be derived
  from the event stream - see :func:`possession_from_events`. This is a
  substantive limitation and is recorded in the arrivals table's ``provider``
  field so it cannot be forgotten downstream.

Event taxonomy actually present (Game 1 / Game 2 counts):
``PASS`` 799/964, ``RECOVERY`` 278/248, ``BALL LOST`` 257/233,
``CHALLENGE`` 233/311, ``SET PIECE`` 77/80, ``BALL OUT`` 51/49,
``SHOT`` 24/24, ``FAULT RECEIVED`` 22/20, ``CARD`` 4/6.

The stream is a paired possession-transfer log: a ``BALL LOST`` by one team is
followed by a ``RECOVERY`` by the other. A ``BALL LOST`` with an
``INTERCEPTION`` subtype carries start and end coordinates and is therefore an
*attempted pass that was intercepted* - a failed-pass arrival, not merely an
outcome annotation.

Licence note: the repository asks that use be responsible and that the source
be acknowledged if anything is made public. Read the repository's own terms
before publishing; this note is not a substitute for them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pcc.data.labels import LabelConfig
from pcc.data.tracking import (
    ArrivalSpec, TrackingTable, assemble_arrivals, build_possession_track,
    compute_velocities, infer_attack_sign,
)
from pcc.geometry import Pitch

#: Verified from the data, not assumed.
METRICA_FPS = 25.0
METRICA_PITCH = Pitch(length=105.0, width=68.0)

#: Event subtypes that mark a dead ball, used for outcome censoring.
STOPPAGE_TYPES = {"BALL OUT", "FAULT RECEIVED", "CARD"}

#: Subtype tokens indicating the ball travelled to another player.
_INTERCEPTION_TOKENS = ("INTERCEPTION",)
_CLEARANCE_TOKENS = ("CLEARANCE",)
_SET_PIECE_TOKENS = ("GOAL KICK", "THROW IN", "CORNER", "FREE KICK", "KICK OFF")


def _subtype(value) -> str:
    return "" if pd.isna(value) else str(value).upper()


def parse_tracking_csv(path: Path) -> dict:
    """Parse one Metrica tracking CSV into arrays.

    Returns ``{time, period, frame, xy, ball_xy, player_ids, jerseys, team}``
    with coordinates still in the provider's normalised ``[0, 1]`` frame.
    """
    raw = pd.read_csv(path, header=None, dtype=str, low_memory=False)
    team_row, jersey_row, name_row = raw.iloc[0], raw.iloc[1], raw.iloc[2]
    body = raw.iloc[3:].reset_index(drop=True)

    period = body.iloc[:, 0].astype(float).astype(int).to_numpy()
    frame = body.iloc[:, 1].astype(float).astype(int).to_numpy()
    time = body.iloc[:, 2].astype(float).to_numpy()

    player_cols, player_ids, jerseys, team_name = [], [], [], None
    ball_col = None
    for col in range(3, raw.shape[1] - 1, 2):
        name = str(name_row.iloc[col]).strip()
        if not name or name.lower() in ("nan", ""):
            continue
        if name.lower() == "ball":
            ball_col = col
            continue
        player_cols.append(col)
        player_ids.append(name)
        jerseys.append(str(jersey_row.iloc[col]).strip())
        if team_name is None:
            t = str(team_row.iloc[col]).strip()
            team_name = t if t and t.lower() != "nan" else None

    if ball_col is None:
        raise ValueError(f"no Ball column found in {path.name}")

    def _pair(col: int) -> np.ndarray:
        return body.iloc[:, [col, col + 1]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    xy = np.stack([_pair(c) for c in player_cols], axis=1) if player_cols else np.zeros((len(body), 0, 2))
    return {
        "time": time, "period": period, "frame": frame,
        "xy": xy, "ball_xy": _pair(ball_col),
        "player_ids": player_ids, "jerseys": jerseys,
        "team": team_name or path.stem.split("_")[-2],
    }


def to_metric(xy_norm: np.ndarray, pitch: Pitch = METRICA_PITCH) -> np.ndarray:
    """Map Metrica's normalised frame onto centred metres.

    ``(0, 0)`` is the top-left corner and ``(1, 1)`` the bottom-right, so the
    y-axis increases downward and is negated here. Failing to negate it mirrors
    the pitch left-to-right in every plot and swaps the half-spaces, while
    leaving every aggregate statistic looking entirely plausible.
    """
    out = np.empty_like(xy_norm)
    out[..., 0] = (xy_norm[..., 0] - 0.5) * pitch.length
    out[..., 1] = -(xy_norm[..., 1] - 0.5) * pitch.width
    return out


def load_tracking(game_dir: Path, *, velocity_window_s: float = 0.4,
                  max_gap_s: float = 0.5, speed_cap: float = 12.0) -> TrackingTable:
    """Build a :class:`TrackingTable` for one Metrica match."""
    game = game_dir.name
    home = parse_tracking_csv(game_dir / f"{game}_RawTrackingData_Home_Team.csv")
    away = parse_tracking_csv(game_dir / f"{game}_RawTrackingData_Away_Team.csv")

    n = min(home["time"].size, away["time"].size)
    if not np.allclose(home["time"][:n], away["time"][:n], atol=1e-6):
        raise ValueError(f"{game}: home and away tracking clocks disagree")

    xy = to_metric(np.concatenate([home["xy"][:n], away["xy"][:n]], axis=1))
    ball_xy = to_metric(home["ball_xy"][:n])
    teams = np.array(["Home"] * home["xy"].shape[1] + ["Away"] * away["xy"].shape[1])
    player_ids = [f"Home_{p}" for p in home["player_ids"]] + [f"Away_{p}" for p in away["player_ids"]]

    vel, observed, diag = compute_velocities(
        xy, fps=METRICA_FPS, window_seconds=velocity_window_s,
        speed_cap=speed_cap, max_gap_seconds=max_gap_s,
    )
    periods = home["period"][:n]
    # Order matters: goalkeepers are identified first, because the attacking
    # direction is inferred from where the keepers stand.
    is_gk = _infer_goalkeepers(xy, teams, periods)
    table = TrackingTable(
        time=home["time"][:n], period=periods, xy=xy, vel=vel, observed=observed,
        ball_xy=ball_xy, player_ids=player_ids, teams=teams,
        is_gk=is_gk,
        attack_sign=infer_attack_sign(xy, teams, periods, is_gk=is_gk),
        fps=METRICA_FPS, pitch=METRICA_PITCH,
        meta={"game": game, "sync_offset": 0.0, **diag},
    )
    table.validate()
    return table


def _infer_goalkeepers(xy: np.ndarray, teams: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Identify each team's goalkeeper as its most extreme-x player on average.

    Metrica's sample data is anonymised and carries no position labels, so the
    goalkeeper must be inferred. The keeper spends the match far behind the
    rest of the team, which makes the mean |x| separation large and the
    inference stable; it is nonetheless an inference, and it matters only for
    the goalkeeper's higher ball-control rate in the physics model.
    """
    is_gk = np.zeros(xy.shape[1], dtype=bool)
    first_period = periods == periods[0]
    for team in np.unique(teams):
        cols = np.flatnonzero(teams == team)
        mean_x = np.full(cols.size, np.nan)
        for k, c in enumerate(cols):
            column = xy[first_period, c, 0]
            if np.isfinite(column).any():
                mean_x[k] = np.nanmean(column)   # guarded: substitutes are all-NaN here
        if not np.isfinite(mean_x).any():
            continue
        team_centre = np.nanmean(mean_x)
        is_gk[cols[int(np.nanargmax(np.abs(mean_x - team_centre)))]] = True
    return is_gk


def load_events(game_dir: Path) -> pd.DataFrame:
    """Read and normalise one Metrica event file."""
    game = game_dir.name
    ev = pd.read_csv(game_dir / f"{game}_RawEventsData.csv")
    ev = ev.rename(columns={
        "Team": "team", "Type": "type", "Subtype": "subtype", "Period": "period",
        "Start Frame": "start_frame", "Start Time [s]": "t",
        "End Frame": "end_frame", "End Time [s]": "t_end",
        "From": "from_player", "To": "to_player",
        "Start X": "sx", "Start Y": "sy", "End X": "ex", "End Y": "ey",
    })
    ev["type"] = ev["type"].astype(str).str.upper().str.strip()
    ev["subtype_norm"] = ev["subtype"].map(_subtype)
    ev["is_stoppage"] = ev["type"].isin(STOPPAGE_TYPES)
    ev["event_index"] = np.arange(len(ev))
    return ev


def possession_from_events(events: pd.DataFrame, tracking: TrackingTable):
    """Derive a per-frame possession label and stoppage mask from the events.

    Metrica supplies no frame-level possession label, so this is a **derived**
    quantity. The event stream is a paired transfer log, which makes the
    derivation about as reliable as such a derivation can be, but the
    limitation is real and is carried into the arrivals table.
    """
    return build_possession_track(
        events, tracking, team_col="team", time_col="t", period_col="period",
        stoppage_col="is_stoppage",
    )


def extract_arrival_specs(events: pd.DataFrame, game: str,
                          pitch: Pitch = METRICA_PITCH) -> list[ArrivalSpec]:
    """Turn the event stream into ball arrivals.

    Two event patterns produce an arrival:

    ``PASS``
        A completed pass. The destination is both the intended and the realised
        arrival point.
    ``BALL LOST`` with an ``INTERCEPTION`` or ``CLEARANCE`` subtype
        An attempted pass that did not reach its target. The realised arrival
        point is where the opponent intervened, **not** where the passer aimed.

    That distinction matters enough to be encoded rather than glossed:
    intercepted arrivals are given their own ``arrival_type`` so the subgroup
    analysis can separate them. Their destination is selected partly by the
    defender's position, which biases them toward ``Y = 0`` for reasons that
    have nothing to do with the model - a selection effect the study must be
    able to see rather than average away.

    Excluded: ``BALL OUT`` (the ball leaves play, so there is no control
    outcome), ``SHOT``, ``CHALLENGE`` and bare ``BALL LOST`` without a travel
    subtype (a dispossession, not a pass), and ``RECOVERY`` (an outcome
    annotation of a preceding event, not an arrival in its own right).
    """
    specs: list[ArrivalSpec] = []
    teams = [t for t in events["team"].dropna().unique() if str(t) != "nan"]
    if len(teams) != 2:
        raise ValueError(f"{game}: expected two teams in the event file, found {teams}")

    # A possession sequence ends whenever the team on the ball changes.
    team_seq = events["team"].astype(str).to_numpy()
    possession_id = np.cumsum(np.r_[True, team_seq[1:] != team_seq[:-1]]) - 1

    prev_type = events["type"].shift(1).fillna("")
    prev_sub = events["subtype_norm"].shift(1).fillna("")

    for i, ev in events.iterrows():
        team_a = str(ev["team"])
        if team_a == "nan":
            continue
        team_b = teams[0] if team_a == teams[1] else teams[1]
        sub = ev["subtype_norm"]
        coords = [ev["sx"], ev["sy"], ev["ex"], ev["ey"]]
        if any(pd.isna(c) for c in coords) or pd.isna(ev["t"]) or pd.isna(ev["t_end"]):
            continue
        if ev["t_end"] <= ev["t"]:
            continue

        etype = ev["type"]
        if etype == "PASS":
            if any(tok in sub for tok in _CLEARANCE_TOKENS):
                arrival_type, endogenous = "clearance", False
            elif "CROSS" in sub:
                arrival_type, endogenous = "cross", True
            else:
                arrival_type, endogenous = "open_pass", True
        elif etype == "BALL LOST":
            if any(tok in sub for tok in _INTERCEPTION_TOKENS):
                arrival_type, endogenous = "interception", True
            elif any(tok in sub for tok in _CLEARANCE_TOKENS):
                arrival_type, endogenous = "clearance", False
            else:
                continue  # a dispossession, not a ball arrival
        else:
            continue

        set_piece = (
            any(tok in sub for tok in _SET_PIECE_TOKENS)
            or (prev_type.iloc[i] == "SET PIECE" and any(tok in prev_sub.iloc[i] for tok in _SET_PIECE_TOKENS))
        )
        height = "high" if "HEAD" in sub or "CROSS" in sub else "unknown"

        origin = to_metric(np.array([float(ev["sx"]), float(ev["sy"])]), pitch)
        dest = to_metric(np.array([float(ev["ex"]), float(ev["ey"])]), pitch)

        specs.append(
            ArrivalSpec(
                arrival_id=f"{game}_{int(ev['event_index']):05d}",
                period=int(ev["period"]), team_a=team_a, team_b=team_b,
                t_release=float(ev["t"]), t_arrival=float(ev["t_end"]),
                origin=origin, destination=dest,
                arrival_type=arrival_type, is_endogenous=endogenous,
                set_piece=bool(set_piece), pass_height=height,
                possession_id=f"{game}_poss{int(possession_id[i]):05d}",
                source_event=f"{etype}|{sub}",
            )
        )
    return specs


def available_games(root: Path) -> list[Path]:
    """Game directories in the CSV layout this adapter can read.

    Game 3 is deliberately excluded: it is in the EPTS/FIFA format and needs a
    different deserialiser. Silently skipping it without saying so would make
    the corpus size look like a data limitation rather than an adapter one.
    """
    data_dir = root / "data" if (root / "data").is_dir() else root
    out = []
    for d in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if (d / f"{d.name}_RawTrackingData_Home_Team.csv").exists():
            out.append(d)
    return out


def load_match(game_dir: Path, *, label_config: LabelConfig | None = None,
               competition: str = "Metrica-Sample", **tracking_kwargs):
    """Load one Metrica match into frames and an arrivals table."""
    tracking = load_tracking(game_dir, **tracking_kwargs)
    events = load_events(game_dir)
    possession, stoppage = possession_from_events(events, tracking)
    specs = extract_arrival_specs(events, game_dir.name)
    return assemble_arrivals(
        tracking, specs, possession, stoppage,
        match_id=game_dir.name, competition=competition,
        tracking_source="optical",
        provider="metrica_sample (possession label DERIVED from events)",
        label_config=label_config,
    )
