"""SkillCorner open broadcast-tracking adapter — the RQ4 arm.

**Verified against the real files** (repository `SkillCorner/opendata`, checked
2026-09-15, Australian A-League 2024/25). What the data actually contains:

* **Git LFS.** Tracking files are LFS pointers in a plain clone. Resolve them
  with ``scripts/fetch_data.py``; this module refuses to parse a stub rather
  than failing obscurely 200 lines later.
* **Coordinates are already centred metres**, and ``match.json`` supplies the
  **true pitch dimensions** (105 x 68 m for the match checked) rather than a
  template. Neither Metrica nor StatsBomb gives both.
* **Attacking direction is stated, not inferred.** ``match.json`` carries
  ``home_team_side`` as a per-period list (e.g. ``["left_to_right",
  "right_to_left"]``), which removes the single most dangerous inference in the
  Metrica adapter.
* **Frames are a global 10 Hz clock** (``match_periods`` gives frame ranges per
  period, increasing monotonically across the match), so ``t = frame / 10``.
  Note that the *timestamps* do overlap between periods - period 2 restarts at
  45:00 while period 1 ran to 48:18 - so the frame number, not the timestamp,
  is the safe clock.
* **Ball height is present** (``ball_data.z``), which Metrica lacks.
* **Per-frame possession** (``possession.group`` = "home team"/"away team") is
  supplied, though it is null in roughly 45% of frames.

The property that makes this corpus the RQ4 arm
------------------------------------------------
Every frame's ``player_data`` lists **all 22 players**, but each carries an
``is_detected`` flag, and on the match checked only **12.7 players per frame are
actually detected** (median 14); **13.8% of frames have none detected at all**,
and the ball is detected in only 60.4% of frames. The file is named
``tracking_extrapolated`` because the provider fills in the undetected players.

That is precisely the hazard this study exists to examine. Feeding extrapolated
positions to a control model and reporting the output as a probability is the
practice under scrutiny, so this adapter **keeps the distinction**: extrapolated
players are included in the state (a model in the field would see them) but
``att_observed`` / ``def_observed`` record which were really seen, and
``frame_completeness`` is computed from detections only. The comparison between
detected-rich and detected-poor arrivals is then available as a within-corpus
contrast, which is stronger than the across-corpus one because it holds the
competition and the provider fixed.

Arrival extraction
------------------
The raw event stream is ``{id}_dynamic_events.csv``. **Only its structural
fields are used** - frame indices, team, event/end type, start and end
coordinates, pass outcome. The same file carries SkillCorner's own
``xpass_completion``, ``xthreat``, ``possession_epv_*``, ``reception_difficulty``
and ``overall_pressure`` columns; those are fitted answers to closely related
questions and using them as inputs would contaminate the comparison. The
allowed set is enumerated in :data:`STRUCTURAL_COLUMNS` so the restriction is
enforced rather than merely intended.

A ``player_possession`` row that ends with ``end_type='pass'`` gives the
release; the **next** ``player_possession`` row gives the arrival - its
``frame_start`` is the arrival frame and its ``(x_start, y_start)`` the arrival
point. That linkage was validated: for successful passes the next possession's
start lies within 1 m of the provider's own stated reception point in 95.4% of
cases (median difference 0.00 m).

Using the next possession's start rather than the stated reception point is
deliberate, because reception coordinates are populated **only for successful
passes**. Taking them would restrict the corpus to passes that worked and leave
nothing to calibrate against. The consequence is that a failed pass's arrival is
placed **where the ball was actually gained**, matching the Metrica convention
and differing from StatsBomb's - a cross-provider incomparability recorded in
``docs/data_sources.md``.

Licence: released jointly by SkillCorner and PySport; the repository asks that
SkillCorner be credited. Read its README before publishing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pcc.data.labels import LabelConfig
from pcc.data.tracking import ArrivalSpec, TrackingTable, assemble_arrivals, compute_velocities
from pcc.geometry import Pitch

SKILLCORNER_FPS = 10.0

#: Event-file columns this adapter is permitted to read. Everything else in
#: ``dynamic_events.csv`` is either a derived metric from SkillCorner's own
#: models or irrelevant here; restricting the read is what keeps the model
#: comparison uncontaminated.
STRUCTURAL_COLUMNS = [
    "event_id", "index", "match_id", "period", "frame_start", "frame_end",
    "event_type", "event_subtype", "team_id", "team_shortname",
    "player_id", "player_name", "x_start", "y_start", "x_end", "y_end",
    "start_type", "end_type", "pass_outcome", "high_pass", "is_header",
    "attacking_side", "attacking_side_id",
    "game_interruption_before", "game_interruption_after",
]

#: End types that put the ball in flight toward another player.
ARRIVAL_END_TYPES = {"pass", "clearance"}

#: Start types that mark a restart rather than open play.
SET_PIECE_START_TYPES = {"throw_in_reception", "goal_kick_reception", "corner_reception", "free_kick_reception"}


class LFSPointerError(RuntimeError):
    """Raised when a tracking file is an unresolved Git LFS stub."""


def _check_not_lfs_pointer(path: Path) -> None:
    if path.stat().st_size < 1024 and path.read_bytes()[:40].startswith(b"version https://git-lfs"):
        raise LFSPointerError(
            f"{path} is an unresolved Git LFS pointer ({path.stat().st_size} bytes), not tracking "
            "data. A plain `git clone` does not fetch LFS content. Resolve it with:\n"
            "    python scripts/fetch_data.py --source skillcorner_open --accept-terms"
        )


def load_match_meta(match_dir: Path) -> dict:
    """Teams, true pitch dimensions, period frame ranges and player-team mapping."""
    mid = match_dir.name
    meta = json.loads((match_dir / f"{mid}_match.json").read_text())
    home, away = meta["home_team"], meta["away_team"]
    return {
        "match_id": mid,
        "home": home.get("short_name") or home["name"],
        "away": away.get("short_name") or away["name"],
        "home_id": home["id"],
        "away_id": away["id"],
        "pitch": Pitch(
            length=float(meta.get("pitch_length") or 105.0),
            width=float(meta.get("pitch_width") or 68.0),
        ),
        # Stated per period, not inferred - see the module docstring.
        "home_side": list(meta.get("home_team_side") or ["left_to_right", "right_to_left"]),
        "periods": {p["period"]: (p["start_frame"], p["end_frame"]) for p in meta.get("match_periods", [])},
        "player_team": {int(p["id"]): int(p["team_id"]) for p in meta.get("players", []) if p.get("team_id")},
        "player_role": {
            int(p["id"]): ((p.get("player_role") or {}).get("acronym") or "")
            for p in meta.get("players", [])
        },
        "date": meta.get("date_time"),
        "competition": (meta.get("competition_edition") or {}).get("name", "SkillCorner-Open"),
    }


def load_tracking(match_dir: Path, meta: dict | None = None, *,
                  velocity_window_s: float = 0.6, max_gap_s: float = 0.5,
                  speed_cap: float = 12.0) -> tuple[TrackingTable, pd.Series]:
    """Parse the tracking JSONL into a :class:`TrackingTable` plus a possession series.

    Returns ``(table, possession)`` where ``possession`` holds the per-frame
    team in possession (or ``None``), taken from the provider's own
    ``possession.group`` field rather than derived from events.
    """
    meta = meta or load_match_meta(match_dir)
    mid = match_dir.name
    path = match_dir / f"{mid}_tracking_extrapolated.jsonl"
    _check_not_lfs_pointer(path)

    frames, periods, poss = [], [], []
    ball, per_frame_players = [], []
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("period") is None:
                continue
            frames.append(int(d["frame"]))
            periods.append(int(d["period"]))
            group = (d.get("possession") or {}).get("group")
            poss.append(
                meta["home"] if group == "home team" else (meta["away"] if group == "away team" else None)
            )
            b = d.get("ball_data") or {}
            ball.append([b.get("x"), b.get("y")])
            per_frame_players.append(d.get("player_data") or [])

    if not frames:
        raise ValueError(f"{path} contained no frames with a period")

    player_ids = sorted({int(p["player_id"]) for pl in per_frame_players for p in pl if p.get("player_id")})
    index = {pid: i for i, pid in enumerate(player_ids)}

    n_t, n_p = len(frames), len(player_ids)
    xy = np.full((n_t, n_p, 2), np.nan)
    detected = np.zeros((n_t, n_p), dtype=bool)
    for t, players in enumerate(per_frame_players):
        for p in players:
            pid = p.get("player_id")
            if pid is None:
                continue
            j = index[int(pid)]
            xy[t, j] = (p.get("x"), p.get("y"))
            detected[t, j] = bool(p.get("is_detected"))

    # The frame index is the safe clock: it increases monotonically across the
    # whole match, whereas the timestamps restart mid-way and overlap.
    time = np.asarray(frames, dtype=float) / SKILLCORNER_FPS
    vel, _interp_observed, diag = compute_velocities(
        xy, fps=SKILLCORNER_FPS, window_seconds=velocity_window_s,
        speed_cap=speed_cap, max_gap_seconds=max_gap_s,
    )

    teams = np.array([
        meta["home"] if meta["player_team"].get(pid) == meta["home_id"] else meta["away"]
        for pid in player_ids
    ])
    is_gk = np.array([meta["player_role"].get(pid, "") == "GK" for pid in player_ids])

    periods_arr = np.asarray(periods, dtype=int)
    attack_sign = {}
    for period in np.unique(periods_arr):
        side = meta["home_side"][period - 1] if period - 1 < len(meta["home_side"]) else "left_to_right"
        home_sign = 1.0 if side == "left_to_right" else -1.0
        attack_sign[(meta["home"], int(period))] = home_sign
        attack_sign[(meta["away"], int(period))] = -home_sign

    table = TrackingTable(
        time=time, period=periods_arr, xy=xy, vel=vel,
        # `observed` is detection, not interpolation: the provider fills in
        # every player every frame, so an "interpolated" mask would be all True
        # and would hide exactly the effect RQ4 is about.
        observed=detected,
        ball_xy=np.asarray(ball, dtype=float),
        player_ids=[str(p) for p in player_ids], teams=teams, is_gk=is_gk,
        attack_sign=attack_sign, fps=SKILLCORNER_FPS, pitch=meta["pitch"],
        meta={"match_id": mid, "sync_offset": 0.0,
              "detection_rate": float(detected.mean()), **diag},
    )
    table.validate()
    return table, pd.Series(poss)


def load_events(match_dir: Path) -> pd.DataFrame:
    """Read only the structural columns of the dynamic-events file."""
    mid = match_dir.name
    path = match_dir / f"{mid}_dynamic_events.csv"
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in STRUCTURAL_COLUMNS if c in header]
    ev = pd.read_csv(path, usecols=usecols, low_memory=False)
    return ev.sort_values(["period", "frame_start"]).reset_index(drop=True)


def extract_arrival_specs(events: pd.DataFrame, meta: dict) -> list[ArrivalSpec]:
    """Pair each pass-ending possession with the next possession's start.

    See the module docstring for why the next possession's start, rather than
    the provider's stated reception point, defines the arrival.

    Coordinate frames: a trap worth stating
    ---------------------------------------
    **The event file and the tracking file do not share a coordinate frame.**
    Event coordinates are already *attack-normalised* - each event is expressed
    so the acting team plays toward ``+x`` - while the tracking is in a fixed
    match frame. Checked directly: for the team whose ``attacking_side`` is
    ``right_to_left``, event and tracking coordinates correlate at exactly
    ``-1.000``; for the other team, ``+1.000``.

    :func:`~pcc.data.tracking.assemble_arrivals` applies its own rotation into
    the canonical frame, so event coordinates must first be converted *back*
    into the match frame, or the normalisation is applied twice and one team's
    arrivals are mirrored every period. That failure is close to invisible in
    aggregate - it leaves base rates, pass lengths and flight times untouched
    and only shows up as a mean pass progression near zero instead of clearly
    positive.

    The per-event ``attacking_side`` column is used for the conversion, in
    preference to the per-period ``attack_sign`` derived from ``match.json``,
    and the two are cross-checked against each other.
    """
    pp = events[events["event_type"] == "player_possession"].reset_index(drop=True)
    if pp.empty:
        return []

    nxt = pd.DataFrame({
        "next_frame": pp["frame_start"].shift(-1),
        "next_x": pp["x_start"].shift(-1),
        "next_y": pp["y_start"].shift(-1),
        "next_period": pp["period"].shift(-1),
        # The NEXT possession's own attacking side. Essential: after a failed
        # pass the next possession belongs to the opponent, whose coordinates
        # are normalised to the opposite direction. Converting the destination
        # with the passer's sign mirrors every unsuccessful arrival.
        "next_side": pp["attacking_side"].shift(-1) if "attacking_side" in pp.columns else None,
    })
    pp = pd.concat([pp, nxt], axis=1)

    specs: list[ArrivalSpec] = []
    teams = [meta["home"], meta["away"]]
    mismatches: list[tuple] = []

    # Attacking direction implied by match.json, for cross-checking against the
    # per-event attacking_side column. Two independent sources agreeing is the
    # only cheap guard against a mirrored pitch.
    meta_sign: dict = {}
    for period_idx, side in enumerate(meta["home_side"], start=1):
        home_sign = 1.0 if side == "left_to_right" else -1.0
        meta_sign[(meta["home"], period_idx)] = home_sign
        meta_sign[(meta["away"], period_idx)] = -home_sign
    for _, row in pp.iterrows():
        if row.get("end_type") not in ARRIVAL_END_TYPES:
            continue
        if pd.isna(row["next_frame"]) or row["next_period"] != row["period"]:
            continue
        if any(pd.isna(row[c]) for c in ("x_end", "y_end", "next_x", "next_y", "frame_end")):
            continue

        flight = (float(row["next_frame"]) - float(row["frame_end"])) / SKILLCORNER_FPS
        if not (0.05 < flight < 6.0):
            continue

        team_a = str(row["team_shortname"])
        if team_a not in teams:
            continue
        team_b = teams[0] if team_a == teams[1] else teams[1]

        # Undo the event file's attack-normalisation so that assemble_arrivals
        # can apply the canonical rotation exactly once.
        side = row.get("attacking_side")
        if isinstance(side, str) and side in ("left_to_right", "right_to_left"):
            event_sign = 1.0 if side == "left_to_right" else -1.0
        else:
            event_sign = float(meta_sign.get((team_a, int(row["period"])), 1.0))
        expected = meta_sign.get((team_a, int(row["period"])))
        if expected is not None and abs(event_sign - expected) > 1e-9:
            mismatches.append((team_a, int(row["period"]), side, expected))

        next_side = row.get("next_side")
        if isinstance(next_side, str) and next_side in ("left_to_right", "right_to_left"):
            dest_sign = 1.0 if next_side == "left_to_right" else -1.0
        else:
            dest_sign = event_sign

        is_clearance = row.get("end_type") == "clearance"
        if is_clearance:
            arrival_type, endogenous = "clearance", False
        elif bool(row.get("high_pass")) or bool(row.get("is_header")):
            arrival_type, endogenous = "cross", True
        else:
            arrival_type, endogenous = "open_pass", True

        specs.append(
            ArrivalSpec(
                arrival_id=f"{meta['match_id']}_{row['event_id']}",
                period=int(row["period"]), team_a=team_a, team_b=team_b,
                t_release=float(row["frame_end"]) / SKILLCORNER_FPS,
                t_arrival=float(row["next_frame"]) / SKILLCORNER_FPS,
                origin=np.array([float(row["x_end"]), float(row["y_end"])]) * event_sign,
                destination=np.array([float(row["next_x"]), float(row["next_y"])]) * dest_sign,
                arrival_type=arrival_type, is_endogenous=endogenous,
                set_piece=bool(row.get("start_type") in SET_PIECE_START_TYPES),
                pass_height="high" if bool(row.get("high_pass")) else "unknown",
                possession_id=f"{meta['match_id']}_poss{int(row['index']):05d}"
                if pd.notna(row.get("index")) else "",
                source_event=f"{row.get('event_type')}|{row.get('end_type')}|{row.get('pass_outcome')}",
            )
        )

    if mismatches:
        import warnings

        warnings.warn(
            f"{len(mismatches)} events whose attacking_side disagrees with match.json's "
            f"home_team_side (e.g. {mismatches[:3]}). One of the two sources is wrong; "
            "resolve it before trusting any spatial result from this match.",
            stacklevel=2,
        )
    return specs


def available_matches(root: Path) -> list[Path]:
    """Match directories whose tracking file is present and LFS-resolved."""
    matches_dir = Path(root) / "data" / "matches"
    if not matches_dir.is_dir():
        return []
    out = []
    for d in sorted(p for p in matches_dir.iterdir() if p.is_dir()):
        track = d / f"{d.name}_tracking_extrapolated.jsonl"
        if not track.exists():
            continue
        try:
            _check_not_lfs_pointer(track)
        except LFSPointerError:
            continue
        if (d / f"{d.name}_dynamic_events.csv").exists():
            out.append(d)
    return out


def load_match(match_dir: Path, *, label_config: LabelConfig | None = None, **tracking_kwargs):
    """Load one SkillCorner match into frames and an arrivals table."""
    meta = load_match_meta(match_dir)
    tracking, possession = load_tracking(match_dir, meta, **tracking_kwargs)
    events = load_events(match_dir)
    specs = extract_arrival_specs(events, meta)

    # No explicit dead-ball flag in the structural columns, so stoppages are
    # taken as frames with no possession attributed. That is conservative: it
    # censors some live loose-ball moments too, and the censored fraction is
    # reported so the cost is visible rather than assumed away.
    stoppage = possession.isna().to_numpy()

    frames, table = assemble_arrivals(
        tracking, specs, possession, stoppage,
        match_id=meta["match_id"], competition=meta["competition"],
        tracking_source="broadcast",
        provider="skillcorner_open (extrapolated tracking; is_detected retained)",
        label_config=label_config, match_date=meta.get("date"),
    )
    if not table.empty:
        table["detection_rate_match"] = tracking.meta["detection_rate"]
    return frames, table
