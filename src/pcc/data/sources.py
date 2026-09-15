"""Registry of candidate data sources, with availability treated as unverified.

Read this before writing a data section
---------------------------------------
The entries below record what each source is *claimed* to provide, what the
study would need from it, and - critically - whether that claim has been
**verified by this repository**. Nothing here should be cited in a paper as
established fact until ``scripts/01_check_data_availability.py`` has been run
and its output committed to ``results/data_availability.json``.

Two failure modes this registry exists to prevent:

* Asserting in a proposal that a dataset contains ball tracking, velocities or
  possession labels when it has not been opened. Provider documentation is
  frequently out of date, and derived fields (velocity in particular) are often
  absent even when positions are present.
* Assuming that "publicly downloadable" means "licensed for redistribution or
  for the intended use". Several football datasets are free to download under
  terms that restrict commercial use, redistribution, or both. Licence terms
  must be read for each source and recorded in ``licence_note``.

``status`` values
-----------------
``unverified``
    Listed from documentation or common knowledge; not checked here.
``verified_reachable``
    ``scripts/01_check_data_availability.py`` reached the endpoint in this
    environment. This says nothing about contents or licence.
``verified_contents``
    A sample was loaded and the field inventory in ``provides`` was confirmed.
``unavailable``
    Checked and not reachable, or access denied.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

#: Capability vocabulary used by ``provides`` / ``required``.
CAPABILITIES = (
    "player_tracking",       # continuous x,y for all outfield players
    "ball_tracking",         # continuous x,y (and ideally z) for the ball
    "event_data",            # timestamped on-ball events
    "pass_start_coords",
    "pass_end_coords",
    "pass_arrival_time",     # measured, not imputed
    "possession_outcome",    # who has the ball, frame by frame or event by event
    "player_velocity",       # supplied by the provider (as opposed to derived here)
    "freeze_frames",         # player positions at the moment of an event only
    "match_metadata",        # teams, date, competition, pitch dimensions
)


@dataclass
class DataSource:
    key: str
    name: str
    modality: str                       # 'optical' | 'broadcast' | 'event' | 'simulated'
    provides: tuple[str, ...]
    status: str = "unverified"
    url: str | None = None
    #: A small, data-bearing endpoint used for reachability checks. GitHub's HTML
    #: pages are blocked by many corporate and sandbox proxies while raw content
    #: is not, so probing the landing page produces false negatives.
    probe_url: str | None = None
    access: str = "unknown"             # 'open_download' | 'registration' | 'request' | 'commercial'
    licence_note: str = "Licence not verified by this repository; read the terms before use."
    coverage_note: str = ""
    study_role: str = ""
    caveats: tuple[str, ...] = ()
    loader: str | None = None           # name of the function in pcc.data.loaders
    verified_fields: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def missing(self, required: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(r for r in required if r not in self.provides)


#: What the *primary* analysis needs from a source to be usable on its own.
PRIMARY_REQUIREMENTS: tuple[str, ...] = (
    "player_tracking",
    "ball_tracking",
    "event_data",
    "pass_start_coords",
    "pass_end_coords",
    "possession_outcome",
    "match_metadata",
)

#: What the *fallback* (freeze-frame) analysis needs. Weaker, and the weakening
#: is itself informative: without velocity the physics models lose their main
#: input, which turns the fallback study into a large-sample version of the
#: "remove velocity" ablation.
FALLBACK_REQUIREMENTS: tuple[str, ...] = (
    "freeze_frames",
    "event_data",
    "pass_start_coords",
    "pass_end_coords",
    "match_metadata",
)


SOURCES: dict[str, DataSource] = {
    "pff_wc2022": DataSource(
        key="pff_wc2022",
        name="PFF FC 2022 FIFA World Cup tracking and event release",
        modality="optical",
        provides=(
            "player_tracking", "ball_tracking", "event_data",
            "pass_start_coords", "pass_end_coords", "possession_outcome", "match_metadata",
        ),
        status="unverified",
        access="request",
        study_role=(
            "The intended primary corpus: a single tournament, one provider, "
            "consistent optical tracking, enough matches for match-clustered "
            "intervals of usable width."
        ),
        caveats=(
            "Access route, current availability and licence terms are NOT verified by "
            "this repository. Treat every property as a hypothesis until checked.",
            "Sampling rate, whether ball height is included, and whether a frame-level "
            "possession label is supplied all materially affect the labelling step and "
            "must be confirmed before the design is fixed.",
            "A single tournament is one competition: it cannot support the "
            "leave-one-competition-out split on its own.",
        ),
        loader="load_pff",
    ),
    "metrica_sample": DataSource(
        key="metrica_sample",
        name="Metrica Sports sample tracking and event data",
        modality="optical",
        provides=(
            "player_tracking", "ball_tracking", "event_data",
            "pass_start_coords", "pass_end_coords", "pass_arrival_time",
            "possession_outcome", "match_metadata",
        ),
        status="verified_contents",
        url="https://github.com/metrica-sports/sample-data",
        probe_url="https://raw.githubusercontent.com/metrica-sports/sample-data/master/README.md",
        access="open_download",
        licence_note=(
            "The repository asks that use be responsible and that the source be "
            "acknowledged if anything is made public. Read its README and any linked "
            "terms before publishing; this note is not a substitute for them."
        ),
        coverage_note=(
            "Three matches. Two (Sample_Game_1, Sample_Game_2) are in the CSV layout "
            "read by pcc.data.metrica; Sample_Game_3 is in the EPTS/FIFA format and is "
            "not read. 25 fps, ~145,000 frames per match, 105 x 68 m, anonymised."
        ),
        study_role=(
            "Development and pipeline-validation corpus. Confirmed adequate for "
            "building and testing the loader end to end; NOT adequate for inference."
        ),
        caveats=(
            "VERIFIED TOO SMALL: two readable matches means two bootstrap clusters. "
            "No confidence interval computed on it is meaningful, and a three-way "
            "match-level split is impossible, so RQ5 (recalibration) cannot be run.",
            "VERIFIED: no frame-level possession label. The outcome is DERIVED from "
            "the event stream (a paired BALL LOST / RECOVERY transfer log). Every row "
            "records this in its provider field.",
            "VERIFIED: tracking and events are already synchronised (events carry "
            "frame numbers), so no clock-offset estimation is needed - unusual, and it "
            "removes one of the pipeline's larger error sources.",
            "VERIFIED: coordinates are normalised to [0,1] with (0,0) at the TOP LEFT, "
            "so the y-axis increases downward and must be negated.",
            "VERIFIED: anonymised, with no position labels, so goalkeepers must be "
            "inferred from position.",
        ),
        loader="load_metrica",
        verified_fields={
            "checked_on": "2026-09-13",
            "commit": "e706dd506b360d69d9d123d5b8026e7294b13996",
            "fps": 25.0,
            "pitch_m": [105.0, 68.0],
            "readable_matches": ["Sample_Game_1", "Sample_Game_2"],
            "unreadable_matches": {"Sample_Game_3": "EPTS/FIFA format; needs kloppy"},
            "frames_per_match": 145006,
            "event_types": {
                "PASS": [799, 964], "RECOVERY": [278, 248], "BALL LOST": [257, 233],
                "CHALLENGE": [233, 311], "SET PIECE": [77, 80], "BALL OUT": [51, 49],
                "SHOT": [24, 24], "FAULT RECEIVED": [22, 20], "CARD": [4, 6],
            },
            "arrivals_extracted": 2046,
            "arrivals_after_filters": 1771,
            "ball_z_present": False,
            "provider_velocity": False,
            "frame_level_possession": False,
        },
    ),
    "skillcorner_open": DataSource(
        key="skillcorner_open",
        name="SkillCorner open broadcast-derived tracking",
        modality="broadcast",
        provides=(
            "player_tracking", "ball_tracking", "event_data",
            "possession_outcome", "match_metadata",
        ),
        status="verified_reachable",
        url="https://github.com/SkillCorner/opendata",
        probe_url="https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches.json",
        access="open_download",
        licence_note=(
            "Released jointly by SkillCorner and PySport; the repository asks that "
            "SkillCorner be credited if the data is used. Read its README before "
            "publishing."
        ),
        coverage_note=(
            "20 match directories, Australian A-League 2024/25 (the README describes "
            "10 matches; the directory count differs and should be reconciled). Per "
            "match: tracking (JSONL), a rich 'dynamic events' CSV, a phases-of-play "
            "CSV and a match JSON. 3D body pose is included for two matches."
        ),
        study_role=(
            "The broadcast-tracking arm of RQ4. Its value is precisely its "
            "imperfection: off-camera players are missing, which is the realistic "
            "data condition for most clubs outside the elite tier."
        ),
        caveats=(
            "VERIFIED: tracking files are stored in Git LFS. A plain `git clone` "
            "yields ~130-byte pointer stubs, NOT data. Fetch the real bytes from "
            "media.githubusercontent.com/media/... (scripts/fetch_data.py does this) "
            "and check the result is not a pointer.",
            "VERIFIED: the tracking JSONL carries a per-frame `possession` object "
            "with player_id and group - a frame-level possession label, which the "
            "study's earlier plan listed as unknown for this source.",
            "VERIFIED: frames carry `image_corners_projection`, and early frames have "
            "null ball and empty player_data - off-camera and pre-kickoff periods are "
            "explicitly represented rather than silently absent.",
            "The 'dynamic events' file is a derived-metrics product carrying "
            "SkillCorner's own xpass_completion, EPV and pressure measures. Using "
            "those as inputs would contaminate the comparison; only raw positional "
            "and outcome fields may be used.",
            "Comparing calibration against optical tracking confounds measurement "
            "quality with the different matches and competitions covered; the "
            "degradation-simulation ablation is required to separate the two.",
        ),
        loader="load_skillcorner",
        verified_fields={
            "checked_on": "2026-09-13",
            "match_directories": 20,
            "competition": "Australian A-League 2024/25",
            "tracking_storage": "git-lfs",
            "tracking_bytes_per_match": 90729279,
            "frame_level_possession": True,
            "bodypose_matches": 2,
            "files_per_match": [
                "{id}_match.json", "{id}_tracking_extrapolated.jsonl",
                "{id}_dynamic_events.csv", "{id}_phases_of_play.csv",
            ],
        },
    ),
    "statsbomb_open": DataSource(
        key="statsbomb_open",
        name="StatsBomb Open Data (events; 360 freeze frames for some competitions)",
        modality="event",
        provides=(
            "event_data", "pass_start_coords", "pass_end_coords",
            "possession_outcome", "freeze_frames", "match_metadata",
        ),
        status="verified_contents",
        url="https://github.com/statsbomb/open-data",
        probe_url="https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json",
        access="open_download",
        licence_note=(
            "Governed by StatsBomb's own user agreement, held in the repository. It "
            "restricts redistribution and commercial use and must be read before any "
            "data is used or any derived data is published."
        ),
        coverage_note=(
            "80 competition-seasons, of which 12 carry 360 freeze frames - including "
            "FIFA World Cup 2022 (competition_id 43, season_id 106), the same "
            "tournament as the PFF optical release. That coincidence makes the "
            "fallback design a same-tournament comparison rather than a different one."
        ),
        study_role=(
            "Fallback corpus. Freeze frames give player positions at the event but no "
            "continuous tracking and hence no velocity, so only velocity-free variants "
            "of the control models can be evaluated - a genuinely weaker but much "
            "larger-sample study."
        ),
        caveats=(
            "VERIFIED: no continuous tracking and no velocity. Every model runs in "
            "its zero-velocity form, so the fallback MEASURES what the 'remove "
            "velocity' ablation only simulates.",
            "CORRECTION (verified): arrival times do NOT have to be imputed. Every "
            "pass carries a `duration`, with implied ball speeds of 6.9-22.1 m/s "
            "(median 13.1). Earlier planning assumed a ball-speed model would be "
            "needed; it is not. Caveat: `duration` runs to the related event, which "
            "is the flight time for a completed pass and an approximation otherwise.",
            "VERIFIED: 360 frames cover 86.6% of passes; the median frame shows 17 "
            "of 22 players and NO frame shows all 22. Frame completeness is a "
            "first-class covariate, not a nuisance.",
            "VERIFIED: 16.4% of pass destinations fall OUTSIDE the visible_area "
            "polygon. For those arrivals 'no defender near the destination' means "
            "'no defender visible'. The adapter flags them destination_not_visible.",
            "VERIFIED: only Pass events carry both a start and an end location, so "
            "the corpus contains NO exogenous arrivals at all. The quasi-exogenous "
            "identification argument of proposal Section 14 is unavailable here.",
            "VERIFIED: each 360 frame carries a `visible_area` polygon and a "
            "`freeze_frame` list of visible players only. 'No defender near the "
            "destination' can therefore mean 'no defender VISIBLE', which would bias "
            "control estimates upward exactly where it matters. Any adapter MUST use "
            "visible_area to mark destinations outside the covered region.",
            "VERIFIED: freeze-frame players carry teammate/actor/keeper flags and a "
            "location, but no identity, so per-player sprint-speed estimation is "
            "impossible.",
            "VERIFIED: only 12 of 80 competition-seasons have 360 frames; that is the "
            "binding constraint on the fallback design's sample size.",
        ),
        loader="load_statsbomb",
        verified_fields={
            "checked_on": "2026-09-13",
            "competition_seasons": 80,
            "with_360": 12,
            "world_cup_2022": {"competition_id": 43, "season_id": 106, "matches": 64},
            "frame_keys": ["event_uuid", "visible_area", "freeze_frame"],
            "freeze_frame_player_keys": ["teammate", "actor", "keeper", "location"],
            "provider_velocity": False,
            "player_identity_in_freeze_frame": False,
            "pass_duration_present": True,
            "implied_ball_speed_ms": {"p5": 6.9, "p50": 13.1, "p95": 22.1},
            "pass_360_coverage": 0.866,
            "median_players_visible": 17,
            "frames_with_all_22": 0,
            "destination_inside_visible_area": 0.836,
            "exogenous_arrival_types_available": [],
            "arrivals_8_matches": 7284,
            "base_rate_control_h1s": 0.90,
            "label_agreement_with_completed_passes": 0.966,
        },
    ),
    "simulated": DataSource(
        key="simulated",
        name="pcc.data.synthetic simulator",
        modality="simulated",
        provides=(
            "player_tracking", "ball_tracking", "event_data", "pass_start_coords",
            "pass_end_coords", "pass_arrival_time", "possession_outcome",
            "player_velocity", "match_metadata",
        ),
        status="verified_contents",
        access="open_download",
        licence_note="Generated by this repository; no third-party licence applies.",
        study_role=(
            "Instrument validation and power analysis only. Never a source of "
            "empirical claims about football."
        ),
        caveats=("Fabricated. Any result from it describes the simulator, not football.",),
        loader="load_simulated",
    ),
}


def summarise_sources() -> "list[dict]":
    """Tabular summary for ``docs/data_sources.md`` and the availability report."""
    rows = []
    for src in SOURCES.values():
        rows.append(
            {
                "key": src.key,
                "name": src.name,
                "modality": src.modality,
                "status": src.status,
                "access": src.access,
                "player_tracking": "player_tracking" in src.provides,
                "ball_tracking": "ball_tracking" in src.provides,
                "event_data": "event_data" in src.provides,
                "freeze_frames": "freeze_frames" in src.provides,
                "provider_velocity": "player_velocity" in src.provides,
                "possession_outcome": "possession_outcome" in src.provides,
                "meets_primary_requirements": not src.missing(PRIMARY_REQUIREMENTS),
                "missing_for_primary": ", ".join(src.missing(PRIMARY_REQUIREMENTS)),
                "meets_fallback_requirements": not src.missing(FALLBACK_REQUIREMENTS),
            }
        )
    return rows
