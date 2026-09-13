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
            "pass_start_coords", "pass_end_coords", "match_metadata",
        ),
        status="unverified",
        url="https://github.com/metrica-sports/sample-data",
        access="open_download",
        study_role=(
            "Development and pipeline-validation corpus. Small, well documented, "
            "widely used in teaching material, so the preprocessing can be checked "
            "against community implementations."
        ),
        caveats=(
            "Very small. Not adequate on its own for the primary analysis: with a "
            "handful of matches the match-clustered bootstrap has too few clusters "
            "for the intervals to mean much.",
            "Whether a frame-level possession label exists must be checked; if not, "
            "outcomes have to be derived from touch events, which changes the "
            "construct slightly and must be documented.",
        ),
        loader="load_metrica",
    ),
    "skillcorner_open": DataSource(
        key="skillcorner_open",
        name="SkillCorner open broadcast-derived tracking",
        modality="broadcast",
        provides=("player_tracking", "ball_tracking", "event_data", "match_metadata"),
        status="unverified",
        url="https://github.com/SkillCorner/opendata",
        access="open_download",
        study_role=(
            "The broadcast-tracking arm of RQ4. Its value to this study is precisely "
            "its imperfection: off-camera players are missing, so it is the realistic "
            "data condition for most clubs outside the elite tier."
        ),
        caveats=(
            "Broadcast tracking omits players outside the camera frame. Frame "
            "completeness is therefore a first-class covariate, not a nuisance.",
            "Comparing calibration between this and optical tracking confounds "
            "measurement quality with the fact that they cover different matches and "
            "competitions; the degradation-simulation ablation is required to "
            "separate the two.",
            "Number of matches, sampling rate and event coverage must be verified.",
        ),
        loader="load_skillcorner",
    ),
    "statsbomb_open": DataSource(
        key="statsbomb_open",
        name="StatsBomb Open Data (events; 360 freeze frames for some competitions)",
        modality="event",
        provides=(
            "event_data", "pass_start_coords", "pass_end_coords",
            "possession_outcome", "freeze_frames", "match_metadata",
        ),
        status="unverified",
        url="https://github.com/statsbomb/open-data",
        access="open_download",
        study_role=(
            "Fallback corpus. Freeze frames give player positions at the event but "
            "no continuous tracking and hence no velocity, so only velocity-free "
            "variants of the control models can be evaluated - a genuinely weaker "
            "but much larger-sample study."
        ),
        caveats=(
            "No continuous tracking: arrival times must be imputed from a ball-speed "
            "model, which makes the flight time partly a modelling artefact.",
            "360 freeze frames cover only some competitions and only show players "
            "within the camera's visible area; the covered set must be verified.",
            "Redistribution and permitted use are governed by StatsBomb's own user "
            "agreement, which must be read before any data is committed anywhere.",
        ),
        loader="load_statsbomb",
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
