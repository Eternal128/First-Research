"""Provider adapters.

Every loader returns the same pair: ``(frames, arrivals_table)`` conforming to
:mod:`pcc.data.schema`. Provider-specific parsing is quarantined here so that
nothing downstream needs to know which company produced the data - which is
what makes the optical-versus-broadcast comparison a controlled contrast rather
than a comparison of two codebases.

Status of these adapters
------------------------
Only :func:`load_simulated` is complete and tested, because it is the only
source this repository can access without third-party data. The provider
adapters are **scaffolds**: they encode the mapping the study intends to use,
raise an explicit, actionable error when the data is absent, and are structured
so that filling them in is a contained task once access is arranged. They are
deliberately not written to look finished. Each carries a ``TODO(access)``
marker naming exactly what must be confirmed against the real files.

The preferred implementation route for all three is ``kloppy``, which already
provides provider-agnostic deserialisers and a common coordinate model; writing
bespoke parsers would duplicate maintained work and introduce a second source
of coordinate-convention bugs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pcc.data.preprocess import PreprocessConfig, Provenance
from pcc.data.schema import ArrivalFrame
from pcc.data.sources import SOURCES


class DataNotAvailable(RuntimeError):
    """Raised when a source is not present locally, with instructions attached."""


def _not_available(key: str, path: Path | None = None) -> "DataNotAvailable":
    src = SOURCES[key]
    lines = [
        f"Data source {src.name!r} is not available at {path or '<no path given>'}.",
        "",
        f"  status in registry : {src.status}",
        f"  access route       : {src.access}",
        f"  url                : {src.url or 'not recorded'}",
        f"  licence            : {src.licence_note}",
        "",
        "This repository does not download provider data automatically. Obtain the",
        "data under its own terms, place it under data/raw/<key>/, and re-run.",
        "Run scripts/01_check_data_availability.py to record what is present.",
    ]
    return DataNotAvailable("\n".join(lines))


def require_kloppy():
    """Import kloppy with an actionable message if it is missing."""
    try:
        import kloppy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "kloppy is required for provider loaders. Install with "
            "`pip install kloppy`, or use load_simulated() to exercise the pipeline."
        ) from exc
    import kloppy

    return kloppy


# ---------------------------------------------------------------------------
# Simulated (complete)
# ---------------------------------------------------------------------------
def load_simulated(**kwargs) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """Generate a simulated corpus. See :mod:`pcc.data.synthetic` for the caveats."""
    from pcc.data.synthetic import SimulationConfig, simulate_dataset

    cfg = SimulationConfig(**kwargs) if kwargs else SimulationConfig()
    return simulate_dataset(cfg)


# ---------------------------------------------------------------------------
# Provider scaffolds
# ---------------------------------------------------------------------------
def load_metrica(
    root: str | Path, *, config: PreprocessConfig | None = None, match_ids: list[str] | None = None
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """Metrica Sports sample data.

    TODO(access): confirm against the real files before relying on any of this.
      1. Which matches are in the normalised CSV layout and which are in the
         EPTS/FIFA format - they need different deserialisers.
      2. Coordinate convention and origin, and whether the y-axis is inverted.
      3. Frame rate per match (do not assume it is uniform across matches).
      4. Whether a frame-level possession label exists. If not, outcomes must be
         derived from touch events and ``LabelConfig.definition`` set to
         ``first_touch`` with the substitution documented per match.
      5. Whether ball z-coordinate is present; without it, aerial and ground
         arrivals cannot be separated and ``pass_height`` stays ``unknown``.

    Implementation sketch (kloppy)::

        from kloppy import metrica
        ds = metrica.load_tracking_csv(home_data=..., away_data=...)
        events = metrica.load_event_csv(...)

    then feed both to :func:`build_arrivals`.
    """
    root = Path(root)
    if not root.exists():
        raise _not_available("metrica_sample", root)
    raise NotImplementedError(
        "load_metrica is a scaffold. Complete it against the real files following the "
        "TODO(access) checklist in its docstring, then remove this guard."
    )


def load_skillcorner(
    root: str | Path, *, config: PreprocessConfig | None = None, match_ids: list[str] | None = None
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """SkillCorner open broadcast-derived tracking.

    TODO(access): confirm before use.
      1. How unobserved (off-camera) players are represented - missing rows,
         null coordinates, or an explicit visibility flag. This determines the
         ``att_observed`` / ``def_observed`` masks, which drive the RQ4 analysis.
      2. Whether the ball track has the same gaps as the player tracks.
      3. Frame rate and whether it is constant.
      4. Which event stream, if any, accompanies the tracking; if none, arrivals
         must be detected from the ball track itself (a change in ball
         acceleration plus proximity to a player), which is a separate and
         error-prone step that needs its own validation against a hand-labelled
         sample.

    Note: with broadcast tracking, ``frame_completeness`` must be computed per
    arrival and carried through. Imputing the missing players and then reporting
    a control probability as though all 22 were observed is exactly the practice
    this study is meant to scrutinise.
    """
    root = Path(root)
    if not root.exists():
        raise _not_available("skillcorner_open", root)
    raise NotImplementedError(
        "load_skillcorner is a scaffold. Complete it against the real files following "
        "the TODO(access) checklist in its docstring, then remove this guard."
    )


def load_statsbomb(
    root: str | Path, *, config: PreprocessConfig | None = None, competitions: list[int] | None = None
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """StatsBomb Open Data, used for the freeze-frame fallback design.

    TODO(access): confirm before use.
      1. Which competitions carry 360 freeze frames - this is the binding
         constraint on the fallback design's sample size.
      2. The freeze-frame coordinate frame and its relation to the event frame.
      3. The visible-area polygon: freeze frames cover only part of the pitch,
         so "no defender near the destination" may mean "no defender visible",
         which would bias control estimates upward exactly where it matters.
      4. The licence terms governing use and redistribution.

    Design consequence, stated plainly: freeze frames are a single snapshot,
    so there is **no velocity**. The physics-based models can only be run in
    their zero-velocity form, and pass arrival time must be imputed from a
    ball-speed model. The fallback study is therefore not a substitute for the
    tracking study; it answers a related, weaker question on a larger sample,
    and the paper must present it as such.
    """
    root = Path(root)
    if not root.exists():
        raise _not_available("statsbomb_open", root)
    raise NotImplementedError(
        "load_statsbomb is a scaffold. Complete it against the real files following "
        "the TODO(access) checklist in its docstring, then remove this guard."
    )


def load_pff(
    root: str | Path, *, config: PreprocessConfig | None = None, match_ids: list[str] | None = None
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """PFF FC 2022 World Cup release.

    TODO(access): nothing about this source is verified by this repository.
      1. Confirm the access route, the current availability, and the licence.
      2. Confirm the file layout, coordinate frame, frame rate and whether
         event and tracking clocks are already synchronised.
      3. Confirm whether a frame-level possession label is supplied.
      4. Confirm whether ball height is present.

    Until (1) is settled, the study plan must carry the fallback design as a
    live option rather than a contingency, and the proposal says so.
    """
    root = Path(root)
    if not root.exists():
        raise _not_available("pff_wc2022", root)
    raise NotImplementedError(
        "load_pff is a scaffold. Complete it against the real files following the "
        "TODO(access) checklist in its docstring, then remove this guard."
    )


# ---------------------------------------------------------------------------
# Shared assembly step
# ---------------------------------------------------------------------------
def build_arrivals(
    tracking: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: PreprocessConfig,
    match_id: str,
    competition: str,
    tracking_source: str,
    provider: str,
    provenance: Provenance | None = None,
) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """Assemble arrival frames and the arrivals table from prepared streams.

    This is the provider-independent half of every loader and is where the
    study's definitions are actually applied. Expected inputs:

    ``tracking``
        Long frame with ``timestamp, player_id, team, x, y, vx, vy, occluded,
        is_gk`` in the canonical frame (the output of
        :func:`~pcc.data.preprocess.prepare_tracking`), plus ball rows with
        ``player_id == 'ball'``.
    ``events``
        Event table with at least ``event_id, type, team, timestamp,
        end_timestamp, start_x, start_y, end_x, end_y``.

    Steps, in order:

    1. Locate each ball arrival (pass reception, interception, deflection).
    2. Take the player state at ``t_release``, never at ``t_arrival``.
    3. Compute the flight time from the ball track where possible and flag it
       as imputed where not.
    4. Label the outcome with :func:`~pcc.data.labels.label_from_possession_track`.
    5. Classify the arrival type and set ``is_endogenous``.

    Left unimplemented here on purpose: step 1 is genuinely provider-specific
    and getting it wrong silently corrupts the quasi-exogenous subsample, so it
    belongs in each adapter where it can be validated against that provider's
    own event definitions.
    """
    raise NotImplementedError(
        "build_arrivals is the shared assembly step and is completed alongside the "
        "first real provider adapter; see its docstring for the required inputs."
    )


LOADERS = {
    "simulated": load_simulated,
    "metrica_sample": load_metrica,
    "skillcorner_open": load_skillcorner,
    "statsbomb_open": load_statsbomb,
    "pff_wc2022": load_pff,
}


def load_source(key: str, **kwargs) -> tuple[list[ArrivalFrame], pd.DataFrame]:
    """Dispatch to a loader by registry key."""
    if key not in LOADERS:
        raise KeyError(f"unknown source {key!r}; available: {sorted(LOADERS)}")
    return LOADERS[key](**kwargs)
