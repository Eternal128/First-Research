# Data sources: status, requirements, and verification checklists

> **The rule this document exists to enforce.** Nothing in this repository, and
> nothing in any write-up derived from it, may assert what a dataset contains
> until the files have been opened and the claim checked. Provider documentation
> goes stale, derived fields (velocity above all) are frequently absent even when
> positions are present, and "freely downloadable" does not mean "licensed for
> the intended use".
>
> Run `python scripts/01_check_data_availability.py` and commit
> `results/data_availability.json` before writing a data section.

---

## 1. What the study needs

### Primary design

Continuous player **and** ball tracking, synchronised with an event stream, for
enough matches that a match-clustered bootstrap has enough clusters to support
inference, with a derivable frame-level or event-level possession label.

Required fields: `match_id`, `timestamp`, `period`, `player_id`, `team`,
`is_goalkeeper`, player `x`/`y`, ball `x`/`y`, `event_type`, release and arrival
coordinates and times, possession. Velocity is **derived here**, not required
from the provider.

### Fallback design (freeze frames)

Event data with per-event freeze frames giving player positions at the moment of
the event. No continuous tracking means **no velocity** and **no measured arrival
time**. Only velocity-free variants of the control models can be evaluated, and
flight time must be imputed from a ball-speed model.

This is a genuinely weaker study on a larger sample, and the paper must present
it as such. Its one compensating virtue: the weakening coincides exactly with the
"remove velocity" ablation, so the fallback *measures* what the ablation only
simulates.

---

## 2. Status of each candidate source

Status vocabulary: `unverified` (listed from documentation, not checked);
`verified_reachable` (endpoint responded here — says nothing about contents or
licence); `verified_contents` (a sample was loaded and the field inventory
confirmed); `unavailable` (checked, not reachable or access denied).

| Source | Modality | Status in this repository | Access | Role |
|---|---|---|---|---|
| PFF FC 2022 World Cup release | optical | `unverified` | request | intended primary corpus |
| Metrica Sports sample data | optical | `unverified` | open download | development, pipeline validation |
| SkillCorner open data | broadcast | `unverified` | open download | broadcast arm of RQ4 |
| StatsBomb Open Data | event (+ freeze frames for some competitions) | `unverified` | open download | fallback design |
| `pcc.data.synthetic` | simulated | `verified_contents` | generated here | instrument validation, power analysis |

### Network note

In the environment where this repository was developed, outbound HTTP to these
hosts returns **HTTP 403 from an egress proxy**. That is a fact about the
network, not about the datasets. `scripts/01_check_data_availability.py` labels
such responses `BLOCKED BY LOCAL NETWORK POLICY` and marks the result
inconclusive, because recording a proxy block as "dataset unavailable" is exactly
the unverified claim this document forbids.

---

## 3. Per-source verification checklists

Work through the checklist for whichever source is obtained, record the answers,
and update `src/pcc/data/sources.py` with the verified status.

### 3.1 PFF FC 2022 World Cup release

**Nothing about this source is verified here.** Before the design is fixed:

- [ ] Confirm the current access route and whether it is still available.
- [ ] Read the licence. Record permitted use and whether derived per-arrival
      forecasts may be published.
- [ ] Confirm the file layout and the coordinate frame (origin, axis directions,
      units, whether normalised or metric).
- [ ] Confirm the tracking frame rate, and whether it is constant across matches.
- [ ] Confirm whether event and tracking clocks are pre-synchronised. If not,
      estimate the per-match offset and record the residual.
- [ ] Confirm whether a **frame-level possession label** is supplied. If not, the
      outcome must be derived from touch events, which changes the construct and
      must be documented per match.
- [ ] Confirm whether ball **height** is present (aerial versus ground arrivals).
- [ ] Confirm whether true per-match pitch dimensions are recorded.
- [ ] Map the provider's event taxonomy onto the study's arrival typology, and
      **spot-check the exogenous classes against video**. Getting this wrong
      contaminates the quasi-exogenous subsample, which is the study's strongest
      identification argument.
- [ ] Count arrivals by type. Confirm the exogenous subsample is large enough to
      support the identification argument — a sample-size question to settle
      *before* committing to the design.

**Design consequence already known:** a single tournament is a single
competition, so this corpus alone cannot support leave-one-competition-out
validation.

### 3.2 Metrica Sports sample data

- [ ] Identify which matches are in the normalised CSV layout and which are in
      the EPTS/FIFA format; they need different deserialisers.
- [ ] Confirm the coordinate convention, origin, and whether the y-axis is
      inverted.
- [ ] Confirm the frame rate **per match** — do not assume uniformity.
- [ ] Confirm whether a frame-level possession label exists.
- [ ] Confirm whether a ball z-coordinate is present.
- [ ] Read the licence.

**Known design consequence:** this is a small sample. With very few matches the
match-clustered bootstrap has too few clusters for intervals to mean much, so it
is a development and pipeline-validation corpus, not a primary one.

### 3.3 SkillCorner open data

- [ ] Confirm **how unobserved (off-camera) players are represented**: missing
      rows, null coordinates, or an explicit visibility flag. This determines the
      `att_observed`/`def_observed` masks that drive the whole RQ4 analysis.
- [ ] Confirm whether the ball track has the same gaps as the player tracks.
- [ ] Confirm the frame rate and whether it is constant.
- [ ] Confirm which event stream, if any, accompanies the tracking. If none,
      arrivals must be detected from the ball track itself — a separate,
      error-prone step needing its own validation against a hand-labelled sample.
- [ ] Read the licence.

**Methodological warning.** Imputing missing players and then reporting a control
probability as though all twenty-two were observed is precisely the practice this
study exists to scrutinise. `frame_completeness` must be computed per arrival and
carried through as a covariate.

### 3.4 StatsBomb Open Data

- [ ] Confirm **which competitions carry 360 freeze frames** — the binding
      constraint on the fallback design's sample size.
- [ ] Confirm the freeze-frame coordinate frame and its relation to the event
      frame.
- [ ] Confirm the **visible-area polygon**. Freeze frames cover only part of the
      pitch, so "no defender near the destination" may mean "no defender
      visible", which would bias control estimates upward exactly where it
      matters most.
- [ ] Read the user agreement governing use and redistribution.

**Design consequences already known:** no velocity; imputed arrival times; the
physics models run only in their zero-velocity form.

---

## 4. Licensing and ethics

- **No provider data is committed to this repository.** `data/raw/` is
  git-ignored. Place obtained data under `data/raw/<source_key>/`.
- **Licence terms govern.** Read them per source and record the terms in
  `sources.py`. Several football datasets are free to download under terms
  restricting commercial use, redistribution, or both.
- **Personal data.** Positional tracking of identifiable individuals is personal
  data under several regimes, even though the performance is public. The study
  reports team-level and aggregate results, does not publish individual-level
  derived performance measures, and does not redistribute raw tracking.
- **What can be released** when the data cannot: all code; all derived aggregated
  results (metric tables, reliability-curve coordinates, subgroup tables,
  ablation grids); the simulator; the preprocessing provenance and leakage audit;
  a schema-conformant synthetic replica matched to the real corpus's marginals,
  clearly labelled as synthetic. Per-arrival forecasts and outcomes for the test
  fold would let a third party recompute every calibration statistic
  independently — worth requesting from the provider explicitly.

---

## 5. Completing a loader

The provider adapters in `src/pcc/data/loaders.py` are **scaffolds**, marked as
such, with a `TODO(access)` checklist in each docstring. They raise an explicit,
actionable error rather than pretending to work.

The recommended route for all three is `kloppy`, which already supplies
provider-agnostic deserialisers and a common coordinate model; writing bespoke
parsers duplicates maintained work and introduces a second source of
coordinate-convention bugs.

Each loader must return `(frames, arrivals_table)` conforming to
`pcc.data.schema`:

1. Locate each ball arrival (reception, interception, deflection, clearance,
   second ball). **Provider-specific and the highest-risk step.**
2. Take the player state at `t_release`, never at `t_arrival` — using
   arrival-time positions leaks the outcome, since defenders converge *because*
   the ball is arriving.
3. Compute flight time from the ball track where possible; flag it as imputed
   where not.
4. Label the outcome with `pcc.data.labels.label_from_possession_track`.
5. Classify the arrival type and set `is_endogenous`.
6. Run `validate_arrivals` and fix anything it reports rather than passing
   `strict=False` and moving on.
