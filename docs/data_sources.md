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
the event. No continuous tracking means **no velocity**, so only velocity-free
variants of the control models can be evaluated.

Arrival time, however, **is measured**: StatsBomb supplies a `duration` for every
pass. Earlier planning assumed a ball-speed model would be needed; verification
showed it is not. That is one fewer modelling artefact in the fallback than
expected.

This is a genuinely weaker study on a larger sample, and the paper must present
it as such. Its one compensating virtue: the weakening coincides exactly with the
"remove velocity" ablation, so the fallback *measures* what the ablation only
simulates.

---

## 2. Status of each candidate source

Status vocabulary: `unverified` (listed from documentation, not checked);
`verified_reachable` (a data-bearing endpoint responded — says nothing about
contents or licence); `verified_contents` (files opened and the field inventory
confirmed); `unavailable` (checked, not reachable or access denied).

| Source | Modality | Status | Access | Role |
|---|---|---|---|---|
| PFF FC 2022 World Cup release | optical | `unverified` | request | intended primary corpus |
| Metrica Sports sample data | optical | `verified_contents`, **adapter implemented** | open download | development, pipeline validation |
| SkillCorner open data | broadcast | `verified_contents`, **adapter implemented** | open download | broadcast arm of RQ4 |
| StatsBomb Open Data | event (+ 360 frames) | `verified_contents`, **adapter implemented** | open download | fallback design |
| `pcc.data.synthetic` | simulated | generated here | — | instrument validation, power analysis |

The machine-readable record — including every verified field, with the date and
commit checked — is in `src/pcc/data/sources.py`. Obtain data with:

```bash
python scripts/fetch_data.py --source metrica_sample   --accept-terms
python scripts/fetch_data.py --source skillcorner_open --accept-terms --max-matches 4
python scripts/fetch_data.py --source statsbomb_open   --accept-terms \
    --competition 43 --season 106 --with-360
```

### What verification established

**Metrica** — 2 readable matches (a third is in EPTS/FIFA format and needs a
different deserialiser), 25 fps, ~145,000 frames each, 105 × 68 m, anonymised.
Coordinates are normalised with **(0,0) at the top left**, so the y-axis
increases downward and must be negated. Tracking and events are **already
synchronised**. There is **no frame-level possession label**; the outcome is
derived from the event stream. 2,046 arrivals extracted, 1,771 after the
inclusion criteria. Adapter implemented in `src/pcc/data/metrica.py` and tested
in `tests/test_metrica_adapter.py`.

**SkillCorner** — 20 match directories, Australian A-League 2024/25. The
repository has changed considerably since the "nine European matches, no events"
description that circulated earlier: it now ships a derived-events file, a
phases-of-play file, and 3D body pose for two matches. The tracking JSONL
**does** carry a per-frame `possession` object. Its tracking files are **Git LFS
pointers** — a plain clone yields ~130-byte stubs, and the real bytes (~90 MB
per match) must be fetched from `media.githubusercontent.com`.

**StatsBomb** — 80 competition-seasons, **12 with 360 freeze frames**, including
**FIFA World Cup 2022** (`competition_id` 43, `season_id` 106, 64 matches), the
same tournament as the PFF release. Each 360 frame has `event_uuid`, a
`visible_area` polygon, and a `freeze_frame` of *visible players only*, each
carrying `teammate` / `actor` / `keeper` and a location but **no identity**.

### A note on the availability probe

The first version of `scripts/01_check_data_availability.py` reported all three
sources unreachable. It was probing each repository's HTML landing page, which
this environment's egress proxy blocks with HTTP 403, while raw content and
`git` were reachable throughout. The probe now targets a small data-bearing
endpoint and distinguishes a proxy block from a missing dataset.

Record it here because it is the same mistake the study is about: **a negative
result from an unvalidated instrument is not evidence.**

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

### 3.2 Metrica Sports sample data — **COMPLETE**

Checked 2026-09-13 against commit `e706dd5`.

- [x] Layouts: `Sample_Game_1` and `Sample_Game_2` are CSV and are read by the
      adapter; `Sample_Game_3` is EPTS/FIFA and is **not** read. `available_games`
      returns only the readable ones, so the corpus size is never silently
      inflated by a half-read match.
- [x] Coordinates: normalised `[0,1]`, **(0,0) top-left, (1,1) bottom-right**.
      The y-axis increases downward and is negated in `to_metric`. Pitch
      105 × 68 m.
- [x] Frame rate: 25 fps, constant, ~145,006 frames per match. Asserted by
      `TrackingTable.validate` against the timestamps rather than assumed.
- [x] Frame-level possession label: **absent**. Derived from the event stream
      (a paired `BALL LOST` / `RECOVERY` transfer log) by
      `possession_from_events`, and recorded in each row's `provider` field.
- [x] Ball z-coordinate: **absent**, so aerial and ground arrivals cannot be
      separated; `pass_height` is inferred only from `HEAD`/`CROSS` subtypes.
- [x] Synchronisation: tracking and events are **already synchronised** (events
      carry frame numbers). No offset estimation needed.
- [x] Licence read: the repository asks for responsible use and acknowledgement
      of the source. Not a formal open licence; read it before publishing.
- [x] Goalkeepers: **no position labels** (the data is anonymised), so keepers
      are inferred as the player furthest in `x` from their team's centroid.

**Verified design consequence (arrival taxonomy):** Metrica yields 1,760 chosen
arrivals but only **11 unchosen** ones (clearances). Its schema folds most
loose-ball contests into a `CHALLENGE` type with no flight coordinates, so the
quasi-exogenous subsample that carries the study's identification argument is
effectively unavailable here. **Count the exogenous arrivals for any candidate
corpus and report that count**, not only the total.

**Verified design consequence (size):** two readable matches means **two
bootstrap clusters**. No confidence interval computed on this corpus is meaningful, and a
three-way match-level split is impossible, so RQ5 cannot be run on it. This is a
development corpus, confirmed, not predicted. `scripts/03_main_analysis.py`
prints a prominent warning and labels such a run a pipeline demonstration.

### 3.3 SkillCorner open data — **COMPLETE**

Checked 2026-09-15 on four LFS-resolved A-League 2024/25 matches. Adapter in
`src/pcc/data/skillcorner.py`, tested in `tests/test_skillcorner_adapter.py`.

- [x] **Git LFS.** Tracking files are LFS pointers in a plain clone (~130-byte
      stubs). `scripts/fetch_data.py` fetches the real ~90 MB per match from
      `media.githubusercontent.com`; the adapter raises an actionable error on a
      stub rather than failing obscurely later.
- [x] **Coordinates are already centred metres**, and `match.json` supplies the
      **true pitch dimensions** (105 × 68 m for the match checked) rather than a
      template — the only one of the three providers to do so.
- [x] **Attacking direction is stated, not inferred**: `home_team_side` gives it
      per period. Cross-checked against goalkeeper positions and confirmed.
- [x] **Frames are a global 10 Hz clock**, monotone across the match
      (`match_periods` gives the ranges). The *timestamps* are not safe: period 2
      restarts at 45:00 while period 1 ran to 48:18, so they overlap.
- [x] **Ball height (`ball_data.z`) is present** — Metrica has none.
- [x] Per-frame `possession.group` is supplied, but is **null in ~45% of
      frames**, so the label is sparse.
- [x] Licence read: released with PySport; the repository asks that SkillCorner
      be credited.

**Verified design consequence (extrapolation).** Every frame's `player_data`
lists all 22 players, but each carries an `is_detected` flag and only **12.7 per
frame are actually detected** (median 14). **13.8% of frames have none detected
at all**, and the ball is detected in only **60.4%**. The file is named
`tracking_extrapolated` because the provider fills the rest in.

This is the hazard the study exists to examine, so the adapter keeps the
distinction: extrapolated players remain in the state (a model in the field
would see them) while `att_observed` / `def_observed` and `frame_completeness`
record what was genuinely observed. That makes a **within-corpus** contrast
available — well-observed against poorly-observed arrivals, holding competition,
season and provider fixed — which is stronger evidence for RQ4 than the
across-corpus comparison can be. On four matches the completeness bands show no
clear signal; that is a statement about the sample size, not about the effect.

**Contamination control.** `dynamic_events.csv` carries SkillCorner's own
`xpass_completion`, `xthreat`, `possession_epv_*`, `reception_difficulty` and
`overall_pressure` columns — fitted answers to closely related questions. The
adapter reads only an enumerated list of **structural** columns
(`STRUCTURAL_COLUMNS`), and a test asserts that the modelled ones are excluded.

**Two coordinate traps, both found the hard way.** Event coordinates are
**already attack-normalised** (each event expressed with the acting team playing
toward +x) while tracking is in a fixed match frame: correlation between the two
is exactly −1.000 for the team attacking right-to-left. Applying the canonical
rotation on top mirrors one team every period. And after a *failed* pass the next
possession belongs to the opponent, whose coordinates use the opposite
normalisation, so the destination must be converted with the **next** event's
`attacking_side`.

Both bugs are close to invisible: base rates, pass lengths, flight times and
outcome orderings all survived them intact. The only symptom was mean pass
progression reading −0.59 m and then +0.71 m instead of +4.16 m. The guard in
`_check_direction_of_play` is now set at +1.5 m, calibrated from these observed
failures — all three corpora sit between +3.9 and +4.2 m.

**Arrival extraction.** A `player_possession` row ending in `end_type='pass'`
gives the release; the *next* `player_possession` row gives the arrival frame
and point. Validated: for successful passes the next possession's start lies
within 1 m of the provider's stated reception point in 95.4% of cases (median
difference 0.00 m). The provider's reception coordinates are **not** used,
because they are populated only for successful passes — taking them would
restrict the corpus to passes that worked and leave nothing to calibrate
against.

**Exogenous arrivals:** `end_type='clearance'` yields about 5 per match (19
across four matches). Thin, but non-zero — unlike StatsBomb, which has none.

### 3.4 StatsBomb Open Data — **COMPLETE**

Checked 2026-09-15 against FIFA World Cup 2022 (competition 43, season 106),
8 matches. Adapter implemented in `src/pcc/data/statsbomb.py`, tested in
`tests/test_statsbomb_adapter.py`.

- [x] **12 of 80 competition-seasons carry 360 frames** — the binding constraint
      on the fallback's sample size. They include **FIFA World Cup 2022**
      (64 matches), the same tournament as the PFF optical release, which would
      make the fallback a *same-tournament* comparison.
- [x] Coordinates: a **120 × 80 template** with (0,0) at the top left, so the
      y-axis is negated. The template is **not metres** — a real pitch may be
      100–110 m long — so mapping it onto 105 × 68 m introduces a metric error
      into every time-to-point calculation, systematic per match and
      unmeasurable without the true dimensions.
- [x] Events are **already attack-normalised** (the team in possession plays
      toward x = 120), so no attacking-direction inference is needed — unlike
      Metrica, where getting that wrong mirrors the pitch.
- [x] **Flight time is measured, not imputed.** Every pass carries a `duration`;
      implied ball speeds run 6.9–22.1 m/s (median 13.1), which is physically
      plausible. *This corrects the earlier assumption that a ball-speed model
      would be required.* Caveat: `duration` runs to the related event, which is
      the flight time for a completed pass and an approximation otherwise.
- [x] **`pass.height`** separates Ground / Low / High, so aerial and ground
      arrivals can be distinguished — which a tracking corpus lacking a ball
      z-coordinate (Metrica) cannot do.
- [x] Possession is labelled at event level (`possession`, `possession_team`),
      so the outcome need not be reconstructed from a transfer log.
- [x] Frame structure: `event_uuid`, `visible_area`, `freeze_frame`. Each
      freeze-frame entry has `teammate`, `actor`, `keeper` and `location` — and
      **no identity**, so per-player sprint-speed estimation is impossible.
- [x] Licence: StatsBomb's user agreement is in the repository. Read it before
      publishing anything derived from the data.

**Verified design consequence (visibility).** 360 frames cover 86.6% of passes.
The median frame shows **17 of 22 players**, and **no frame shows all 22**. Most
importantly, **16.4% of pass destinations fall outside the `visible_area`
polygon**; on those arrivals barely half have any visible defender within 10 m.
The adapter computes `dest_visible` per arrival and flags the rest
`destination_not_visible`. Treating them as ordinary arrivals would bias control
upward exactly where the study is looking.

**Verified design consequence (no exogenous arrivals).** Only `Pass` events carry
both a start and an end location. Clearances, interceptions and duels are
recorded as single-location events, so the corpus yields **no deflections,
clearances or second balls at all**. The quasi-exogenous contrast — the study's
strongest identification argument (Section 14) — is therefore **unavailable** on
the fallback design. Combined with the Metrica finding (11 exogenous arrivals in
two matches), this makes arrival taxonomy a first-order selection criterion for
the primary corpus rather than a detail.

**Verified label quality.** The derived control label agrees with the provider's
own pass outcome on **96.6%** of completed passes. On *incomplete* passes the two
diverge substantially — about half of failed passes still end with the same team
in control a second later — which is not an error but the construct distinction
of Section 7.2 made visible: "the pass completed" and "the team controls the
ball" are different questions.

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
