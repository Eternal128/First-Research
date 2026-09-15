# Are football pitch-control models calibrated probabilities?

Research code and protocol for an empirical study of whether pitch-control
values behave like the probabilities they are read as.

**The question.** Pitch-control models emit a field `C_A(x, y, t) ∈ [0, 1]`,
routinely interpreted as the probability that team A would control the ball if it
arrived at `(x, y)` at time `t`. That reading is load-bearing: possession-value
frameworks multiply by it, space metrics integrate it, pass recommendations
threshold it. It is also, as far as we can establish, untested.

**Why it is testable.** Control is counterfactual almost everywhere on the pitch
— but at every pass, deflection and clearance the ball genuinely arrives
somewhere, and the realised outcome there is a draw from the distribution the
model claims to describe. That makes it an ordinary probabilistic forecast
evaluation problem.

📄 **[Read the full proposal → `docs/proposal.md`](docs/proposal.md)**

---

## Status

| | |
|---|---|
| Protocol | Complete — 25 sections, pre-registered hypotheses and analysis plan |
| Implementation | Complete and tested — models, metrics, splits, selection layer, downstream propagation, ablations |
| Tests | 120 passing, including 8 instrument-validation checks |
| Real data | Metrica, SkillCorner and StatsBomb obtained and **verified against the files**; PFF unobtained. See `docs/data_sources.md` |
| Loaders | **All three implemented and tested** (Metrica optical, SkillCorner broadcast, StatsBomb freeze-frame); PFF unobtained |
| Results | Not committed (regenerable). The freeze-frame corpus is **179 matches / 137k arrivals** and is powered; the tracking corpora are 2 and 4 matches and are not |

---

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

python scripts/00_environment_report.py      # what can run here
python scripts/09_validate_instrument.py     # is the calibration estimator correct?
bash    scripts/run_all.sh                   # the whole pipeline, on simulated data
pytest -q                                    # the test suite
```

No network access and no provider data are needed for any of that — it runs on
the built-in simulator.

To run on real football:

```bash
# optical tracking, 2 matches - the development corpus
python scripts/fetch_data.py --source metrica_sample --accept-terms
python scripts/03_main_analysis.py --source metrica_sample

# broadcast tracking, A-League - the RQ4 arm (Git LFS; the fetcher resolves it)
python scripts/fetch_data.py --source skillcorner_open --accept-terms --max-matches 4
python scripts/03_main_analysis.py --source skillcorner_open

# event data + 360 freeze frames, World Cup 2022 - the fallback design
python scripts/fetch_data.py --source statsbomb_open --accept-terms \
    --competition 43 --season 106 --max-matches 8 --with-360
python scripts/03_main_analysis.py --source statsbomb_open

# one protocol across all three
python scripts/10_cross_corpus.py
```

First parse of a large corpus takes minutes; results are cached under
`results/cache/` and keyed by a fingerprint of the raw data plus the label and
preprocessing settings, so adding matches invalidates the cache rather than
silently serving a smaller one.

The two tracking corpora are small, and the analysis script will say so at
length: the cluster bootstrap resamples *matches*, so two matches means two
clusters. The freeze-frame corpus can be scaled to 179 matches across three
competitions, which is enough for inference — but it is the **fallback** design
(no velocity, camera-limited), so what it measures is the velocity-free variant
of each model, not the published full-tracking ones. See proposal Section 20.1,
which states that scope limit before any number.

```bash
# scale the fallback corpus to three competitions (~1.8 GB)
for spec in "43 106" "55 282" "72 107"; do set -- $spec
  python scripts/fetch_data.py --source statsbomb_open --accept-terms \
      --competition $1 --season $2 --with-360
done
python scripts/03_main_analysis.py  --source statsbomb_open --n-boot 1000
python scripts/11_transportability.py --source statsbomb_open
```

---

## What is here

| Path | Contents |
|---|---|
| `docs/proposal.md` | The research proposal and paper blueprint |
| `docs/data_sources.md` | Per-source verification checklists; licensing; **nothing asserted unverified** |
| `docs/references.bib` | Bibliography, with unverified entries flagged `CITATION-REQUIRED` |
| `src/pcc/models/` | M0 marginal, M1 Voronoi, M2 physics, M2a reachability sigmoid, M3 logistic, M4 GBM, M5 Deep Sets (optional) |
| `src/pcc/calibration/` | Proper scores, CORP decomposition, reliability curves, recalibration maps, cluster bootstrap |
| `src/pcc/evaluation/` | Leakage-safe splits, the evaluation protocol, subgroups, decision curves |
| `src/pcc/selection/` | Candidate arrivals, density-ratio weights, overlap diagnostics, sensitivity bounds |
| `src/pcc/downstream/` | Positional value surface, EPV, space metrics, decision displacement |
| `src/pcc/data/` | Schema contract, labelling, preprocessing, source registry, simulator |
| `src/pcc/data/metrica.py`, `skillcorner.py`, `statsbomb.py` | Implemented provider adapters |
| `src/pcc/data/tracking.py` | Provider-independent state container and arrival assembly |
| `scripts/` | The numbered pipeline; `run_all.sh` runs all of it |
| `configs/default.yaml` | Every analyst choice, hashed into each results manifest |

---

## Three design commitments

**1. The primary metric is not binned ECE.** It is the match-cross-fitted
miscalibration term of the CORP decomposition of the Brier score. Binned ECE
depends on an arbitrary bin count, is biased, is not a proper scoring rule, and
can be driven near zero by coarsening. `tests/test_metrics.py` constructs a
forecast whose two-bin ECE is under 0.02 and whose forty-bin ECE is over 0.08.

**2. Nothing is fitted on the test fold.** Splits are by match and competition,
never by row — arrivals nest within possessions within matches. Recalibration
maps are fitted on a disjoint validation fold; the protocol raises if a split has
no validation fold. A leakage audit is written with every experiment.

**3. Selection bias is a design problem, not a caveat.** Passers choose their
destinations, partly on information the data does not contain. Reweighting fixes
the covariate-shift part and *cannot* fix the rest. The primary response is a
quasi-exogenous subsample — deflections, clearances and second balls, which
arrive where nobody aimed — plus a sensitivity bound on how strong unmeasured
selection would have to be to explain a finding away.

---

## Data

**No provider data is committed to this repository** (`data/raw/` is
git-ignored), but three sources have now been obtained and opened, and
`docs/data_sources.md` records what was verified against the files rather than
what the documentation claims. Highlights: Metrica has no frame-level possession
label, so the outcome is derived; SkillCorner's tracking files are Git LFS
pointers that a plain clone will not resolve; StatsBomb's 360 frames cover only
a `visible_area` polygon, so an absent defender may be an unseen one. Run:

```bash
python scripts/01_check_data_availability.py --probe-network
```

It reports presence and reachability only, distinguishes a network block from a
missing dataset, and writes `results/data_availability.json`. It probes a
data-bearing endpoint rather than a repository landing page — an earlier version
probed the HTML page and reported every source unreachable because a proxy
blocked it, which is the same mistake the study is about.

All three open-data loaders are implemented and tested, across three different
data shapes: continuous optical tracking, broadcast tracking with explicit
detection flags, and per-event freeze frames. `scripts/10_cross_corpus.py` runs
one protocol over all of them.

Verification changed the plan in three places, which is the main argument for
doing it before writing a data section: StatsBomb arrival times turned out to be
**measured** rather than needing imputation; **neither** open corpus supplies
enough exogenous arrivals (deflections, clearances, second balls) to support the
study's strongest identification argument, making arrival taxonomy a first-order
criterion for corpus choice; and the two providers place a failed pass's
"arrival" at **different points**, so pooling them without adjustment would mix
two estimands.

---

## On the simulated results

`pcc.data.synthetic` generates fabricated matches from a known data-generating
process. It exists for three reasons: to exercise the pipeline without provider
data; to validate the calibration estimator against a known truth; and to size
the corpus the real study needs.

Its football is crude — Gaussian formations, no offside, no possession structure.
Every artefact it produces is tagged `tracking_source = "simulated"`. **No number
from it may appear in a results section as a claim about football.**

---

## Licence

No licence has been chosen yet; the author should add one before publishing.
Note that this repository's licence does not and cannot extend to any provider
data placed under `data/raw/`.
