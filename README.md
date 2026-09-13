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
| Real data | **None.** No provider data is included or verified. See `docs/data_sources.md` |
| Results | Simulated only. They describe the simulator, **not football** |

---

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

python scripts/00_environment_report.py      # what can run here
python scripts/09_validate_instrument.py     # is the calibration estimator correct?
bash    scripts/run_all.sh                   # the whole pipeline, on simulated data
pytest -q                                    # 120 tests
```

No network access and no provider data are needed. Everything runs on the
built-in simulator.

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
| `src/pcc/data/` | Schema contract, labelling, preprocessing, source registry, loaders, simulator |
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

**No provider data is in this repository, and none of the candidate sources'
properties have been verified here.** `docs/data_sources.md` records what each
source is documented to provide, what must be checked before relying on it, and
the licensing position. Run:

```bash
python scripts/01_check_data_availability.py --probe-network
```

It reports presence and reachability only, distinguishes a network block from a
missing dataset, and writes `results/data_availability.json`.

Provider adapters in `src/pcc/data/loaders.py` are deliberately unfinished
scaffolds with `TODO(access)` checklists. They raise actionable errors rather
than pretending to work.

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
