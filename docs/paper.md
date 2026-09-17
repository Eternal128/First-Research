# Sharp but Wrong: Pitch-Control Values Are Not Calibrated Probabilities

**An evaluation of velocity-free pitch-control models against 137,545 realised
ball arrivals**

---

> **Scope, stated before anything else.** This paper evaluates the
> **velocity-free, camera-limited** variants of three pitch-control formulations
> on freeze-frame data. A freeze frame is a single snapshot, so no model here has
> access to player velocity, and the camera records a median of 16 of 22 players.
> **This is not a test of published full-tracking pitch-control models.** It is a
> test of what those formulations do under the data conditions most clubs and
> researchers outside the elite tier actually work with — a different question,
> and one the results suggest is worth answering on its own terms.
>
> All results below are reproducible from `scripts/` on data obtainable with
> `scripts/fetch_data.py`. No number in this paper was chosen after seeing the
> data: the hypotheses, subgroups, horizons and metrics were fixed in
> `docs/proposal.md` beforehand, and the analysis order is frozen in code.

---

## 1. Introduction

A pitch-control model takes the instantaneous positions of the players and
returns a field over the pitch, `C_A(x, y, t) ∈ [0, 1]`, routinely described as
the probability that team A would control the ball were it to arrive at that
point at that time. The number is not merely displayed. It is multiplied by
positional values to produce expected possession value, integrated over regions
to produce space-creation metrics, differenced across counterfactual player
positions to value off-ball runs, and compared against thresholds to recommend
passes. Every one of those operations presupposes that `C_A` is a calibrated
probability.

That presupposition has, as far as we can establish, not been tested. We are not
aware of a published reliability diagram or proper-scoring-rule decomposition for
any pitch-control model. The apparent obstacle is counterfactuality: control is
unobserved at almost every point on the pitch, so the field looks unfalsifiable.

It is not. At every pass, cross and clearance the ball genuinely arrives
somewhere, and at that point the model's value is an ordinary probability
forecast whose outcome is an ordinary draw. A season of data supplies tens of
thousands of such forecast–outcome pairs. What follows is a standard
probabilistic forecast evaluation, using apparatus settled in meteorology fifty
years ago and not previously applied here.

**Contributions.**

1. The first calibration assessment of pitch-control models we are aware of, on
   137,545 arrivals across 179 matches and three competitions.
2. A demonstration that discrimination and calibration diverge sharply in this
   domain: the physics-based model ranks situations well (AUC 0.76) while
   scoring *worse than forecasting the base rate*.
3. A characterisation of where the error concentrates — the attacking third,
   long passes, poorly observed frames — and evidence that it is structural
   rather than local, transporting unchanged across competitions and across the
   men's/women's boundary.
4. Evidence that the defect is repairable: a recalibration map fitted on two
   competitions removes 97–99% of the measured miscalibration on a third.
5. Quantification of the downstream consequence: aggregate possession value is
   understated by 28%, and 36% of threshold-based pass recommendations change
   under recalibration.
6. A tested, open implementation of the whole protocol, including adapters for
   three providers and an instrument-validation suite.

---

## 2. Related work

Dominant-region models partition the pitch by which player reaches each point
first (Taki and Hasegawa, 2000; Fujimura and Sugihara, 2005). Physics-based
formulations make arrival time uncertain and let control accumulate dynamically
(Spearman et al., 2017; Spearman, 2018); related space-occupation and
possession-value work embeds such surfaces in full valuation frameworks
(Fernández and Bornn, 2018; Fernández, Bornn and Cervone, 2021). Brefeld, Lasek
and Mair (2019) learn movement models from data and derive zones of control from
them — the closest existing work to a statistically grounded control field.

None of this literature reports whether the resulting values behave like
probabilities. The apparatus for asking exists and is mature: proper scoring
rules (Brier, 1950; Gneiting and Raftery, 2007), the sharpness-subject-to-
calibration framing (Gneiting, Balabdaoui and Raftery, 2007), score
decompositions (Murphy, 1973), the calibration hierarchy (Van Calster et al.,
2016), and binning-free reliability diagrams via isotonic regression
(Dimitriadis, Gneiting and Jordan, 2021). Recalibration is equally well
established (Platt, 1999; Zadrozny and Elkan, 2002; Kull, Silva Filho and Flach,
2017).

The contribution here is a transfer, not an invention. Full references, with
unverified bibliographic details flagged, are in `docs/references.bib`.

---

## 3. Problem formulation

### 3.1 Three constructs, routinely conflated

Reachability ("who gets there first") is a question about locomotion. Possession
("who ends up with the ball") adds the contest at the point of contact.
Retention ("who still has it after the next action") adds the quality of the
subsequent decision. These are different quantities, and in aggregate
`C^{reach} ≥ C^{possess} ≥ C^{retain}`.

Dominant-region models compute the first. The probabilistic reading used
downstream asserts the second. We evaluate against the second, and return to the
resulting objection in Section 8.

### 3.2 Operational definition

> `Y = 1` if and only if a player of team A is in possession at `t_arrival + h`,
> with `h = 1.0 s`. Arrivals whose horizon window is interrupted by a stoppage
> are **censored and excluded**, not counted as failures.

Censoring rather than coding stoppages as losses avoids biasing every model
downward in the zones where fouls concentrate. It is not free: censoring is
itself selective, and the censored fraction is reported throughout.

### 3.3 The estimand

A model is calibrated on a population of arrivals if
`E[Y | C_A(X) = c] = c` for almost every `c` in the support of the forecast.
This is *weak* calibration in the Van Calster et al. hierarchy — conditioning on
the forecast value, not on the full covariate vector. Strong calibration is not
testable at realistic sample sizes and is not claimed.

### 3.4 Selection

The ball arrives only where someone sent it. Writing `D = 1` for an observed
arrival, the data identify `P(Y | X, D = 1)` while the field claims to describe
`P(Y | X)`. These coincide only if `Y ⫫ D | X`. Two obstacles separate them:
covariate shift in where the ball goes, which reweighting can address, and
selection on what the passer knew and the state vector does not contain, which
it cannot. Section 6 treats both.

---

## 4. Data

### 4.1 Corpus

179 matches of StatsBomb 360 freeze-frame data across three competitions: FIFA
World Cup 2022 (64 matches), UEFA Euro 2024 (51) and Women's World Cup 2023
(64). 151,418 ball arrivals were extracted and 137,545 retained after the
pre-registered inclusion criteria, of which 40,867 fall in the test fold across
53 held-out matches.

| Competition | Matches | Arrivals | Base rate |
|---|---|---|---|
| FIFA World Cup 2022 | 64 | 51,390 | 0.909 |
| UEFA Euro 2024 | 51 | 42,077 | 0.921 |
| Women's World Cup 2023 | 64 | 44,078 | 0.859 |

### 4.2 What the data does and does not contain

Verified against the files rather than taken from documentation:

* **Flight time is measured**, not imputed: every pass carries a duration, with
  implied ball speeds of 6.9–22.1 m/s (median 13.1).
* **Possession is labelled at event level**, so the outcome is not reconstructed
  from a transfer log.
* **There is no velocity.** A freeze frame is one snapshot.
* **The camera limits what exists.** 360 frames cover 83.9% of passes; the
  median pass event shows 16 of 22 players; all 22 are visible at roughly 1
  arrival in 2,500; and **21% of pass destinations fall outside the
  `visible_area` polygon**, where "no defender near the destination" means "no
  defender visible".
* **No exogenous arrivals.** Only pass events carry both a start and an end
  location, so the corpus contains no deflections, clearances or second balls.
  This removes the strongest available identification strategy against selection
  (Section 6).

The derived outcome agrees with the provider's own pass outcome on 96.6% of
completed passes. On incomplete passes the two diverge — roughly half of failed
passes still end with the same team in control a second later — which is the
Section 3.1 construct distinction made visible rather than an error.

### 4.3 Dependence

Arrivals cluster within matches. The measured intra-match ICC is 0.013 and the
**design effect is 11.1**, so intervals computed as though arrivals were
independent would be about 3.3 times too narrow. All intervals below are
percentile cluster bootstraps over 1,000 replicates resampling whole matches.

---

## 5. Methods

### 5.1 Models

All models receive identical information and are evaluated on identical
arrivals.

| | Fitted? | Description |
|---|---|---|
| **M0** Base rate | yes | Predicts the training base rate everywhere. Perfectly calibrated, useless; the reference for resolution. |
| **M1** Voronoi | no | `C_A = 1[min_j τ_j < min_k τ_k]`. Hard 0/1. |
| **M2** Physics | no | Logistic arrival times with competing-risks accumulation of control. Parameters (`σ`, `λ`, `v_max`, `t_r`) assumed, not estimated. |
| **M2a** Reachability sigmoid | one parameter | `C_A = σ((min_k τ_k − min_j τ_j)/s)`. Isolates how much of M2 is a monotone squash of a time-to-point difference. |
| **M3** Logistic | yes | Penalised logistic regression on engineered kinematic features computed from exactly the state M1/M2 receive. |
| **M4** GBM | yes | Histogram gradient boosting on the same features. |

### 5.2 Evaluation

Splits are by **match**, never by row: arrivals nest within possessions within
matches, and a random split rewards memorising a match's shape. Recalibration
maps are fitted on a validation fold disjoint from both the fitting and test
data; the protocol raises if a split lacks one. A leakage audit is written with
every run.

The primary calibration statistic is the **match-cross-fitted CORP
miscalibration term (MCB)** of the Brier score, not binned expected calibration
error. Binned ECE depends on an arbitrary bin count, is biased, is not a proper
scoring rule, and can be driven near zero by coarsening; our test suite
constructs a forecast whose two-bin ECE is under 0.02 and whose forty-bin ECE
exceeds 0.08.

### 5.3 Instrument validation

Before measuring anything, the calibration estimator was validated against a
known ground truth on simulated data: it reports near-zero miscalibration for an
oracle forecast, recovers an injected logit distortion to within 0.01 of the
injected slope, confirms that AUC is unchanged to machine precision by that
distortion, and produces intervals 2.3 times wider under clustering than a naive
bootstrap. All eight checks pass and are re-run by `scripts/09_validate_instrument.py`.

---

## 6. Results

Test fold: 40,867 arrivals across 53 held-out matches, base rate 0.894.

### 6.1 Overall calibration

| Model | Brier [95% CI] | MCB [95% CI] | DSC | Slope | AUC | Forecast SD |
|---|---|---|---|---|---|---|
| M1 Voronoi | 0.231 [0.216, 0.244] | 0.142 [0.133, 0.152] | 0.006 | 0.114 | 0.675 | 0.428 |
| M2 Physics | 0.167 [0.156, 0.177] | **0.081 [0.075, 0.087]** | 0.008 | **0.306** | 0.762 | 0.313 |
| M2a Reach. sigmoid | 0.155 [0.146, 0.163] | 0.069 [0.064, 0.073] | 0.008 | 0.636 | 0.761 | 0.247 |
| M0 Base rate | 0.094 [0.088, 0.101] | 0.000 | 0.000 | — | 0.500 | 0.000 |
| M3 Logistic | 0.078 [0.072, 0.084] | 0.001 [0.001, 0.001] | 0.017 | 1.014 | 0.837 | 0.133 |
| M4 GBM | 0.073 [0.068, 0.079] | 0.000 [0.000, 0.001] | 0.021 | 1.018 | 0.864 | 0.145 |

The unfitted geometric models are not calibrated. The physics model's
miscalibration interval is far from zero, and its calibration slope of 0.31 is
the signature of severe overconfidence in scale: the forecast spreads toward 0
and 1 roughly three times further than the evidence supports. Its mean forecast
is 0.72 against a base rate of 0.89, so it also under-forecasts systematically.

**The central observation of this paper is the ordering of the Brier and AUC
columns.** M2 discriminates well — AUC 0.76, far above chance — while scoring
0.167, nearly twice as bad as forecasting the base rate everywhere (0.094). A
model that looks good on a discrimination metric can be worse than useless as a
probability. M0 makes the converse point: it is perfectly calibrated (MCB 0.000)
and carries no information whatever (DSC 0.000, AUC 0.500). Neither property
substitutes for the other, and reporting only one is how a field arrives at
numbers that look like probabilities and do not behave like them.

### 6.2 Comparison against fitted baselines

Paired cluster-bootstrap differences in Brier score against M3, on identical
test arrivals (positive favours the baseline):

| Model | ΔBrier vs M3 [95% CI] | Verdict |
|---|---|---|
| M1 Voronoi | +0.153 [0.142, 0.163] | baseline |
| M2 Physics | +0.089 [0.082, 0.096] | baseline |
| M2a Reach. sigmoid | +0.077 [0.072, 0.082] | baseline |
| M0 Base rate | +0.017 [0.015, 0.019] | baseline |
| M4 GBM | −0.005 [−0.005, −0.004] | model |

A penalised logistic regression, given *exactly* the state the physics model
receives, is decisively better. Two further observations. M2 and M2a differ by
0.012 Brier while both sit 0.08–0.09 above M3, so the competing-risks dynamics
add little beyond a monotone squash of a time-to-point difference — most of what
M2 does could be done with one fitted parameter. And M2's DSC (0.008) is less
than half M3's (0.017), so the gap is not only scale: the physics model also
carries less information about which arrivals differ.

### 6.3 Where the error concentrates

All 47 pre-registered strata show FDR-adjusted evidence of miscalibration
(α = 0.05). Every directional prediction registered in advance is supported.

| Stratum | n | MCB | Slope |
|---|---|---|---|
| Attacking third | 12,161 | 0.158 | 0.273 |
| Middle third | 20,448 | 0.057 | 0.359 |
| Defensive third | 8,258 | 0.031 | 0.324 |
| Pass 0–10 m | 10,790 | 0.070 | 0.447 |
| Pass 20–30 m | 7,484 | 0.098 | 0.264 |
| Pass 30–45 m | 3,124 | 0.126 | 0.190 |
| Pass 45 m+ | 987 | 0.156 | 0.092 |
| Pressure 0 opponents | 12,679 | 0.075 | 0.306 |
| Pressure 3+ opponents | 1,705 | 0.114 | 0.336 |
| **Attacking fifth, central** | 2,175 | **0.292** | 0.185 |

Miscalibration is five times worse in the attacking third than the defensive
third, rises monotonically with pass length, and is worst of all in the central
attacking fifth — the penalty area, where the mean forecast is 0.34 against an
observed rate of 0.84. This is precisely the region where space-creation and
possession-value metrics are most used and most consequential.

### 6.4 How much of it is the camera?

Calibration improves monotonically with the fraction of the frame observed:

| Frame completeness | n | MCB | Slope |
|---|---|---|---|
| < 60% | 11,040 | 0.095 | 0.221 |
| 60–75% | 12,277 | 0.086 | 0.300 |
| 75–90% | 13,104 | 0.076 | 0.385 |
| 90%+ | 4,446 | 0.049 | 0.556 |

Destinations outside the visible area are far worse (MCB 0.145, slope 0.167,
n = 8,565) than those inside it (0.064, 0.390, n = 32,302). Restricting to
near-complete frames *with* a visible destination gives MCB 0.045 and slope
0.571, against 0.081 and 0.306 overall — so roughly **45% of the measured
miscalibration is attributable to incomplete observation and 55% survives it**.
Even when the camera sees nearly everything, the velocity-free model remains
badly overconfident while its AUC rises to 0.86: the same divergence, at better
data quality.

This decomposition is **suggestive, not causal**. High-completeness frames are
not a random subsample — the ball is more often in crowded central areas — so
the comparison confounds observability with the kind of situation observed.

### 6.5 Selection

The reweighting arm of the analysis **failed its own diagnostics, and we report
it as failed rather than reporting the number it produced.**

A classifier separating chosen from proposed destinations achieves AUC 0.972:
passers select their targets very strongly, confirming the premise. But the
overlap diagnostics are damning. **52.6% of observed arrivals lie outside the
support of the candidate distribution**; the top 1% of weights carry 33% of the
mass; the largest weight is 521; and the effective sample size collapses from
48,268 to 7,753 — 16% — under the trimmed, stabilised weights the analysis
actually uses (1,525, or 3%, before trimming). Under those conditions the
reweighted estimate (MCB 0.127 against an unweighted 0.085) is extrapolation,
not correction, and we do not interpret it.

Stratification is more informative. By quintile of the selection score
(stratum 0 = least typical destinations):

| Stratum | n | Base rate | Mean forecast | MCB |
|---|---|---|---|---|
| 0 (least typical) | 9,654 | 0.790 | 0.507 | 0.148 |
| 1 | 9,653 | 0.896 | 0.612 | 0.157 |
| 2 | 9,654 | 0.932 | 0.782 | 0.065 |
| 3 | 9,653 | 0.959 | 0.865 | 0.039 |
| 4 (most typical) | 9,654 | 0.969 | 0.906 | 0.022 |

The model is worst on the destinations players choose least often, by a factor
of roughly seven. The headline figure of 0.081 is therefore a weighted average
dominated by routine passes, and it *understates* the error in the unusual
situations where a control model would most plausibly be consulted.

**The strongest available identification strategy could not be run.** The
proposal's primary answer to selection was a quasi-exogenous subsample —
deflections, clearances and second balls, which arrive where nobody aimed. This
corpus contains none: only pass events carry both a start and an end location.
That is a real limitation, and its absence is why the sensitivity analysis
carries more weight here than intended.

That sensitivity analysis is reassuring. Modelling unmeasured selection as a
multiplicative bias Γ on the outcome odds among observed arrivals, MCB remains
0.039 even at Γ = 3 — an implausibly strong assumption about what passers know
and the state vector does not. **No tipping point exists within the range
examined**: the finding is not explicable by unmeasured selection of any
plausible magnitude.

### 6.6 Does it transport, and can it be fixed?

Holding out each competition in turn:

| Model | MCB within competition | MCB across competitions | Penalty for crossing |
|---|---|---|---|
| M2 Physics | 0.085 | 0.083 | **−0.002** |
| M3 Logistic | 0.0010 | 0.0011 | +0.0001 |
| M4 GBM | 0.0006 | 0.0004 | −0.0002 |

Crossing a competition boundary costs essentially nothing, **including the
men's-to-women's boundary**, despite base rates differing materially (0.909,
0.921, 0.859). The miscalibration is structural, not local: the model is equally
wrong everywhere. That is a stronger and more useful claim than being wrong on
one league.

It follows that one fix travels. A recalibration map fitted on two competitions
and applied to a third, unseen one removes **97–99%** of the physics model's
measured miscalibration (isotonic 98.9% on average, worst case 98.6%; beta
98.6%/98.3%; Platt 97.4%/96.4%). On the main match-level split, all three maps
drive MCB from 0.081 to within 0.001 of zero.

The model's *ordering* of situations is therefore sound; only its scale is
wrong. The defect is repairable downstream by a two- or three-parameter map that
a practitioner can fit once and deploy — a deflating result, and a useful one.

### 6.7 Does the conclusion survive the definition of control?

The study imposes an operational definition on an undefined construct. The
pre-registered rule was that a conclusion which flips across the sweep would be
reported as flipping. It does not flip.

| Definition | Horizon | Base rate | Censored | MCB | Slope | AUC |
|---|---|---|---|---|---|---|
| Control at horizon | 0.25 s | 0.917 | 0.6% | 0.088 | 0.306 | 0.772 |
| Control at horizon | 1.0 s | 0.911 | 1.2% | 0.084 | 0.300 | 0.768 |
| Control at horizon | 3.0 s | 0.883 | 2.9% | 0.070 | 0.278 | 0.748 |
| First touch | — | 0.918 | 0.0% | 0.088 | 0.304 | 0.772 |
| Retained sequence | — | 0.912 | 1.2% | 0.084 | 0.303 | 0.770 |
| Possession at 5 s | — | 0.855 | 1.9% | 0.064 | 0.219 | 0.705 |

Across six horizons and four outcome definitions, MCB stays in [0.064, 0.088]
and the slope in [0.219, 0.305]. The conclusion is not an artefact of the
horizon. Notably, the **first-touch** definition — the closest observable proxy
for the reachability construct that dominant-region models actually compute —
gives MCB 0.088, no better than the possession definition. The models are not
merely being scored against the wrong target (Section 8).

Censoring rises from 0.6% to 2.9% as the horizon lengthens. Since fouls cluster
in contested areas, a longer horizon buys construct breadth at the cost of a
more selected sample; the effect is small at these rates but is reported rather
than assumed away.

---

## 7. Practical implications

### 7.1 Possession value is understated by more than a quarter

Propagating raw and recalibrated control values through a linear expected-
possession-value construction over the 40,867 test arrivals:

| Control variant | Mean EPV | Total EPV |
|---|---|---|
| Raw | 0.559 | 22,844 |
| Platt | 0.774 | 31,613 |
| Isotonic | 0.773 | 31,581 |
| Beta | 0.773 | 31,576 |

Aggregate possession value computed from raw control values is **28% lower**
than from calibrated ones. This is the accumulation argument made concrete:
because EPV is linear in `C`, a calibration error with a consistent sign does
not average out across a season — it compounds.

### 7.2 A third of pass recommendations change

| Threshold | Recommendations changed | Accuracy raw | Accuracy recalibrated |
|---|---|---|---|
| 0.50 | 25.2% | 0.762 | 0.895 |
| 0.60 | 30.6% | 0.727 | 0.895 |
| 0.70 | **36.0%** | 0.680 | 0.892 |
| 0.80 | 24.0% | 0.622 | 0.792 |

### 7.3 But read the mechanism carefully

Recalibration improves net benefit at 90 of 91 thresholds, and doubles the range
over which the model beats the best default policy (from 15 thresholds to 31).
It is tempting to conclude that recalibration makes the model better at
football. It does not.

Every map used here is monotone, so AUC is unchanged to machine precision: the
model's *ranking* of arrivals is identical before and after. What recalibration
restores is the **meaning of the threshold**. A practitioner who sets a cut-off
from a stated risk appetite — "attempt the pass when control probability exceeds
0.7" — is, with the raw forecast, actually operating at a far stricter point
than intended, because a raw 0.7 corresponds to an observed rate near 0.9. A
practitioner who instead tuned their threshold empirically on the raw forecasts
would already be getting the same decisions.

So the honest statement is narrower than "recalibration improves decisions":
**miscalibration corrupts any use of the value that is not purely ordinal** —
summing, multiplying, integrating, or comparing against an externally chosen
threshold. Purely ordinal uses, such as ranking a player's passing options
against each other, are unaffected.

---

## 8. Limitations

**This does not test full-tracking pitch-control models.** The single most
important limitation. No model here has velocity, and velocity is the primary
input the physics formulations were designed around. The results bound what
those models do under degraded data; they do not measure what they do under the
data they were built for. Section 6.4 suggests observability explains under half
of the error, but that decomposition is not causal, and the velocity component
cannot be separated at all on this corpus.

**Construct mismatch is a live objection.** Dominant-region models answer "who
arrives first"; we score against "who ends up with the ball". A defender could
argue we are scoring a model against a target it never claimed. Two responses.
First, the downstream literature does use these fields for the second question;
if they answer only the first, that invalidates the downstream uses rather than
excusing the models. Second, and more directly: under the **first-touch**
definition, the closest observable proxy for reachability, MCB is 0.088 —
no better. The mismatch does not explain the result.

**The selection analysis is incomplete.** The quasi-exogenous subsample could
not be constructed, the reweighting failed its overlap diagnostics, and we are
left with stratification and a sensitivity bound. The bound is strong (no
tipping point below Γ = 3), but it is a bound, not identification.

**Observational data cannot identify control where the ball never arrives.** Our
conclusions extend to realised arrivals. That is also where every downstream
application operates, so the limitation is narrower than it sounds — but the
full field's claims remain untested and, on observational data, untestable.

**Three competitions, two of them elite men's tournaments.** Transportability
held across the men's/women's boundary, which is evidence against
population-specificity, but youth and lower-division football are untested and
the locomotion assumptions have less reason to hold there.

**Parameter sensitivity was not exhausted.** The arrival-time uncertainty σ
governs the field's sharpness directly, and our ablations on smaller corpora
showed it moving the calibration slope by more than the difference between
models. A σ fitted to outcomes would likely improve M2 substantially — which is
itself the point: a parameter that must be fitted to outcomes is not physics.

---

## 9. Conclusion

On 137,545 realised ball arrivals, velocity-free pitch-control values are not
calibrated probabilities. The physics-based model scores worse than forecasting
the base rate while discriminating well above chance; its calibration slope is
0.31; a penalised logistic regression given identical inputs is decisively
better. The error concentrates in the attacking third, grows with pass length,
worsens where the camera sees less, and is seven times larger on the
destinations players choose least often. It transports unchanged across three
competitions and across the men's/women's boundary, and it survives every
horizon and outcome definition we pre-registered.

It is also almost entirely removable. A recalibration map fitted on two
competitions strips 97–99% of it from a third. The ordering these models produce
is sound; the numbers attached to that ordering are not.

The practical recommendation is therefore specific rather than dismissive.
Pitch-control fields remain useful for ranking. They should not be summed,
integrated, multiplied by positional values, or compared against externally
chosen thresholds without recalibration — and on this evidence a single global
map suffices to make them fit for those uses.

Whether the same holds for full-tracking implementations is the obvious next
question, and it needs a corpus this study did not have.

---

## Reproduction

```bash
pip install -r requirements.txt && pip install -e .
python scripts/09_validate_instrument.py          # validate the estimator first
for spec in "43 106" "55 282" "72 107"; do set -- $spec
  python scripts/fetch_data.py --source statsbomb_open --accept-terms       --competition $1 --season $2 --with-360
done
python scripts/03_main_analysis.py        --source statsbomb_open --n-boot 1000
python scripts/04_subgroup_analysis.py    --source statsbomb_open
python scripts/05_selection_analysis.py   --source statsbomb_open --max-matches 60
python scripts/06_downstream_analysis.py  --source statsbomb_open
python scripts/11_transportability.py     --source statsbomb_open
python scripts/12_construct_sensitivity.py --source statsbomb_open --max-matches 64
```

Every results directory carries a manifest recording the git revision, package
versions, and a hash of the configuration. Protocol, hypotheses and subgroups
are in `docs/proposal.md`; dataset verification in `docs/data_sources.md`.

