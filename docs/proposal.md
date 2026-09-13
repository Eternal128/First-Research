# Are Football Pitch-Control Models Calibrated Probabilities, or Visual Heuristics?

**A research proposal and paper blueprint**

---

> **Status of this document.** This is a *protocol*, not a report of findings. It
> states in advance what will be measured, on what data, by what estimator, and
> what each possible outcome would license us to conclude. No empirical result
> about football appears anywhere in it, because none has been obtained. Where a
> dataset property is asserted by its provider but has not been checked against
> the files, it is marked as unverified. Where a citation's bibliographic details
> are uncertain, it is marked `[citation required]` rather than reconstructed
> from memory.
>
> The accompanying code (`src/pcc/`, `scripts/`, `tests/`) implements the full
> protocol and runs end-to-end on simulated data. The simulated outputs in
> `results/` exist to demonstrate that the machinery works and to validate the
> calibration estimator against a known ground truth. **They describe the
> simulator, not football**, and must not be cited as results.

---

## Table of contents

1. [Proposed title](#1-proposed-title)
2. [Abstract](#2-abstract)
3. [Introduction](#3-introduction)
4. [Research questions](#4-research-questions)
5. [Hypotheses](#5-hypotheses)
6. [Literature review](#6-literature-review)
7. [Conceptual framework](#7-conceptual-framework)
8. [Formal mathematical formulation](#8-formal-mathematical-formulation)
9. [Model comparison](#9-model-comparison)
10. [Data requirements](#10-data-requirements)
11. [Data preprocessing](#11-data-preprocessing)
12. [Evaluation design](#12-evaluation-design)
13. [Calibration methodology](#13-calibration-methodology)
14. [Selection bias and propensity weighting](#14-selection-bias-and-propensity-weighting)
15. [Main empirical analysis](#15-main-empirical-analysis)
16. [Downstream analysis](#16-downstream-analysis)
17. [Robustness and ablation studies](#17-robustness-and-ablation-studies)
18. [Threats to validity](#18-threats-to-validity)
19. [Expected contributions](#19-expected-contributions)
20. [Possible results and their interpretation](#20-possible-results-and-their-interpretation)
21. [Reproducibility plan](#21-reproducibility-plan)
22. [Proposed paper structure](#22-proposed-paper-structure)
23. [Dissertation-level extensions](#23-dissertation-level-extensions)
24. [Publication strategy](#24-publication-strategy)
25. [Final research proposal](#25-final-research-proposal)

---

## 1. Proposed title

**Option A.**
*Are Pitch-Control Values Calibrated Probabilities? A Proper-Scoring-Rule
Evaluation of Spatial Control Models in Association Football*

**Option B.**
*Measurement Validity in Football Spatial Analytics: Probabilistic Calibration
of Pitch-Control Models Against Realised Ball Arrivals*

**Option C.**
*From Heatmaps to Probabilities: Calibration, Selection, and Downstream
Consequences of Pitch-Control Models in Elite Football*

**Selected: Option A.**

Option A is the strongest for three reasons. First, it poses a question that is
*falsifiable and answerable*, which signals to a reviewer that the paper has a
result rather than a survey. Second, it names the methodological apparatus
("proper scoring rule") in the title, which correctly positions the work for
both a statistics audience and a sports-analytics audience and pre-empts the
most common reviewer objection — that calibration is being assessed by an
ad-hoc binned statistic. Third, it is honest about scope: it claims to evaluate
*pitch-control values*, not to propose a better model, which is exactly what the
study does.

Option B is accurate but front-loads "measurement validity", a term that reads
as methodological throat-clearing to an applied audience and will lose the
football reader in the first six words. Option C is the most readable and would
work well for a workshop or a practitioner venue, but "From Heatmaps to
Probabilities" promises a constructive contribution — a method for turning one
into the other — that the study does not primarily deliver.

---

## 2. Abstract

Pitch-control models are among the most widely used tools in football spatial
analytics. Given the positions and velocities of all players, they return a
field `C_A(x, y, t) ∈ [0, 1]` that is routinely interpreted, in published work
and in club practice, as the probability that team A would control the ball were
it to arrive at location `(x, y)` at time `t`. That probabilistic reading is
load-bearing: expected-possession-value frameworks, off-ball-run valuation,
space-creation metrics and pass-difficulty models all multiply, integrate or
threshold these values as though they were probabilities. Yet the reading is
rarely tested. Control is counterfactual at almost every point on the pitch,
which has made the field appear unfalsifiable. It is not: at every pass,
deflection and clearance, the ball genuinely arrives somewhere, and the realised
outcome at that point is a draw from the distribution the model claims to
describe. This study evaluates whether pitch-control values are probabilistically
calibrated by scoring them against realised ball arrivals using strictly proper
scoring rules. We will compare a Voronoi dominant-region baseline, a
physics-based control model, a penalised logistic regression and a
gradient-boosted model on identical inputs, under match-level and
competition-level splits, and report the CORP (isotonic) decomposition of the
Brier score into miscalibration, discrimination and uncertainty, alongside
calibration slope and intercept, with cluster-bootstrap intervals. Because
passers choose their destinations, the evaluation sample is a non-random slice
of the pitch; we address this with density-ratio reweighting, stratification,
a quasi-exogenous subsample of unchosen arrivals (deflections, clearances,
second balls), and sensitivity analysis. We then measure how far recalibration
moves downstream possession-value estimates and pass recommendations. Datasets
are specified with their availability and licensing treated as unverified until
checked, and a freeze-frame fallback design is pre-registered.

*(283 words.)*

---

## 3. Introduction

### 3.1 What pitch control is used for

A pitch-control model consumes the instantaneous state of a football match — the
positions, and usually the velocities, of twenty-two players and the ball — and
returns a scalar field over the playing surface. The field is normally rendered
as a two-colour heatmap and described as "the space each team controls". Over
roughly two decades the underlying formulations have progressed from purely
geometric dominant regions, through motion-model-based reachability, to
dynamical formulations in which each player's probability of taking control
accumulates over time after the ball's arrival.

The reason this matters beyond visualisation is that pitch control has become an
*input* rather than an output. It appears, directly or in close variants, in at
least five downstream constructions:

1. **Possession valuation.** Expected-possession-value frameworks decompose the
   value of a possession into the value of each available action, and the
   probability that a pass into a given area is retained is typically supplied
   by a control field.
2. **Pass evaluation.** Pass difficulty, pass risk and "pass options" surfaces
   are built by evaluating control at candidate destinations.
3. **Off-ball movement analysis.** The value of a run is quantified as the change
   in the control field caused by the runner's displacement — a difference of
   two fields.
4. **Space creation and occupation metrics.** Team-level summaries integrate the
   field over regions of the pitch, producing "space owned" or "dangerous space"
   numbers that are reported to coaches.
5. **Tactical description.** Compactness, pressing intensity and defensive
   organisation are summarised through aggregate properties of the field.

In every one of these, the number is treated as a probability. Multiplying a
control value by a positional value to get an expected value, integrating it to
get an area, or comparing it to a threshold to decide whether a pass is "on" —
each of these operations is only meaningful if `C_A` is calibrated.

### 3.2 The measurement-validity problem

A quantity in `[0, 1]` is not thereby a probability. For `C_A` to be a
probability forecast in any useful sense, it must satisfy *calibration*: among
all situations in which the model says 0.7, team A should subsequently control
the ball about 70% of the time. This is a property of the forecast-outcome
joint distribution and it is empirically checkable. It is also entirely
independent of the model's *discrimination* — its ability to rank contested
situations above uncontested ones — which is what a model is usually judged on
when it is judged at all.

The gap between the two is not a technicality. A monotone transformation of a
forecast leaves the area under the ROC curve exactly unchanged while moving the
calibration arbitrarily. A model can therefore be the best available ranker of
football situations and simultaneously be systematically overconfident by a
wide margin; and a model that predicts the base rate everywhere is perfectly
calibrated and completely useless. Neither property substitutes for the other,
and reporting only one is how a field arrives at numbers that look like
probabilities and do not behave like them.

For pitch control specifically, three features make miscalibration likely a
priori, and none of them is a criticism of the original authors:

- **The models are unfitted.** The leading physics-based formulations estimate
  no parameter from football outcome data. Their outputs are derived from a
  locomotion envelope (a sprint speed, a reaction time, an arrival-time
  uncertainty, a rate of converting presence into control). There is no
  mechanism by which such a construction would come out calibrated, other than
  the physics happening to be right in every respect that matters.
- **The free parameters are nuisance parameters in disguise.** The arrival-time
  uncertainty σ in particular controls how sharply the field transitions from 0
  to 1. It is conventionally fixed at a plausible value. Our own ablation on
  simulated data shows that varying σ across a plausible range moves the
  calibration slope by more than the difference between competing models — which
  means σ is not a detail, it is the thing being measured.
- **The construct is ambiguous.** "Control" is not defined operationally in most
  of the literature. Whether it means first touch, possession one second later,
  or a retained possession sequence changes the target probability materially,
  and no model can be calibrated to an undefined quantity.

### 3.3 Why the field appeared unfalsifiable, and why it is not

The obstacle has been counterfactuality. `C_A` is defined at every point on the
pitch, but the ball is at one place at a time, so the field's claims at the
other 7,000 square metres are never tested. This has been taken to mean that
the field as a whole cannot be validated.

The observation that unlocks the study is narrower and sufficient: **at every
ball arrival, the model's value at that specific point and time is a genuine
probability forecast, and the realised outcome is a genuine draw.** Passes,
crosses, deflections, clearances and second balls collectively supply tens of
thousands of such forecast-outcome pairs per season. That is an ordinary
probabilistic forecast evaluation problem, for which the statistical apparatus —
proper scoring rules, reliability diagrams, score decompositions, recalibration
— is mature and was largely settled in meteorology decades ago.

Two things are *not* recovered by this move, and the study states both plainly
rather than obscuring them. First, calibration at realised arrival points does
not establish calibration over the whole field; the untargeted regions remain
untested by observational data. Second, arrivals are not randomly located.
Passers choose destinations, and they choose them partly on information the
recorded state does not contain. Section 14 treats this as the study's central
inferential problem rather than as a caveat.

### 3.4 Why miscalibration would matter

If `C_A` is systematically overconfident — the most likely failure mode, and the
one a hard-thresholded field guarantees — the consequences propagate in a
specific and predictable way, because most downstream metrics are approximately
*linear* in `C`. Expected possession value is a control-weighted average of two
positional values; space metrics are integrals of the field; off-ball run value
is a difference of two fields.

Linearity has a sharp implication: **a calibration error with a consistent sign
over a region does not average out when integrated over that region — it
accumulates.** A model overconfident by 0.05 in the final third biases every
final-third space metric in the same direction, in every match, all season. By
contrast, a purely random error would wash out. It is exactly the *systematic*
component, which the calibration slope and intercept measure, that survives
aggregation. This is why calibration, rather than mean squared error, is the
right diagnostic for a quantity that is destined to be summed.

The practical consequence is a decision-threshold displacement. If a model's 0.7
is really a 0.55, then every recommendation the model makes at a 0.7 threshold
is systematically too aggressive. That is a football error, not merely a
scoring-rule error, and Section 16 measures it directly as the fraction of pass
recommendations that change under recalibration.

### 3.5 Scope

The study evaluates published *classes* of pitch-control model. It does not
propose a new one. That restraint is deliberate: the contribution is an
evaluation protocol and an empirical answer, and a paper that both criticises
existing evaluation and introduces a competing model invites the reader to
assess the model instead of the critique. If a better-calibrated construction
falls out of the analysis, it belongs in a short final section as an extension,
not in the headline.

---

## 4. Research questions

**RQ1 — Calibration.** To what extent are the outputs of established
pitch-control models calibrated probabilities when evaluated against realised
ball arrivals, under a stated operational definition of control?

*Estimand:* the miscalibration component (MCB) of the CORP decomposition of the
Brier score, and the calibration slope and intercept, for each model, on
held-out matches, with cluster-bootstrap intervals.

**RQ2 — Comparison against simple baselines.** Do physics-based pitch-control
models outperform a penalised logistic regression and a gradient-boosted model
that are given *exactly the same information* (player positions, velocities,
destination, flight time), in calibration and in discrimination separately?

*Estimand:* paired differences in Brier score, MCB and DSC against the logistic
baseline on the same test arrivals, with paired cluster-bootstrap intervals.

**RQ3 — Heterogeneity.** Does calibration vary systematically by pitch zone,
pass length, flight time, defensive pressure, game state, arrival type and team?

*Estimand:* stratum-level MCB and calibration slope over a pre-registered
subgroup family, with FDR control across the family.

**RQ4 — Tracking modality.** Does calibration differ between optical tracking
and broadcast-derived tracking, and if so, how much of the difference is
attributable to measurement degradation as opposed to the different matches the
two modalities cover?

*Estimand:* MCB and slope under a leave-one-source-out split, compared against a
degradation-simulation ablation in which optical data is artificially occluded
and noised to broadcast-like quality.

**RQ5 — Recalibration.** Can post-hoc recalibration (Platt, isotonic, beta,
partially pooled) restore calibration, and how much of the lost score does each
recover out of sample and out of competition?

*Estimand:* change in Brier score and MCB on test matches, for maps fitted on a
disjoint validation fold; and the residual MCB after each map, which measures
how much of the distortion is not logit-linear.

**RQ6 — Downstream consequence.** How much do possession-value estimates,
space-creation metrics and threshold-based pass recommendations change when a
model's control values are recalibrated?

*Estimand:* mean and total shift in expected possession value; fraction of
decisions that flip at practitioner-relevant thresholds; net benefit from
decision-curve analysis, raw versus recalibrated.

**Scope note.** RQ1, RQ2, RQ5 and RQ6 constitute the minimum viable study
(Section 24.4). RQ3 requires a corpus large enough for stratified inference; RQ4
requires two tracking modalities. Both are staged as extensions rather than
prerequisites.

---

## 5. Hypotheses

Hypotheses are stated as null/alternative pairs with the statistic that tests
them. All intervals are cluster-bootstrap percentile intervals resampling whole
matches; "evidence against H0" means the interval excludes the null value.

### H1 — Overall calibration

- **H1a₀:** `MCB = 0` for the physics-based model (perfect calibration).
  **H1a₁:** `MCB > 0`.
  *Test:* cross-fitted CORP MCB with a cluster-bootstrap interval. Note that MCB
  is non-negative by construction in-sample and biased upward; the cross-fitted
  estimator is used precisely so the null is testable.
- **H1b₀:** calibration slope `b = 1`. **H1b₁:** `b ≠ 1`, and the directional
  prediction is `b < 1` (overconfidence).
  *Test:* logistic recalibration regression `logit P(Y=1) = a + b · logit(C)`,
  interval on `b`.
- **H1c₀:** calibration-in-the-large `a₀ = 0`. **H1c₁:** `a₀ ≠ 0`.

*Prior expectation and its basis.* We expect `b < 1` for the geometric models.
The reasoning is structural rather than empirical: a Voronoi field emits only
0 and 1, which is the extreme case of overconfidence, and a physics-based field
softens that boundary by an amount governed by an assumed σ rather than by data.
We record this expectation in advance so that a finding of `b ≈ 1` counts as
genuinely surprising evidence rather than being absorbed after the fact.

### H2 — Physics versus fitted baselines

- **H2a₀:** the paired difference in Brier score between the physics model and
  the logistic baseline is zero. **H2a₁:** it is non-zero.
- **H2b₀:** the paired difference in MCB is zero. **H2b₁:** the physics model has
  strictly larger MCB.
- **H2c₀:** the paired difference in DSC (discrimination) is zero.

Separating H2b from H2c is the point. The scientifically interesting outcome is
`MCB_physics > MCB_logistic` together with `DSC_physics ≈ DSC_logistic`: the
physics model identifies the same situations but misstates their probabilities.
The opposite pattern — better discrimination, worse calibration — would argue
that physics-based fields should be used as *features* inside a fitted model
rather than as probabilities in their own right, which is a constructive and
actionable conclusion.

### H3 — Spatial and contextual heterogeneity

- **H3₀:** MCB is constant across the pre-registered subgroup family.
  **H3₁:** MCB varies; specifically, we predict larger MCB in the attacking
  third, at longer flight times, and under higher pressure.

*Test:* stratum-level MCB with FDR control at 5% across the family. The
directional predictions are pre-registered so that an after-the-fact story about
whichever zone turns out significant is not available to us.

### H4 — Distribution shift

- **H4a₀:** MCB under a leave-one-competition-out split equals MCB under a
  within-competition match split. **H4a₁:** MCB is larger out of competition.
- **H4b₀:** MCB on broadcast-derived tracking equals MCB on optical tracking.
  **H4b₁:** MCB is larger on broadcast tracking.

H4b is confounded by design: the two modalities cover different matches. The
confound is addressed, not assumed away, by the degradation-simulation ablation;
agreement between the observational contrast and the simulated degradation is
the evidence that the effect is measurement rather than sampling.

### H5 — Recalibration

- **H5a₀:** Platt scaling does not improve out-of-sample Brier score.
  **H5a₁:** it does.
- **H5b₀:** isotonic regression gives no further improvement over Platt.
  **H5b₁:** it does — which would indicate non-logit-linear distortion.
- **H5c₀:** a map fitted on competition A transfers without loss to competition
  B. **H5c₁:** it does not.
- **H5d₀:** partially pooled per-stratum recalibration gives no improvement over
  a single global map. **H5d₁:** it does.

H5a is deliberately a weak hypothesis; a two-parameter map almost always helps a
miscalibrated forecast. The informative results are H5b (is the distortion
simple?) and H5c (does the fix transfer, or must every club refit?).

### H6 — Downstream decision quality

- **H6a₀:** mean expected possession value computed from raw and recalibrated
  control is equal. **H6a₁:** it differs.
- **H6b₀:** the fraction of threshold-based pass recommendations that change
  under recalibration is zero. **H6b₁:** it is positive.
- **H6c₀:** net benefit at practitioner-relevant thresholds is equal for raw and
  recalibrated forecasts. **H6c₁:** recalibrated forecasts have higher net
  benefit.

H6c is the only one of the three that speaks to *decision quality* rather than
sensitivity. H6a and H6b can both be resoundingly rejected while H6c is not, and
that combination — the numbers move a lot but the decisions are no better —
would itself be an important and publishable finding about the limits of
recalibration as a remedy.

---

## 6. Literature review

This review is organised around the *gap*, not around a chronology. Its purpose
is to establish three things: that pitch-control values are read as
probabilities; that the statistical apparatus for testing such a reading is
mature and well understood outside football; and that the two have not been
brought together.

**Citation discipline.** Every reference below is one whose existence and
substance the author is confident of. Where the venue or year is uncertain, the
entry is marked `[citation required]` and must be verified against the primary
source before submission. Full entries are in `docs/references.bib`. No
reference has been reconstructed from partial memory and presented as verified.

### 6.1 Geometric dominant regions

The oldest formulation partitions the pitch by which player could reach each
point first. With instantaneous reachability determined by Euclidean distance
this is the Voronoi tessellation; with a motion model it becomes a "dominant
region" (Taki and Hasegawa, 2000; Fujimura and Sugihara, 2005). Gudmundsson and
Horton's (2017) survey of spatio-temporal analysis in team sports remains the
best entry point to this line.

*Critical assessment.* Dominant regions are a partition, not a probability: each
point belongs wholly to one team. As a probability forecast such a field is
maximally overconfident, and any error it makes is maximally penalised by a
proper score. This is not a defect of the original work — it was never presented
as a probability — but it becomes one when the partition is used as a
probability by downstream tooling, which it routinely is whenever an analyst
shades a pitch by "whose space this is" and then computes an area.

### 6.2 Physics-based control models

The standard modern formulation makes arrival time uncertain and lets control
accumulate dynamically. Spearman et al. (2017) introduced a physics-based model
of pass probabilities; Spearman (2018) extended it into off-ball scoring
opportunity. Fernández and Bornn (2018) developed a related space-occupation
formulation, and Fernández, Bornn and Cervone's expected-possession-value
framework (Sloan 2019; journal version in *Machine Learning*, 2021) embedded
control surfaces in a full possession-valuation model. Brefeld, Lasek and Mair
(2019) took an explicitly probabilistic route, learning movement models from
data and deriving zones of control from them — the closest existing work to a
statistically grounded control field. Widely used open implementations follow
Spearman's formulation (Shaw and Glickman, 2019 `[citation required]` for the
Barça Sports Analytics Summit version; the "Friends of Tracking" code base is
the practical reference for most users `[citation required]`).

*Critical assessment.* These models introduce genuine probabilistic structure —
a logistic arrival-time distribution, a competing-risks accumulation of control —
and their outputs are consequently smooth and plausible. But the parameters that
govern the probabilistic behaviour (the arrival-time scale σ, the control rate
λ, the sprint speed and reaction time) are set by assumption, not estimated from
outcomes. The resulting number is therefore a *model-implied* probability whose
agreement with observed frequencies has, as far as we can establish, never been
reported. Brefeld et al. are the partial exception in that their movement models
are learned; even there, the control field's calibration against realised
arrivals is not the reported quantity.

### 6.3 Possession value, action value and spatial threat

Decroos et al. (2019) introduced VAEP, valuing actions by their effect on
scoring and conceding probabilities within a possession; Singh's expected threat
is a widely used positional value grid, disseminated through a blog post rather
than a peer-reviewed venue `[citation required]`, which is itself relevant to
the field's citation practices. Power et al. (2017) modelled pass risk and
reward directly from event and tracking data. Anzer and Bauer (2022) modelled
pass success from synchronised positional and event data, and their earlier work
(2021) did the same for shot probability. Van Roy et al. (2020) evaluated
decision quality in soccer `[citation required]`. Merhej et al. (2021) valued
defensive actions `[citation required]`.

*Critical assessment.* This literature is closer to good forecasting practice
than the control literature is, because its targets (goal, pass completion) are
directly observed and its models are fitted. Anzer and Bauer's pass models in
particular are fitted, evaluated, probabilistic quantities. The gap is that
possession-value frameworks *consume* control fields as inputs without
propagating any uncertainty about whether those inputs are calibrated, so a
systematic bias in the control field enters the valuation silently. To our
knowledge no published possession-value paper reports a calibration diagnostic
for its control component.

### 6.4 Probabilistic calibration and proper scoring rules

The relevant statistics long predates football analytics. Brier (1950)
introduced the quadratic score; Murphy (1973) gave its decomposition into
reliability, resolution and uncertainty; DeGroot and Fienberg (1983) formalised
calibration and refinement for forecast comparison. Gneiting and Raftery (2007)
is the definitive treatment of strictly proper scoring rules, and Gneiting,
Balabdaoui and Raftery (2007) established the modern framing: maximise sharpness
*subject to* calibration. Van Calster et al. (2016) set out a calibration
hierarchy — mean, weak, moderate, strong — that is directly transferable here,
and Van Calster et al. (2019) argued the case for calibration as the neglected
criterion in applied prediction. The calibration slope and intercept trace to
Cox's work on the logistic function (1958) `[citation required for the exact
bibliographic details]`.

*Critical assessment.* Nothing in this apparatus is new, and that is the point.
The contribution available to a football paper is not methodological invention
but correct application to a domain where the apparatus has not been applied.

### 6.5 Reliability diagrams and the limits of ECE

Binned reliability diagrams are standard but bin-dependent. Naeini et al. (2015)
introduced Bayesian binning; Guo et al. (2017) popularised expected calibration
error in machine learning and documented modern neural networks' overconfidence.
The critique followed quickly: Nixon et al. (2019) proposed adaptive binning;
Vaicenavicius et al. (2019) and Kumar, Liang and Ma (2019) showed that binned
ECE is a biased estimator whose bias depends on the bin count in ways that do
not vanish usefully with sample size; Roelofs et al. (2022) quantified the bias
`[citation required]`. Dimitriadis, Gneiting and Jordan (2021) proposed the CORP
approach: reliability diagrams and score decompositions built on
pool-adjacent-violators isotonic regression rather than binning, giving a
reproducible, binning-free decomposition consistent with the reported score.

*Critical assessment.* This settles the study's primary-metric choice. ECE is
retained only for comparability with a literature that reports it, always with
its bin count attached and never as the basis of a claim. The primary statistic
is the CORP MCB term, cross-fitted by match to remove the upward bias that
in-sample isotonic fitting introduces — a bias our own test suite demonstrates
and quantifies on simulated data.

### 6.6 Recalibration

Platt (1999) introduced logistic scaling; Zadrozny and Elkan (2002) introduced
isotonic recalibration; Niculescu-Mizil and Caruana (2005) surveyed which model
families need which. Kull, Silva Filho and Flach (2017) introduced beta
calibration, which unlike Platt scaling can represent asymmetric tail
distortion. Kumar, Liang and Ma (2019) gave recalibration methods with
verifiable guarantees.

*Critical assessment.* The methods are well understood; what is not established
for this domain is whether the distortion in pitch-control fields is simple
(fixable by two parameters) or structured (varying by zone, competition and
tracking quality). That distinction is the substance of RQ5, and it has a direct
practical consequence: a global two-parameter fix is deployable, a
zone-and-competition-varying fix is a research programme.

### 6.7 Distribution shift and selective prediction

Quiñonero-Candela et al. (2009) is the standard reference for dataset shift.
Selective prediction — abstaining where the model is unreliable — is developed
in El-Yaniv and Wiener (2010) and Geifman and El-Yaniv (2017). Conformal
prediction (Vovk et al., 2005; Angelopoulos and Bates, 2023) offers
distribution-free coverage guarantees under exchangeability, an assumption that
match-level clustering in football violates in ways that are themselves worth
documenting.

*Critical assessment.* Relevant to this study in two ways: as the framing for
RQ4 (broadcast tracking is a shifted measurement regime), and as the natural
extension if the answer to RQ1 is "miscalibrated in identifiable regions" — a
model that abstains where it is unreliable is more useful than one that is
uniformly hedged.

### 6.8 Selection, propensity and counterfactuals

Horvitz and Thompson (1952) introduced inverse-probability weighting; Rosenbaum
and Rubin (1983) the propensity score; Robins, Hernán and Brumback (2000)
marginal structural models and stabilised weights. Hernán and Robins' textbook
is the standard modern treatment, and Pearl (2009) the structural-causal
framing. For sensitivity to unmeasured confounding, VanderWeele and Ding (2017)
introduced the E-value; Rosenbaum-style bounding under an odds-ratio sensitivity
parameter is the older tradition `[citation required for the specific form used
here]`. Cameron and Miller (2015) is the practical reference for cluster-robust
inference.

*Critical assessment.* This is where the study must be most careful not to
overclaim. Reweighting corrects *covariate shift* in where the ball arrives; it
does not correct selection on what the passer knew and the state vector did not.
The applied literature frequently elides this. Our design responds with a
quasi-exogenous subsample rather than relying on weighting alone, and reports a
sensitivity bound rather than an adjusted point estimate presented as unbiased.

### 6.9 Football tracking-data methodology

Rein and Memmert (2016) survey the promise and the methodological hazards of
tracking data in elite football. Pappalardo et al. (2019) published an open
event dataset with an explicit data-descriptor treatment of quality. Giancola
et al. (2018) and Deliège et al. (2021) established SoccerNet as a benchmark for
video-derived football understanding. Kurach et al. (2020) released the Google
Research Football environment. Wang et al. (2024) demonstrated geometric deep
learning on corner kicks in *Nature Communications*.

*Critical assessment.* The methodological literature on tracking-data quality —
synchronisation error, occlusion, identity switches, derived-velocity artefacts —
is thin relative to the modelling literature that depends on it. This study
treats preprocessing as a first-class source of uncertainty and reports the
spread of its headline statistic across plausible preprocessing settings, which
is itself a small contribution to that thin literature.

### 6.10 Where this study contributes

Bringing the strands together, the gap is specific and, we believe, real:

1. **No published evaluation of pitch-control fields as probability forecasts.**
   The control literature reports fields and downstream applications; the
   forecasting literature supplies the tools; they have not met. We are not
   aware of a paper reporting a reliability diagram or a proper-score
   decomposition for a pitch-control model. *This claim is a literature-search
   result and must be re-verified immediately before submission; the appropriate
   framing in the paper is "we are not aware of", not "there is no".*
2. **No operational definition of the control construct.** The study proposes,
   defends and tests sensitivity to one.
3. **Selection bias in control evaluation is unaddressed.** The observation that
   passers choose destinations, and what follows for evaluating a field defined
   everywhere, appears not to have been treated.
4. **No propagation of calibration error into downstream metrics.** The
   linearity argument of Section 3.4 — that systematic control error accumulates
   rather than averaging out — is, as far as we know, not made anywhere in the
   applied literature.
5. **Binned ECE would be the natural but wrong tool.** Importing the CORP
   framework is a small but genuine methodological transfer into sports
   analytics.

The contribution is therefore evaluative and methodological, not a new model. We
regard that as a feature: the field has many models and few instruments for
telling whether they mean what they say.

---

## 7. Conceptual framework

### 7.1 Definitions

The following are the study's operational definitions. They are stated before
any analysis and are not revised in light of results.

| Term | Definition |
|---|---|
| **Pitch control** | A field `C_A(x, y, t) ∈ [0, 1]` over the playing surface, interpreted as the probability that team A would end up in control of the ball if the ball arrived at `(x, y)` at time `t`, given the state at the decision moment. |
| **Ball-control probability** | The same quantity evaluated at a *single* realised arrival point and time. This, not the field, is what this study can measure. |
| **Arrival** | An event at which the ball reaches a location after a period of flight or travel initiated by a player action: a pass reception, an interception, a cross, a deflection, a clearance, a second ball. |
| **Reception** | An arrival at which a player of the passing team takes the ball under control. |
| **Interception** | An arrival at which a player of the opposing team takes the ball under control. |
| **Loose-ball recovery** | An arrival after which neither team immediately controls the ball, resolved by a subsequent contest. Requires explicit treatment: coding it as a "loss" for the passing team is a modelling decision, not a fact. |
| **Decision time `t₀`** | The moment the ball is released. All model inputs are taken here. |
| **Arrival time `t₁`** | The moment the ball reaches the destination. `t₁ − t₀` is the flight time. |
| **Control horizon `h`** | The interval after `t₁` at which control is assessed. Primary value `h = 1.0 s`. |
| **Outcome `Y`** | `1` if team A is in control at `t₁ + h`, else `0`. |
| **Calibration** | The property that `E[Y | C_A = c] = c` for all `c` in the support of the forecast. |
| **Sharpness / refinement** | The dispersion of the forecast distribution, a property of the forecast alone. Measured by forecast variance and by the DSC term of the CORP decomposition. |
| **Distribution shift** | A difference between the joint distribution of state and outcome in the fitting data and in the deployment data — across competitions, seasons, or tracking modalities. |

### 7.2 Three distinct questions that look like one

Much of the conceptual confusion in this area collapses three different
quantities. They are not interchangeable and they do not have the same value.

**(i) "Which team is more likely to reach the location first?"**
A question purely about locomotion. It depends on positions, velocities and a
kinematic envelope, and it is in principle answerable from physics alone. It is
what a dominant-region model computes. It says nothing about what happens when
the player gets there.

**(ii) "Which team will actually possess the ball after it arrives?"**
This adds everything that happens at the point of contact: the first touch, the
body orientation, the contest, the bounce, the aerial duel, the ball's height
and spin. A defender who arrives first but is facing the wrong way, or who can
only reach the ball with the outside of a boot, frequently does not win it. This
is the quantity the probabilistic reading of `C_A` implicitly claims, and it is
strictly harder than (i).

**(iii) "Which team will retain possession after receiving it?"**
This adds the quality of the subsequent decision — the receiver's next action.
A player who controls a pass with three opponents converging and is dispossessed
two seconds later did control the arrival. This quantity is of most interest to
a coach and is furthest from what a control field computes.

Reachability is necessary but not sufficient for possession; possession is
necessary but not sufficient for retention. So `C^{(i)} ≥ C^{(ii)} ≥ C^{(iii)}`
is the expected ordering in aggregate, though not pointwise.

### 7.3 What is this study measuring?

Given the above, the honest answer is: **reception control**, formalised as
possession at a short fixed horizon after arrival. This is question (ii), not
(i) and not (iii).

The recommendation and its defence:

> **Primary operational definition.** `Y = 1` if and only if a player of team A
> is in uninterrupted possession of the ball at time `t₁ + 1.0 s`, with arrivals
> whose horizon window is interrupted by a stoppage treated as *censored and
> excluded*, not as failures.

Four reasons:

1. **It is what the probabilistic reading commits to.** Every downstream use —
   multiplying by a positional value, thresholding to recommend a pass — is
   about whether the team gets the ball, not about who could theoretically run
   there fastest.
2. **It is reproducible across providers.** Possession at a timestamp can be
   derived from any provider's stream, whereas "first touch" depends on touch
   attribution that is unreliable for contested aerial balls.
3. **One second is short enough to exclude the next decision.** It admits a
   deflected touch followed by a recovery, but not a controlled turn and a
   subsequent giveaway.
4. **It is falsifiable and the sensitivity is testable.** The study reports the
   same analysis at `h ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}`. If conclusions flip
   across that range, the claim "pitch control is miscalibrated" is
   under-specified and the paper must say so rather than pick the horizon that
   supports it.

**A necessary concession.** Under this definition a model that perfectly answers
question (i) *should* appear miscalibrated, because (i) and (ii) are different
quantities. A defender of the physics models could reasonably say the study is
scoring them against a target they never claimed. The response is twofold, and
the paper must make it explicitly rather than leave it to a reviewer. First, the
downstream literature does treat these fields as answering (ii); if they only
answer (i), that is itself the finding, and it invalidates the downstream uses
rather than excusing the models. Second, we report the analysis under the
first-touch definition as well, which is the closest observable proxy for (i),
so that a reader can see how much of any measured miscalibration is
construct mismatch and how much is model error.

### 7.4 Directed acyclic graph of the measurement problem

```
            (unobserved)
          passer's private
          information  U
             │      │
             │      └──────────────┐
             ▼                     ▼
   destination choice D ──────►  outcome Y
             ▲                     ▲
             │                     │
     observed state X ─────────────┘
             │
             ▼
     model forecast C = f(X)
```

The study observes `(X, C, Y)` only on the subset where `D = 1`. The arrow
`U → Y` alongside `U → D` is precisely what prevents reweighting on `X` from
recovering `P(Y | X)`. If `U` were absent, selection would be on observables
and reweighting would suffice. Section 14 is organised around this diagram, and
the quasi-exogenous subsample is the design that severs `U → D`.

---

## 8. Formal mathematical formulation

### 8.1 State

Fix a match. Let `𝒫 = 𝒜 ∪ ℬ` be the players, `𝒜` the team in possession and
`ℬ` the opponent. At time `t`, player `j` has state

```
s_j(t) = ( r_j(t), v_j(t), a_j(t), θ_j(t) ) ∈ ℝ² × ℝ² × ℝ² × S¹
```

with position, velocity, acceleration and body orientation; the last two are
optional and usually unavailable, and the study reports which were present. The
ball has state `s_0(t) = (r_0(t), v_0(t))`. The match state is
`S_t = ( {s_j(t)}_{j∈𝒫}, s_0(t), c_t )` where `c_t` collects contextual
variables (score, period, clock, numerical advantage, set-piece flag).

The pitch is `Ω = [−L/2, L/2] × [−W/2, W/2]` with the origin at the centre spot
and `+x` toward the goal team `𝒜` attacks.

### 8.2 The estimand

A pitch-control model `M` is a map

```
C_A( x, y, t ; S_t, M )  =  P_M ( Y = 1 | x, y, t, S_t )
```

The study's question is whether the left-hand side, as computed by `M`, equals
the right-hand side, as realised in data. Formally, `M` is **calibrated** on a
population of arrivals if

```
E [ Y | C_A(X) = c ]  =  c        for almost every c in the support of C_A(X)
```

where the expectation is over the arrival population. This is *weak* (or
"moderate") calibration in the hierarchy of Van Calster et al.: the conditioning
is on the forecast value, not on the full covariate vector. Strong calibration —
`E[Y | X] = C_A(X)` for all `X` — is not testable at realistic sample sizes and
is not claimed as the estimand.

### 8.3 The outcome variable

Let `t₀` be release, `t₁` arrival, `h` the horizon, and `π(t) ∈ {𝒜, ℬ, ∅}` the
team in possession at `t`. Four definitions, with their assumptions:

| Definition | Formula | Assumes |
|---|---|---|
| **D1 First touch** | `Y = 1[ first player to touch after t₁ ∈ 𝒜 ]` | Touch attribution is reliable; a touch equals control; a touch that immediately concedes possession still counts. |
| **D2 Control at horizon** *(primary)* | `Y = 1[ π(t₁ + h) = 𝒜 ]` | A frame-level or event-derivable possession label exists and is consistent across matches; `h` is meaningful and fixed. |
| **D3 Retained sequence** | `Y = 1[ 𝒜 completes ≥ 1 further controlled action before losing the ball ]` | Possession-sequence boundaries are well defined; conflates control with the next decision's quality. |
| **D4 Possession at 5 s** | `Y = 1[ π(t₁ + 5) = 𝒜 ]` | As D2 with a long horizon; drifts toward "sustained possession" and away from control. |

`D2` is primary. `D1` is reported as the closest proxy for the reachability
construct (Section 7.3). `D3` and `D4` are robustness checks. Censoring: if a
stoppage occurs in `(t₁, t₁ + h]`, the arrival is censored and excluded. Coding
such arrivals as failures would bias every model downward exactly in the zones
where fouls concentrate.

### 8.4 Time-to-point models

For a target `x ∈ Ω`, reaction time `t_r` and sprint speed `v_max`:

**Constant speed** (the form used by the published physics models):

```
τ_j(x) = t_r + ‖ x − ( r_j + v_j · t_r ) ‖ / v_max
```

**Bounded acceleration** (recommended default). Let `d = ‖x − (r_j + v_j t_r)‖`,
let `u = max(0, ⟨v_j, x̂⟩)` be the along-path component of current velocity
clipped at zero and `v_max`, and `d_acc = (v_max² − u²) / (2a)`. Then

```
τ_j(x) = t_r +  ( −u + √(u² + 2 a d) ) / a                  if d ≤ d_acc
       = t_r +  (v_max − u)/a  +  (d − d_acc)/v_max          if d >  d_acc
```

Clipping `u` at zero is deliberate: a player running away from the target is
credited with zero useful initial speed rather than a negative one, which would
otherwise reward the model for assuming instantaneous reversal. The bounded
form is never faster than the constant-speed form (asserted in the test suite),
and the difference between them is an ablation axis rather than an assumption.

**Arrival probability.** With arrival-time uncertainty `σ`:

```
f_j( t | x ) = [ 1 + exp( − π ( t − τ_j(x) ) / ( √3 · σ ) ) ]⁻¹
```

The `π/√3` factor makes `σ` the standard deviation of the underlying logistic,
so the parameter is interpretable as "seconds of uncertainty about arrival time"
rather than an arbitrary temperature. `σ` absorbs path curvature, contact,
deceleration into a contest, and tracking error. **It is a nuisance parameter
and must be either estimated or ablated, never fixed and then reported as
physics.**

### 8.5 Model 1 — Voronoi / nearest-player baseline

```
C_A(x) = 1 [  min_{j ∈ 𝒜} τ_j(x)  <  min_{k ∈ ℬ} τ_k(x)  ]
```

With `v_j ≡ 0` and constant-speed `τ`, this is the Euclidean Voronoi partition.
Assumptions: control is decided by first arrival; ties are measure-zero;
arrival is deterministic. As a probability forecast it takes only the values 0
and 1, so its log loss is infinite whenever it is wrong; the implementation
clips to `[ε, 1−ε]` and states `ε`, and the Brier score is preferred as primary
for this reason.

### 8.6 Model 2 — Physics-based control with ball-control dynamics

Let `PPC_j(t)` be the probability that player `j` has taken control by time `t`,
and `λ_j` the rate at which a present player converts presence into control
(higher for goalkeepers, who may use hands). Integrating from the ball's arrival:

```
d PPC_j / dt  =  ( 1 − Σ_{k ∈ 𝒫} PPC_k(t) ) · f_j( t | x ) · λ_j
C_A(x) = Σ_{j ∈ 𝒜} PPC_j(∞)
```

This is a competing-risks construction: all players draw from a common pool of
remaining probability mass. Free parameters: `v_max, t_r, σ, λ`. In the default
configuration **none is estimated from outcome data**, which is the property
under test.

Two implementation details that are usually left implicit and that the study
makes explicit and ablates:

- *Integration.* Explicit Euler with step `dt`; too coarse a step over-allocates
  mass per step. The implementation rescales any step that would exceed the
  remaining mass, and `dt` is swept.
- *Unassigned mass.* Truncating the integration leaves `1 − Σ_k PPC_k > 0`, so
  `C_A + C_B < 1`: the output is not a distribution over the two outcomes. The
  implementation reports the unassigned mass as a diagnostic and optionally
  renormalises. A model requiring substantial renormalisation to sum to one is
  already failing the probabilistic reading, independently of calibration.

### 8.7 Model 2a — Reachability sigmoid (a minimal soft baseline)

```
C_A(x) = σ( ( min_k τ_k(x) − min_j τ_j(x) ) / s )
```

One free parameter `s`, fitted by minimising the Brier score on training
arrivals. No dynamics, no multi-player accumulation. Its purpose is to answer a
question the headline comparison cannot: how much of the physics model's
calibration comes from its dynamics, and how much simply from monotonically
squashing a time-to-point difference? If this model matches Model 2, the
dynamics are decorative.

### 8.8 Model 3 — Penalised logistic regression

```
logit P( Y = 1 | z )  =  β₀ + βᵀ z
```

with `z` a vector of engineered features computed from *exactly* the information
the geometric models receive — reachability gaps, distances to nearest players,
local player counts, closing speeds, flight time, pass geometry, destination
coordinates. No event-stream annotations and no post-arrival information, so a
win for this model cannot be attributed to extra data.

L2 penalty strength selected by grouped cross-validation on log loss. The
penalty is a genuine threat to calibration — shrinkage flattens the linear
predictor and pushes the calibration slope above one — so the resulting slope is
reported rather than assumed.

### 8.9 Model 4 — Gradient-boosted trees

Histogram-based gradient boosting on the same features, minimising log loss (a
strictly proper rule). Pre-registered expected failure mode: boosted trees
fitted on a proper loss are typically well calibrated *in distribution* but
degrade under covariate shift, because a piecewise-constant fit cannot
extrapolate beyond its training support. This makes Model 4 the natural carrier
of the RQ4 and cross-competition contrasts.

### 8.10 Model 5 (optional) — Permutation-invariant set model

A Deep Sets architecture: a shared per-player encoder `φ` applied to
destination-relative player features, mean-and-max pooling, concatenation with
global context, then `ρ` to a logit. It receives no hand-built notion of
"nearest player" and can in principle discover interactions the engineered
features cannot express. Marked optional; the minimum viable study omits it,
because the central claim concerns whether *published* models are calibrated and
a bespoke network is not one.

### 8.11 Hierarchical / mixed-effects formulation

To separate match-level and team-level variation in calibration from noise:

```
logit P( Y_{im} = 1 )  =  ( α + a_m ) + ( β + b_m ) · logit C_A( x_{im} )
( a_m , b_m ) ~ N( 0 , Σ )
```

for arrival `i` in match `m`. The population slope `β` is the calibration slope;
`Var(b_m)` measures between-match heterogeneity in calibration and directly
tests whether "the model is overconfident" is a property of the model or of
particular matches. Fitted by MCMC (Stan, or `brms` in R); this is the
principled version of the partially pooled recalibrator implemented in code, and
it is optional for the minimum viable study.

### 8.12 Calibration models

Given raw forecasts `p` and a map `g`:

| Map | Form | Parameters |
|---|---|---|
| Platt | `g(p) = σ( a + b · logit p )` | 2 |
| Beta | `g(p) = σ( c + a·ln p − b·ln(1−p) )` | 3 |
| Isotonic | `g = ` PAV fit of `Y` on `p`, monotone | non-parametric |
| Partially pooled | per-stratum Platt with `ω_s = n_s /(n_s + κ)` shrinkage toward the pooled map | 2 per stratum + κ |

**Protocol constraint, and it is not negotiable:** every map is fitted on a
validation fold disjoint from both the fitting data and the test data, split by
match. Fitting and evaluating isotonic regression on the same arrivals makes any
model look perfectly calibrated; the test suite includes that demonstration so
the hazard is documented rather than merely asserted.

### 8.13 Handling the kinematic parameters

| Quantity | Treatment |
|---|---|
| Reaction time `t_r` | Assumed constant (default 0.7 s); ablated over `{0, 0.3, 0.7, 1.0}`. |
| Sprint speed `v_max` | Per-player estimate from the 99.5th percentile of observed speed where enough frames exist, with a floor; squad-level default otherwise. A high quantile rather than the maximum, because the maximum of a noisy derivative estimates the noise. |
| Acceleration `a_max` | Assumed bound (default 7 m/s²); ablated. |
| Arrival uncertainty `σ` | The critical nuisance parameter. Ablated over `{0.2, 0.45, 0.8, 1.2}`; a variant fitted by minimising the Brier score is reported separately and labelled as fitted. |
| Control rate `λ` | Assumed; ablated. Goalkeepers get a higher rate. |
| Travel time | Measured from ball tracking where available; otherwise imputed from a constant-speed model and **flagged as imputed**, since an imputed flight time makes the conditioning partly circular. |
| Velocity | Never observed. Estimated by Savitzky–Golay differentiation; the filter bandwidth is an ablation axis. |

---

## 9. Model comparison

### 9.1 The comparison table

All five models answer the same question from the same inputs. That is what
makes the comparison fair, and it is enforced in code by a common interface
(`pcc.models.base.ControlModel`): a learned model sees exactly the state an
unlearned one sees, so any advantage is attributable to functional form and
fitting rather than to extra information.

| | **M1 Voronoi** | **M2 Physics** | **M3 Logistic** | **M4 GBM** | **M5 Deep Sets** *(optional)* |
|---|---|---|---|---|---|
| **Inputs** | positions, velocities, target | + flight time, GK flags | engineered features from the same state | same features as M3 | raw per-player relative vectors |
| **Output** | `{0,1}` (clipped) | `[0,1]` | `[0,1]` | `[0,1]` | `[0,1]` |
| **Fitted?** | No | No (parameters assumed) | Yes | Yes | Yes |
| **Key assumptions** | control = first arrival; deterministic | logistic arrival times; competing-risks control; assumed `σ, λ, v_max, t_r` | logit-linear in engineered features | smooth conditional mean, learnable by axis-aligned splits | permutation invariance over players |
| **Training need** | none | none | seconds | minutes | GPU-minutes; grouped early stopping |
| **Interpretability** | complete | high — every parameter has units | high — signed standardised coefficients | moderate — needs attribution methods | low |
| **Expected strengths** | transparent; the reference for sharpness | smooth, physically plausible, transfers across leagues without refitting | calibrated by construction in-sample; stable under shift | best in-distribution discrimination | can capture multi-player interactions |
| **Expected failure modes** | maximal overconfidence; infinite log loss when wrong | miscalibrated scale; sensitive to `σ`; blind to contact, orientation, ball height | linearity in logit may be wrong in the tails; penalty inflates the slope | degrades under covariate shift; cannot extrapolate | overconfident when trained past fit; needs the most data |
| **Role in the study** | sharpness reference | the object under test | **the adversary** — the baseline H2 is stated against | upper bound on in-distribution performance | upper bound on achievable calibration |

**M0, the marginal baseline**, predicts the training base rate everywhere. It is
perfectly calibrated and completely uninformative, and it is included precisely
to make that point concrete in the results table: any model that does not beat
M0 on the Brier score has no skill, and M0's own MCB of approximately zero is
the standing demonstration that calibration alone is not a sufficient criterion.

**M2a, the reachability sigmoid**, is the diagnostic described in Section 8.7.

### 9.2 Preventing leakage

Arrivals are nested: within possessions, within matches, within team pairings,
within competitions. Three leakage channels operate simultaneously under a naive
random row split:

1. **Within-possession duplication.** Consecutive passes in one possession share
   almost the same defensive shape. A random split puts pass `k` in training and
   pass `k+1` in test; the test observation is a near-copy of a training one.
2. **Within-match structure.** Teams hold a shape for ninety minutes. A model can
   learn "this match's low block" and score well on held-out arrivals from the
   same match without learning anything transferable.
3. **Team identity.** With enough features a model can partially identify the
   teams; only cross-team or cross-competition splitting shows the result is
   about football rather than about these twenty-two players.

The defences, all implemented and tested:

- Splits operate on **groups** — possession, match, competition — never rows.
  `pcc.evaluation.splits.Split.assert_disjoint` raises if any group appears on
  both sides, and the test suite includes a deliberately leaky split to confirm
  the guard actually fires.
- The nested validation fold is carved out of the **training** groups, never the
  test fold. Hyper-parameters, the reachability-sigmoid scale, and every
  recalibration map are selected there.
- A **leakage audit** table (group counts and overlaps per fold) is written with
  every experiment's results. A leakage check that lives only in a code review is
  not reproducible evidence.
- The headline numbers come from the strictest split the sample supports:
  leave-one-competition-out where more than one competition is available,
  match-level otherwise.
- `σ`-fitting, when reported, is done on the training fold only and the resulting
  model is labelled "M2 (fitted σ)" so it is never confused with the published,
  unfitted formulation.

### 9.3 What a fair comparison requires beyond splitting

- **Identical test arrivals.** All models are scored on the same rows, and
  comparisons are *paired* cluster bootstraps, which absorbs the shared
  match-level variation and is materially more powerful than an unpaired test.
- **Identical outcome definition.** One `Y`, one horizon, applied to all.
- **Identical preprocessing.** One velocity filter, one synchronisation, one
  inclusion rule.
- **Separate reporting of calibration and discrimination.** A model is never
  declared better on a composite score alone; MCB and DSC are reported side by
  side, because a model can lower MCB by flattening toward the base rate and the
  DSC column is what exposes that.

---

## 10. Data requirements

### 10.1 Variable-level requirements

`R` = required for the primary design; `O` = optional but valuable; `F` =
sufficient for the freeze-frame fallback design.

| Variable | Definition | Need | Type | Sampling | Source | Why it is needed | Measurement error |
|---|---|---|---|---|---|---|---|
| `match_id` | Unique match key | R | string | per match | metadata | Primary clustering unit for all inference | none |
| `competition` | Competition and season | R | string | per match | metadata | Cross-competition split; distribution-shift stratum | none |
| `period` | 1–4 | R | int | per frame | tracking | Clock resets; prevents cross-period time arithmetic | none |
| `timestamp` | Seconds from period start | R | float | 10–30 Hz | tracking | Aligns events with frames | clock drift; provider-dependent |
| `pitch_length`, `pitch_width` | True pitch dimensions (m) | R | float | per match | metadata | Metric normalisation; a template frame introduces metric error into every `τ` | often unrecorded; template values assumed |
| `player_id` | Stable player key | R | string | per frame | tracking | Per-player `v_max`; identity-switch detection | identity switches in congested play |
| `team` | Team of the player | R | string | per frame | tracking | Partition into 𝒜 and ℬ | none |
| `is_goalkeeper` | GK flag | R | bool | per match | metadata | GKs get a different control rate (hands) | none |
| `x`, `y` | Player position (m, canonical frame) | R | float | 10–30 Hz | tracking | Every `τ` and every feature | optical ~0.1 m; broadcast substantially worse and unquantified |
| `vx`, `vy` | Player velocity (m/s) | R (derived) | float | 10–30 Hz | **derived** | Reachability; the physics models' main input | *entirely* filter-dependent; not a measurement |
| `ax`, `ay` | Player acceleration | O | float | 10–30 Hz | derived | Bounded-acceleration `τ` | second derivative of noise; very unreliable |
| `orientation` | Body orientation | O | float | 10–30 Hz | tracking/pose | A defender facing away wins fewer contests | rarely supplied; pose estimation needed |
| `ball_x`, `ball_y` | Ball position | R | float | 10–30 Hz | tracking | Arrival detection; measured flight time | dropout during occlusion and flight |
| `ball_z` | Ball height | O | float | 10–30 Hz | tracking | Separates aerial from ground arrivals — a major source of heterogeneity | often absent |
| `ball_speed` | Ball speed | O | float | 10–30 Hz | derived | Flight-time fallback when ball tracking drops out | derived |
| `event_type` | Provider event label | R | string | per event | events | Arrival typology; the endogenous/exogenous split | provider taxonomies differ; **must be mapped explicitly** |
| `origin_x`, `origin_y` | Release location | R | float | per event | events | Pass geometry; candidate-destination proposal | event–tracking spatial disagreement |
| `dest_x`, `dest_y` | Arrival location | R | float | per event | events/tracking | **The evaluation point.** | provider-annotated; may differ from the tracked ball |
| `t_release` | Release time | R | float | per event | events | State is taken here | synchronisation error |
| `t_arrival` | Arrival time | R | float | per event | tracking preferred | Forecast horizon | **imputed when ball tracking is absent — flag it** |
| `pass_completed` | Provider completion flag | O | bool | per event | events | Cross-check on `Y`; *not* `Y` itself | label noise; different definitions |
| `interception` | Interception flag | O | bool | per event | events | Arrival typology | as above |
| `possession_team` | Team in possession | R | string | per frame or event | tracking/events | **Constructs `Y`** | definition varies by provider |
| `possession_id` | Possession-sequence key | R | string | per event | events | Nested clustering unit | boundary definitions vary |
| `pressure` | Opponents within 5 m of release | R (derived) | float | per event | derived | Pre-registered subgroup | derived |
| `player_role` | Position/role | O | string | per match | metadata | Team-style analysis | coarse; changes within a match |
| `score_diff` | Goals for − against at release | O | int | per event | events | Game-state subgroup | none |
| `minute` | Match minute | O | float | per event | events | Fatigue; temporal analysis | none |
| `n_players_a/b` | Players on the pitch | R | int | per frame | tracking | Red cards change the control problem entirely | none |
| `set_piece` | Dead-ball restart flag | R | bool | per event | events | Analysed separately; physics assumptions fail | none |
| `tracking_source` | optical / broadcast | R | string | per match | metadata | RQ4 | none |
| `frame_completeness` | Fraction of 22 players observed at `t₁` | R | float | per event | derived | The key quality covariate for RQ4 | none (it *is* the quality measure) |
| `sync_offset` | Estimated event–tracking offset applied | O | float | per match | derived | Residual synchronisation as a covariate | the estimate itself is uncertain |

### 10.2 Minimum viable versus ideal datasets

**Minimum viable.** Synchronised player *and* ball tracking plus an event stream,
for enough matches that a match-clustered bootstrap has enough clusters to mean
something, with a derivable possession label. Concretely: `match_id`,
`timestamp`, `player_id`, `team`, `is_goalkeeper`, `x`, `y`, ball `x`, `y`,
`event_type`, release and arrival coordinates and times, and possession. Velocity
is derived. This supports RQ1, RQ2, RQ5, RQ6 and a coarse version of RQ3.

**Ideal.** The above plus ball height, body orientation, per-match true pitch
dimensions, provider-supplied tracking-quality flags, ≥ 2 competitions across
≥ 2 seasons, and both optical and broadcast tracking **of the same matches** —
which would turn RQ4 from a confounded observational contrast into a clean
paired measurement comparison and is by some distance the single most valuable
addition.

**What is *not* needed,** and is worth saying because it is often assumed: video,
pose estimates, physiological data, or possession-value labels. The study's
target is directly observed.

### 10.3 Candidate sources — status after verification

> **What changed.** Section 10.3 originally listed every source as unverified.
> Three have since been obtained and opened, and the entries below distinguish
> what was *read from the files* from what is still taken on documentation.
> `scripts/01_check_data_availability.py` writes
> `results/data_availability.json`; `src/pcc/data/sources.py` carries the
> machine-readable record, and `docs/data_sources.md` the full checklists.

| Source | Modality | Status | Study role |
|---|---|---|---|
| PFF FC 2022 World Cup release | optical | **Unverified.** Access route, availability, licence and contents have *not* been checked. | Intended primary corpus |
| Metrica Sports sample | optical | **Verified.** Loader implemented and tested against the files. | Development corpus — confirmed too small for inference |
| SkillCorner open data | broadcast | **Verified reachable**, structure inspected; loader not yet written. | Broadcast arm of RQ4 |
| StatsBomb Open Data | event + 360 | **Verified.** Structure inspected; loader not yet written. | Fallback design |
| `pcc.data.synthetic` | simulated | Generated here. | Instrument validation only |

#### What verification established, and what it costs the design

**Metrica** (two readable matches, 25 fps, 105 × 68 m). Three findings matter.
Tracking and events are *already synchronised* — events carry frame numbers —
so the clock-offset step is a no-op, removing one of the pipeline's larger error
sources. There is **no frame-level possession label**, so the outcome is derived
from the event stream, which is a paired `BALL LOST` / `RECOVERY` transfer log;
the derivation is recorded in every row's `provider` field. And the corpus is
**two matches, hence two bootstrap clusters** — which confirms, rather than
merely predicts, Section 24.5's judgement that Metrica is a development corpus.
With two matches a three-way match-level split is impossible, so there is no
validation fold and RQ5 cannot be run on it at all. The analysis script now
refuses to present such a run as anything but a pipeline demonstration.

A sanity check worth recording, because it is the strongest available evidence
that the derived labels are sound: control rates order as football requires —
open passes 0.93, clearances 0.23, interceptions 0.15.

**SkillCorner** differs substantially from the description this proposal
originally carried. It is 20 match directories of Australian A-League 2024/25,
not nine matches of European football, and it now ships a derived-events file, a
phases-of-play file and 3D body pose for two matches. Two consequences. First,
the tracking JSONL **does** carry a per-frame `possession` object, which the
checklist had listed as unknown. Second, the tracking files are **Git LFS
pointers**: a plain clone yields ~130-byte stubs, and the real bytes must be
fetched from `media.githubusercontent.com`. An adapter written against the stubs
would fail confusingly far downstream. Note also that the derived-events file
carries SkillCorner's own pass-completion and possession-value models; using
those as inputs would contaminate the comparison, so only raw positional and
outcome fields may be used.

**StatsBomb** has 80 competition-seasons, of which **12 carry 360 freeze
frames** — including **FIFA World Cup 2022**, the same tournament as the PFF
release. That coincidence is worth more than it first appears: it makes the
fallback a *same-tournament* comparison, so the velocity-free and
velocity-bearing analyses could in principle be run on the same matches, which
turns the "remove velocity" ablation into a measurement rather than a
simulation. Verified structure: each frame carries `event_uuid`, a
`visible_area` polygon, and a `freeze_frame` of *visible players only*, each with
`teammate` / `actor` / `keeper` flags and a location but **no identity**. The
`visible_area` finding confirms the checklist's central worry: "no defender near
the destination" can mean "no defender visible", which would bias control
upward exactly where it matters. Any adapter must use `visible_area` to mark
destinations outside the covered region rather than treating absence as empty
space.

#### A methodological note on the availability check

The original availability probe reported all three sources unreachable. That was
wrong, and instructively so: it probed each repository's HTML landing page,
which this environment's egress proxy blocks with HTTP 403, while raw content
and `git` were reachable throughout. The check now probes a small data-bearing
endpoint instead, and distinguishes a proxy block from a missing dataset. The
general lesson is the one this section is built on — **a negative result from an
unvalidated instrument is not evidence** — and it applies to the study's own
calibration estimator exactly as much as to its availability script, which is
why `scripts/09_validate_instrument.py` exists.

#### Consequence for the design

The primary corpus remains unobtained. The fallback design is therefore not a
contingency but a live option, and the availability of World Cup 2022 in both
the intended primary source and the fallback source is the strongest argument
for pursuing the fallback first: it is obtainable today, it is large, and if the
optical release is later secured the two can be compared on the same matches.

---

## 11. Data preprocessing

The pipeline is implemented in `pcc.data.preprocess`. Its design principle is
that **preprocessing uncertainty must be measurable**, because the choices below
plausibly move the calibration statistics by more than the difference between
the models being compared — which our simulated ablation already demonstrates is
possible.

### 11.1 Steps, in order

1. **Coordinate normalisation.** Provider coordinates → a centred metric frame
   `[−L/2, L/2] × [−W/2, W/2]`. Where the provider uses a fixed template rather
   than true metres, the rescaling introduces metric error into every `τ`; the
   assumption is recorded per match and swept in the ablation.
2. **Attacking-direction normalisation.** Rotate so the team in possession always
   attacks `+x`. Control is asymmetric in `x` (offside line, block depth,
   goalkeeper sweeping); without this, every spatial statistic averages two
   mirror-image populations.
3. **Event–tracking synchronisation.** Estimate a per-match constant offset by
   cross-correlating event timestamps against peaks in ball-speed acceleration.
   A 0.3 s offset moves a sprinting defender ~1.5 m, which would be charged to
   the model as miscalibration. Residual offset is retained as a covariate, so a
   subgroup analysis by synchronisation quality can rule this explanation in or
   out rather than leaving it as an untested alternative.
4. **Occlusion handling.** Gaps are interpolated *and masked*: frames inside a
   gap longer than a tolerance are flagged `occluded`. Silently interpolating
   manufactures players who are not known to be where the model places them —
   the failure mode RQ4 exists to detect.
5. **Smoothing and derivatives.** Savitzky–Golay, with the first derivative taken
   analytically from the fitted local polynomial rather than by differencing a
   smoothed track (which would compound two filters and hide the effective
   bandwidth). Savitzky–Golay rather than a moving average because a boxcar
   filter flattens turns, biasing speed downward exactly where control models
   are most discriminative.
6. **Implausible-speed capping.** Speeds above 12 m/s are tracking artefacts, not
   fast players. Capped frames are **counted and reported** as a quality
   indicator, not silently fixed.
7. **Ball-trajectory handling.** Flight time is measured from the ball track
   where possible; otherwise imputed from a constant-speed model and flagged.
   The imputed fraction is reported per source, and a subgroup analysis compares
   measured against imputed arrivals.
8. **Duplicate events.** Removed by `arrival_id`, with the count recorded.
9. **Arrival detection and typology.** Each arrival is classified as
   `open_pass`, `cross`, `set_piece`, `deflection`, `clearance` or `second_ball`,
   and flagged endogenous or exogenous. **The provider-label concordance is a
   first-class artefact**: getting it wrong contaminates the quasi-exogenous
   subsample, which is the study's strongest identification argument, so it is
   documented per provider and spot-checked against video where possible.
10. **Inclusion criteria.** Passes shorter than 3 m (degenerate), longer than
    80 m or with flights beyond 5 s (outside the modelled regime), and censored
    outcomes are excluded. Each exclusion is recorded with its count and reason.
11. **Set pieces.** Separated, not deleted. Walls and zonal blocks violate every
    physics assumption; they are analysed as their own stratum.
12. **Low completeness.** Flagged, **not** dropped, and retained as a stratum,
    because their existence is the subject of RQ4.
13. **Substitutions and dismissals.** Players entering and leaving change the
    team sizes; `n_players_a/b` is carried per arrival, and red-card periods are
    flagged, because an 11-v-10 control problem is a different problem.

### 11.2 Reporting preprocessing uncertainty

Two mechanisms, both implemented:

- **A provenance table** written with every dataset build: every step, its
  parameters, how many rows it removed and why. Retention rate is reported in
  the paper's data section.
- **A preprocessing sensitivity sweep** (`preprocessing_uncertainty`): the
  headline calibration statistic is recomputed across a grid of smoothing
  bandwidths, occlusion tolerances and completeness thresholds. The **range**
  across that grid is reported alongside the bootstrap interval.

The decision rule is stated in advance: if the preprocessing range is comparable
to the between-model differences, then the model comparison is not identified by
the data, and the paper reports that rather than selecting the configuration
that separates the models.

---

## 12. Evaluation design

### 12.1 Why random row-level splitting is invalid

See Section 9.2. In one sentence: arrivals are nested within possessions within
matches within competitions, and a random row split lets a model be rewarded for
memorising a specific match's shape.

### 12.2 The split hierarchy

| Level | Split | Answers | Requirement |
|---|---|---|---|
| 1 | Match-level grouped hold-out, stratified by competition and source | RQ1, RQ2, RQ5 | any multi-match corpus |
| 2 | Match-level grouped *k*-fold | stability of the level-1 answer | as above |
| 3 | Leave-one-competition-out | RQ3 transportability, H4a | ≥ 2 competitions |
| 4 | Leave-one-source-out (optical ↔ broadcast) | RQ4, H4b | both modalities |
| 5 | Temporal forward-chaining | drift across seasons | ≥ 2 seasons |
| 6 | Cross-team (hold out a team's matches) | team-identity leakage | enough matches per team |

Every split carries a nested validation fold drawn from the training groups.
Nothing is selected on the test fold — not a hyper-parameter, not a bin count,
not a recalibration map.

### 12.3 Metrics

**Primary.**

| Metric | What it answers | Why primary |
|---|---|---|
| **Brier score** | overall forecast quality | strictly proper; finite even for hard 0/1 forecasts |
| **CORP MCB** (cross-fitted by match) | miscalibration, in Brier units | binning-free, reproducible, exactly consistent with the reported score |
| **CORP DSC** | sharpness/resolution | prevents "calibrated by flattening" |
| **Calibration slope `b`** | over/underconfidence | directly interpretable; the quantity that propagates downstream |
| **Calibration-in-the-large `a₀`** | systematic over/under-forecasting | separates level from scale |

**Secondary.** Log loss (with the clipping `ε` stated, since M1 makes it
necessary); spherical score, as a second proper rule — a conclusion that flips
between the Brier and spherical scores is a conclusion about the loss function,
not the model; binned ECE, MCE and ACE, always with bin counts, for
comparability with a literature that reports them; Murphy's binned decomposition
with its residual term reported so the identity can be audited; AUC and average
precision.

**Decision-analytic.** Net benefit across thresholds; standardised net benefit
for cross-subgroup comparison (base rates differ sharply between thirds);
expected cost under explicit cost ratios; threshold displacement.

### 12.4 Why discrimination alone is insufficient

This is the study's core methodological message, and it deserves a stated
argument rather than an assertion.

AUC is a function of the *ranking* only. For any strictly increasing
`g : [0,1] → [0,1]`, the forecasts `p` and `g(p)` have identical AUC and can have
arbitrarily different calibration. Concretely, take a perfectly calibrated
forecast and pass it through `g(p) = σ(2 · logit p)`: AUC is unchanged to machine
precision, while the calibration slope falls to 0.5 and every value is
materially overstated. Our test suite asserts exactly this.

For pitch control the consequence is direct. A model that correctly ranks
contested situations — exactly what makes a heatmap look right — can be
systematically overconfident at every level, and no ranking metric will ever
reveal it. Since downstream metrics multiply and integrate the *values*, not the
ranks, the property that matters for every published application of pitch
control is the one that discrimination metrics cannot see.

### 12.5 Uncertainty quantification

- **Cluster bootstrap resampling whole matches**, 1000 replicates, percentile
  intervals. Matches, not possessions: possession-level resampling would still
  treat within-match correlation as independent noise.
- **Paired** bootstrap for model comparisons on shared test arrivals.
- **Design effect reported.** `DEFF = 1 + (m̄ − 1)·ICC` states how much too small
  naive independent-observation standard errors would have been. On our
  simulated corpus this factor is around 8, implying naive intervals roughly
  three times too narrow. Much published football analytics does not cluster;
  reporting the factor makes the correction auditable and the stakes visible.
- **Percentile rather than BCa.** BCa's acceleration needs a cluster jackknife
  which, with 60–100 matches, is itself noisy; the percentile interval's coverage
  error is second-order and dominated by the small number of clusters. With
  fewer than about 30 matches the interval is reported as indicative and
  explicitly described as such.
- **Effective sample size** reported with every weighted statistic.

---

## 13. Calibration methodology

### 13.1 Reliability curves

Two constructions, both implemented in `pcc.calibration.reliability`:

- **Binned**, with equal-width and equal-frequency options. Equal-frequency
  (adaptive) binning is preferred for pitch-control outputs because the forecast
  distribution is strongly concentrated — most arrivals occur in space the
  passing team already dominates — so equal-width bins leave most bins nearly
  empty and the resulting curve is dominated by noise in a handful of them.
- **CORP (isotonic)**, requiring no bin choice. This is the headline figure.

**Every reliability diagram in the paper shows the forecast histogram beneath
it.** A calibration curve without the distribution invites the reader to weigh a
badly-calibrated region holding 1% of the data equally against a well-calibrated
region holding 60%. This is enforced by the plotting function rather than left
to discipline.

### 13.2 Primary metric and its justification

**The primary calibration statistic is the cross-fitted CORP MCB term.**

Justification, stated as a comparison rather than an assertion:

| Candidate | Why not primary |
|---|---|
| Binned ECE | Depends on an arbitrary bin count; biased, with bias depending on bins and `n`; not a proper scoring rule, so it can be lowered without improving the forecast; **can be driven near zero by coarsening** when the reliability curve crosses the diagonal. Our test suite constructs exactly that case: a forecast with `ECE₂ < 0.02` and `ECE₄₀ > 0.08`. |
| MCE | An extreme-value statistic; without a minimum bin count it estimates sampling noise. |
| Calibration slope alone | Summarises a monotone logit-linear distortion only; blind to non-monotone or tail-specific miscalibration. Reported, but as a *complement*. |
| Hosmer–Lemeshow-type tests | Binning-dependent, and with `n ≈ 10⁴`–`10⁵` they reject essentially always, so the *p*-value carries no information about magnitude. |
| **CORP MCB** | Binning-free; consistent with the reported Brier score by construction; in Brier units, so "how much does miscalibration cost?" is answerable in the units the field already reports; decomposes alongside DSC so calibration and sharpness are read together. |

**An honest caveat about MCB.** Estimated in-sample it is biased *upward*, not
downward: the PAV fit is the best monotone fit on that sample, so it absorbs
sampling noise and `S(x̂)` is too low. A perfectly calibrated forecast shows a
small positive in-sample MCB purely from this, and the bias grows as the sample
shrinks. We therefore use the match-cross-fitted estimator throughout for
headline claims, report that it may be slightly negative, and never truncate a
negative estimate at zero — truncation would reintroduce upward bias precisely
where the null is true. Both properties are asserted in the test suite.

### 13.3 The recalibration family

| Map | Parameters | What it can fix | What it cannot |
|---|---|---|---|
| Platt | 2 | monotone logit-linear distortion | tail asymmetry; non-monotone structure |
| Beta | 3 | asymmetric tail distortion | non-monotone structure |
| Isotonic | non-parametric | any monotone distortion | non-monotone structure; extrapolates poorly beyond observed forecast values; unstable in the tails where control forecasts pile up |
| Partially pooled Platt | 2 per stratum + `κ` | stratum-varying distortion with borrowed strength | as Platt, within stratum |
| Hierarchical Bayesian *(optional)* | random slope and intercept by match | as above, with full posterior uncertainty | as Platt, within cluster |

Two quantities are reportable results rather than intermediate steps:

- **Isotonic − Platt gap.** How much of the miscalibration is *not* logit-linear.
  A small gap means a two-parameter fix suffices, which is a clean and
  deployable practitioner recommendation.
- **Residual MCB after each map.** What recalibration cannot fix. A large residual
  means the model's *ordering* of situations is wrong, which is far more serious
  than a scale error and cannot be patched downstream at all.

### 13.4 Stratified calibration

Calibration is reported for each pre-registered stratum: pitch zone (a 5×3
lattice — the coarsest partition that still separates own box, build-up,
midfield, half-space and final third), pass-length band, flight-time band,
pressure band, game state, arrival type, tracking source and quality flag.

Three disciplines:

- **Pre-registration.** Bands are fixed on football grounds before analysis. A
  subgroup analysis whose cut-points were chosen after seeing the outcome is not
  a test.
- **Suppression, not deletion.** Strata below a minimum count are reported *with
  their counts* and marked suppressed. For the spatial map this matters most: the
  sparsely targeted cells (the defensive corner, the opposition six-yard box) are
  exactly where a physics model is least constrained by data and most likely to
  be wrong, and a figure that interpolates over them manufactures confidence.
- **Multiplicity.** Benjamini–Hochberg FDR control across the whole
  pre-registered family; raw and adjusted values both reported.

### 13.5 Confidence intervals

- **Cluster bootstrap by match** for every calibration statistic, including
  per-stratum statistics (with the number of *clusters* in each stratum reported
  alongside `n`, since a stratum with 5,000 arrivals from 3 matches supports very
  little inference).
- **Cluster-robust standard errors** as a cross-check for the slope and
  intercept, via a GLM with match-clustered variance.
- **Bayesian posterior intervals** where the hierarchical model of Section 8.11
  is fitted; these are preferred for the between-match heterogeneity parameter,
  which the bootstrap estimates poorly with few clusters.
- **Bootstrap bands on reliability curves**, computed with the bin edges held
  fixed at their full-sample values so that replicate variation is in the curve
  rather than in the binning.

---

## 14. Selection bias and propensity weighting

### 14.1 The problem, formally

Let `D = 1` indicate that an arrival at state-and-destination `X` was observed.
The data identify

```
P( Y = 1 | X , D = 1 )
```

while the pitch-control field claims to describe

```
P( Y = 1 | X ) .
```

These coincide if and only if `Y ⫫ D | X`. Two distinct obstacles separate them,
and conflating them is the standard error in this area.

**(a) Covariate shift.** Even under `Y ⫫ D | X`, the *distribution* of `X` among
observed arrivals is far from uniform over the pitch: players pass into space
they believe they control. Conditional calibration at the level of `X` transfers
under this shift, but **marginal, binned calibration does not**, because a
reliability bin aggregates over `X` and its composition changes with the
evaluation distribution. This is the part reweighting genuinely fixes.

**(b) Selection on unobservables.** The passer knows things the state vector does
not: that the receiver has already started the run, that the defender has
mis-stepped, that the pass will be disguised. Conditional on `X`, observed
arrivals are then systematically the *easy* ones and `Y ⫫̸ D | X`. **No
reweighting scheme fixes this**, because the required information is absent by
construction. Presenting IPW as though it did would be the central methodological
error available to this study, and we will not make it.

### 14.2 The response, in descending order of strength

#### 1. Quasi-exogenous arrivals — the primary identification argument

Deflections, clearances, blocked passes and aerial second balls arrive where
nobody aimed. Their destination is not chosen conditional on private information
about control, so on this subsample the `U → D` arrow of Section 7.4 is largely
severed. **Calibration measured here is the study's most credible estimate**, and
the contrast against chosen destinations is a direct, if partial, measurement of
the selection effect itself.

The contrast is not clean in one respect, and the paper must say so: unchosen
arrivals also differ in *football* — faster, more often aerial, more often
contested. The contrast therefore confounds selection with arrival physics. The
remedy is to match on observables (flight time, height where available, pitch
zone, local player density) before comparing, and to report post-matching
covariate balance; an unbalanced match is not a match.

A practical constraint that must be checked early, and which checking has
already shown to bite: **the exogenous subsample can be far smaller than the
arrival count suggests.** On Metrica's two matches, the event taxonomy yields
1,760 chosen arrivals but only **11** unchosen ones — too few to estimate
anything, so the subsample is suppressed entirely by the analysis. The
limitation is not the corpus's size but its *labelling*: Metrica records
clearances and deflections sparsely and folds most loose-ball contests into a
`CHALLENGE` type that carries no flight coordinates.

Two consequences follow, and they change the design rather than decorate it.
First, **a corpus must be selected partly on its arrival taxonomy**, not only on
its size and tracking quality: a provider that distinguishes deflections,
clearances and aerial second balls with coordinates is worth more to this study
than one with twice the matches and a coarser event schema. Second, the
verification checklist for any candidate corpus must include *counting the
exogenous arrivals*, and that count must be reported alongside the total. A
paper that leans on the quasi-exogenous contrast while drawing it from a few
dozen arrivals would be resting its identification argument on noise.

Relatedly, the same run puts the selection strength in the corpus at an AUC of
0.998 for separating chosen from proposed destinations, with the effective
sample size after reweighting falling from 1,771 to 253. Both numbers are from
two matches and are reported here only as evidence that the diagnostics
function; but if selection really is that strong in elite football, the
reweighted arm of the analysis will be badly underpowered relative to its
nominal `n`, and the design should lean correspondingly harder on the
quasi-exogenous contrast and the stratified reporting.

#### 2. Density-ratio reweighting — for (a) only

For each observed arrival, propose counterfactual destinations from a reference
distribution `q` (recommended: uniform over the physically reachable disc about
the release point, intersected with the pitch — the uniform-over-pitch proposal
generates many destinations no player would attempt, which makes the density
ratio extreme and the weights unstable). Fit a classifier separating chosen from
proposed destinations. Under this case–control design the classifier's odds are
proportional to `p_obs(X)/q(X)`, so

```
w(X)  ∝  q(X) / p_obs(X)  ∝  1 / odds(X) ,
```

normalised to mean one. The constant of proportionality from case–control
sampling cancels in the normalisation, so the classifier need not — and does
not — estimate an absolute propensity.

This answers a precisely stated question: *how calibrated would this model be if
the ball arrived according to `q` rather than according to player choice?* It
does **not** supply outcomes at unchosen destinations, because none exist.

Reported with it, always: the selection AUC (how strongly passers select — an
AUC near 0.5 would make the whole concern moot and is itself worth knowing); the
effective sample size and its ratio to nominal `n`; the weight share held by the
top 1% of observations; the trim point and the weight mass it removed; and the
fraction of observed arrivals outside the candidates' support, which marks where
the reweighted estimate is extrapolation rather than data.

#### 3. Stratification

Quintiles of the selection score, each reported with its own `n` and calibration
statistics. This is the transparent alternative to one reweighted number whose
construction the reader must take on trust. **A disagreement between the weighted
and stratified answers is treated as evidence that the weights are fragile**, not
as a reason to prefer whichever is more convenient.

#### 4. Matched comparisons

1:1 nearest-neighbour propensity matching with a caliper, used for the
chosen/unchosen contrast, with standardised mean differences reported before and
after.

#### 5. Synthetic ball arrivals — explicitly not an identification strategy

Candidate destinations are used *only* to estimate the density ratio. They carry
no outcome. Any framework that generates candidate arrivals and then imputes
their outcomes from a model is evaluating the model against itself. We state
this because the temptation is real and the resulting circularity is easy to
miss.

#### 6. Negative controls

Beyond the exogenous subsample, two further checks: (i) arrivals where the ball
tracking shows the destination differs materially from the provider-annotated
intended destination — a mis-hit pass, where the destination is closer to
exogenous; (ii) goal kicks and long clearances into contested areas, where the
"choice" is coarse.

#### 7. Sensitivity analysis

Model unmeasured selection as a multiplicative bias on the outcome odds among
observed arrivals: for `Γ ≥ 1`,

```
p^±(x) = Γ^{±1} p(x) / ( Γ^{±1} p(x) + 1 − p(x) ) .
```

Recomputing the calibration statistics under the deflated bound answers: *if
observed arrivals are systematically easier than the state alone implies, by an
odds factor of `Γ`, would the model still look miscalibrated?* The reported
quantity is the smallest `Γ` at which the finding would be explained away, with
a judgement — informed by the chosen/unchosen contrast, which gives an empirical
anchor for plausible `Γ` — as to whether selection that strong is credible.

This is a bounding argument in the spirit of Rosenbaum sensitivity analysis. It
says how bad the unmeasured selection would have to be. **It cannot say how bad
it is**, and the paper will not imply otherwise.

### 14.3 What remains unidentified

Stated plainly, because it is the study's principal limitation and a reviewer
will find it regardless:

> **Observational tracking data cannot identify `P(Y | X)` at destinations where
> the ball never arrives.** No weighting, matching or sensitivity analysis
> changes this. The study therefore evaluates pitch-control models *at realised
> ball arrivals*, and its conclusions extend to the full field only under an
> extrapolation assumption that it states rather than conceals.

This is not a fatal limitation; it is the correct scope. The realised-arrival
population is exactly the population on which every downstream application
operates — possession value is computed over passes that happen, pass
recommendations concern destinations a player is considering. A model
miscalibrated on realised arrivals is miscalibrated where it is used.

---

## 15. Main empirical analysis

The analysis order is fixed in code (`pcc.evaluation.protocol.run_protocol`) so
that the same sequence is applied to every model, split and subgroup. Freezing
it prevents the analyst degrees of freedom to which an evaluation paper is most
vulnerable: choosing a bin count after seeing the curve, or a subgroup after
seeing which one is significant.

### 15.1 Descriptive foundation

Arrivals by type and zone; the base rate of control overall and by stratum; the
forecast distribution of each model; the intra-match ICC and the design effect;
retention through each preprocessing filter; frame completeness by source. The
design effect is reported first, because it determines how seriously every
subsequent interval should be taken.

### 15.2 Overall calibration (RQ1, H1)

For each model on held-out matches: Brier, log loss, spherical; cross-fitted
CORP MCB/DSC/UNC; calibration slope and intercept; AUC; forecast dispersion.
Each with a cluster-bootstrap interval. Headline figure: CORP reliability curves
for all models with forecast histograms.

### 15.3 Model ranking (RQ2, H2)

Paired cluster-bootstrap differences against the logistic baseline, separately
for Brier, MCB and DSC. Reported as a table with an explicit `favours` column
taking the values `model` / `reference` / `inconclusive`, so that an
inconclusive comparison is reported as inconclusive rather than as a ranking.

### 15.4 Subgroup analysis (RQ3, H3)

Stratum-level MCB and slope over the pre-registered family, with FDR control and
suppression of thin strata.

### 15.5 Error decomposition

CORP MCB/DSC/UNC per model, with Murphy's binned decomposition and its residual
term reported alongside for comparability. The key display pairs MCB against DSC
so that "calibrated by flattening" is immediately visible.

### 15.6 Spatial calibration maps

Cell-level `E[Y | cell] − E[C | cell]` on a 12×8 lattice, with insufficient cells
left blank. **The blankness is a finding**, not a gap: it marks where the ball
essentially never arrives and where a control model's claims are empirically
untestable on observational data.

### 15.7 Temporal analysis

Calibration by match minute and by period, to test whether fatigue-driven
departures from the assumed locomotion envelope degrade the physics models late
in matches — a directional prediction the envelope makes and that has an obvious
practical consequence for substitution-window analysis.

### 15.8 Team-style analysis

Per-team calibration with match-clustered intervals, and the between-team
variance from the hierarchical model. The question is whether "the model is
overconfident" is a property of the model or of particular tactical setups; a
high-line, high-press team presents a systematically different control problem
and may be where a fixed locomotion envelope fails hardest.

### 15.9 Pass-type analysis

Calibration by arrival type, with the chosen/unchosen contrast of Section 14 as
the centrepiece; and by pass height where ball `z` is available, since aerial
arrivals are where the reachability construct and the possession construct
diverge most.

### 15.10 Robustness and sensitivity

- **Control horizon.** The full analysis at `h ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}`
  and under all four outcome definitions. Reported as a figure of MCB and slope
  against `h`. **A conclusion that flips across this range is reported as a
  conclusion that flips.**
- **Tracking noise.** Gaussian position noise and simulated occlusion injected
  into optical data at several severities, re-running the whole analysis. This
  is what separates measurement degradation from sampling in RQ4.
- **Player parameters.** The locomotion sweep of Section 17. Our simulated
  ablation shows `σ` alone moving MCB by more than the between-model
  differences; if that holds on real data it is a headline result in its own
  right and reframes the model comparison as a statement about assumed
  parameters.
- **Preprocessing.** The sweep of Section 11.2.
- **Second proper score.** Every ranking re-checked under the spherical score.

---

## 16. Downstream analysis

A calibration paper that stops at the Brier score has not shown that anything
matters. This section propagates raw and recalibrated control values into the
metrics pitch control is used to build.

### 16.1 The propagation argument

Most pitch-control-derived quantities are approximately **linear in `C`**:

```
EPV( pass to x )  ≈  C_A(x) · V(x)  −  ( 1 − C_A(x) ) · δ · V_B(x)
```

space metrics are integrals `∫ C_A dA`; off-ball run value is a difference of two
fields `∫ ( C'_A − C_A ) V dA`.

Linearity gives the sharp consequence stated in Section 3.4: **a calibration
error with a consistent sign over a region accumulates when integrated over that
region rather than averaging out.** A purely random error would wash out; the
systematic component measured by the calibration slope and intercept does not.

It also gives a testable structural prediction that distinguishes the metrics:

> EPV, being a *level*, is sensitive to calibration-in-the-large. Off-ball run
> value, being a *difference* of two fields, is insensitive to a constant offset
> but sensitive to slope error, since a slope error scales the two fields
> differently where they differ. We predict, and will test, that off-ball metrics
> are less affected by `a₀` and comparably or more affected by `b`.

### 16.2 Metrics propagated

| Metric | Construction | Predicted sensitivity |
|---|---|---|
| Expected possession value | control-weighted positional value | high to both `a₀` and `b` |
| Pass completion / difficulty | `C_A` at the destination, directly | highest — it *is* the forecast |
| Off-ball run value | `∫ (C'_A − C_A) V dA` | low to `a₀`, high to `b` |
| Space created | `∫ C_A dA` over a region | high to `a₀` |
| "Hard" vs "soft" space owned | `area(C_A > 0.5)` vs `∫ C_A dA` | the *gap* between them is a pure calibration artefact |
| Team compactness | dispersion of the control field | moderate |
| Defensive pressure | `1 − C_A` near the ball | moderate |
| Tactical similarity | distance between control fields across matches | moderate; a monotone distortion may leave rankings intact |

The hard-versus-soft space comparison deserves emphasis as a presentational
device: practitioners routinely compute thresholded "space owned", and the gap
between it and the probability-weighted integral is a clean, immediately legible
demonstration that the calibration question has practical teeth.

### 16.3 Comparisons made

For each metric: raw control, each recalibrated variant, and the fitted logistic
baseline. Reported as mean, total over the corpus, mean absolute per-pass
difference from the recalibrated reference, and the correlation between variants
— the last matters because a metric that shifts in level but preserves ordering
is still usable for ranking players, which is how many of these metrics are
actually consumed.

### 16.4 The value surface

**No positional value grid is hard-coded anywhere in this repository**, and the
class that would supply a fabricated one refuses to construct without an
explicit acknowledgement flag. Published expected-threat style grids belong to
their authors and their values depend on the league and season they were fitted
on. The surface is either fitted from the corpus at hand by empirical-Bayes cell
means with shrinkage, or supplied explicitly, and its provenance is carried
through into every output table. Sensitivity to the loss-discount parameter `δ`
is reported across a range rather than at a single analyst-chosen value.

### 16.5 Statistical versus football improvement

The distinction the paper must not blur:

- **Statistical improvement** — a better proper score. Necessary, not sufficient.
- **Decision improvement** — better decisions at a realistic operating point.

The instruments for the second:

1. **Decision-curve analysis.** Net benefit across thresholds, against the
   treat-all and treat-none defaults. The threshold encodes the decision-maker's
   exchange rate between a foregone progression and a turnover, so the whole
   curve is reported rather than one operating point. A coach treating a midfield
   turnover as roughly twice as costly as a foregone progression operates near
   `p_t = 2/3`.
2. **Threshold displacement.** The fraction of arrivals on which raw and
   recalibrated forecasts fall on opposite sides of a threshold. This is the
   translation of "the number was miscalibrated" into "a coach would have decided
   differently", and it is the number the paper should lead with in its practical
   implications.
3. **Explicit cost-sensitive utility**, costing a turnover by the opponent's
   resulting positional value.

**The pre-registered interpretive rule.** It is entirely possible to find that
recalibration moves the numbers substantially *and* does not improve net benefit
at realistic thresholds — because a monotone recalibration preserves the ranking
and therefore changes only which threshold corresponds to which decision. If
that is what we find, the honest conclusion is: *miscalibration matters for
quantities that are summed or multiplied (valuation, space accounting) and
matters less for decisions that are ranked (which pass is best), and
practitioners should re-tune their thresholds rather than expect recalibration
to improve their rankings.* That is a useful, falsifiable, and appropriately
modest conclusion, and we commit to it now so that it cannot later be
presented as a disappointment.

---

## 17. Robustness and ablation studies

Implemented in `scripts/07_ablations.py`. The purpose is adversarial: to find
out whether the model comparison survives the assumptions it rests on.

| Axis | Values swept | What it tests |
|---|---|---|
| Time-to-point model | constant speed; bounded acceleration | Whether the published models' optimistic short-range `τ` matters |
| Sprint speed `v_max` | 4, 5, 6, 7 m/s | Sensitivity to the locomotion envelope |
| Reaction time `t_r` | 0, 0.3, 0.7, 1.0 s | Whether the reaction delay is load-bearing |
| Arrival uncertainty `σ` | 0.2, 0.45, 0.8, 1.2 s | **The key nuisance parameter** — it governs the field's sharpness directly |
| Control rate `λ` | 2, 4.3, 8 s⁻¹ | Sensitivity of the dynamics |
| Acceleration bound `a_max` | 4, 7, 12 m/s² | Whether the bound binds |
| Integration step `dt` | 0.02–0.25 s | Numerical, not substantive; must be shown not to matter |
| Renormalisation | on / off | Whether the field sums to one across teams |
| Velocity | present / zeroed | The freeze-frame regime; how much velocity buys |
| Acceleration features | present / absent | Whether the second derivative is usable at all |
| Nearest-distance only | on / off | How much of the signal is one number |
| Control horizon `h` | 0.25–3.0 s | Construct sensitivity |
| High-speed passes | included / excluded | Whether long, fast arrivals drive the result |
| Set pieces | excluded / analysed separately | Whether the exclusion matters |
| Low-completeness arrivals | included / excluded / separate stratum | RQ4 |
| Pitch dimensions | 100×64, 105×68, 110×72 | Metric-normalisation error |
| Smoothing bandwidth | 0.2, 0.4, 0.8 s | Preprocessing sensitivity |
| Occlusion tolerance | 0.25, 0.5, 1.0 s | Preprocessing sensitivity |
| Competition | train/test across competitions | Transportability |
| Tracking source | optical ↔ broadcast, plus simulated degradation | RQ4, with the confound addressed |

**The reporting rule, fixed in advance.** For each axis we report the *range* of
the headline statistic across the sweep, and compare it against the between-model
differences from the main analysis. If the largest within-axis range is of the
same order as the between-model differences, the model ranking is an artefact of
the assumed locomotion envelope rather than a finding, and the paper reports it
that way. On simulated data this comparison is already informative: sweeping `σ`
alone moves MCB and the calibration slope by considerably more than the models
differ from one another.

---

## 18. Threats to validity

### 18.1 Construct validity

*The central threat.* "Control" is not defined in the source literature, so the
study must impose a definition (Section 7.3) and thereby risks scoring models
against a target they never claimed. A defender of the physics models can
reasonably say they answer "who reaches it first", not "who ends up with it".

*Mitigations.* The operational definition is stated, defended and varied; all
four definitions are reported; the first-touch definition is included precisely
as the closest observable proxy for the reachability construct. The argument the
paper makes is not that the models fail at their own task, but that the
*downstream literature* uses them for a different task, and that this is where
the error enters. That argument survives the objection.

### 18.2 Internal validity

- **Event–tracking synchronisation error** masquerades as miscalibration.
  *Mitigation:* per-match offset estimation; residual retained as a covariate; a
  subgroup analysis by synchronisation quality; a sensitivity sweep on the
  applied offset.
- **Derived velocity** is a filter output, not a measurement. *Mitigation:* the
  bandwidth is an ablation axis, and a velocity-free variant is always reported.
- **Label noise in `Y`.** Possession definitions vary by provider; contested
  arrivals are genuinely ambiguous. *Mitigation:* four definitions; censoring
  rather than coding stoppages as failures; where feasible, a hand-labelled
  sample to estimate label-noise rates, which also gives a floor on achievable
  calibration.
- **Leakage.** *Mitigation:* group splits, assertions, a committed audit table.
- **Analyst degrees of freedom.** *Mitigation:* the protocol is code; the config
  is hashed into every manifest; subgroups and horizons are pre-registered.

### 18.3 External validity

- **Elite men's football only.** Available tracking corpora are overwhelmingly
  elite men's competition. Locomotion envelopes, pressing intensity and pass
  distributions differ in women's football, in youth football, and at lower
  levels — a physics model calibrated with a 5 m/s sprint assumption has no
  reason to transfer. *Mitigation:* the claim is scoped explicitly to the
  competitions analysed; transfer is listed as a dissertation extension rather
  than asserted.
- **Tournament versus league football.** A World Cup is not a league season:
  unfamiliar opponents, compressed schedules, different risk profiles.
- **Single-provider generalisation.** Conclusions from one provider's tracking
  may not transfer. *Mitigation:* multiple sources where available; the
  degradation ablation where not.

### 18.4 Statistical threats

- **Dependence.** Arrivals cluster within possessions and matches. *Mitigation:*
  match-level cluster bootstrap throughout; design effect reported.
- **Class imbalance.** Base control rates are high (roughly 0.75–0.85 for open
  passes), so accuracy is uninformative and the uncertainty term dominates the
  Brier score. *Mitigation:* proper scores and their decompositions; UNC reported
  with every stratum; precision–recall metrics alongside AUC.
- **Multiplicity.** Many strata × many models. *Mitigation:* FDR control over the
  pre-registered family; unadjusted values also shown.
- **Few clusters.** With under ~30 matches the bootstrap intervals are
  indicative only. *Mitigation:* stated as such; the power analysis
  (`scripts/08_power_analysis.py`) sizes the corpus in advance, with between-match
  slope heterogeneity — not the arrival count — as the parameter that governs
  power.
- **Model misspecification in the baselines.** A poorly tuned logistic baseline
  makes the physics model look better. *Mitigation:* the baseline is tuned by
  grouped CV on a proper loss and its coefficients are reported; the GBM provides
  an upper bound on what the feature set supports.

### 18.5 Data-availability threats

- **The primary corpus may be inaccessible.** *Mitigation:* the fallback design
  is pre-registered, not improvised, and the availability check is a committed
  artefact.
- **Licensing may restrict use or redistribution.** *Mitigation:* no provider
  data is committed; only derived, aggregated results and code are released.
- **Small open corpora are too small.** *Mitigation:* stated honestly; wide
  intervals reported as wide.

### 18.6 Simulation-to-reality gap

The simulator validates the instrument and sizes the design. Its football is
crude: Gaussian formations, no offside, no possession structure, no fatigue.
Real arrivals are more strongly clustered within matches than the simulator's
are, so simulated power estimates are optimistic and are reported as **lower
bounds** on the corpus required. No simulated number appears as an empirical
claim, and every simulated artefact is tagged `tracking_source = "simulated"` so
it cannot be mixed with real data by accident.

---

## 19. Expected contributions

Stated at the level the evidence would actually support.

**Theoretical.** A precise articulation of the measurement-validity problem in
football spatial analytics: the distinction between reachability, possession and
retention as three different estimands that the literature collapses; and the
linearity argument showing that systematic control error accumulates under
aggregation rather than averaging out. Neither is deep mathematics; both are
conceptual clarifications that appear not to have been stated.

**Statistical.** An evaluation protocol for counterfactual spatial fields that
are only ever observed at selected points: realised-arrival evaluation, the
quasi-exogenous subsample as an identification device, density-ratio reweighting
for the covariate-shift component, and an explicit statement of what remains
unidentified. Transfer of the CORP framework into sports analytics, with a
documented demonstration that binned ECE can be driven near zero by coarsening on
exactly the kind of reliability curve these models produce.

**Machine learning.** A controlled comparison of unfitted physics-based models
against fitted statistical baselines *on identical inputs*, separating
calibration from discrimination. A characterisation of which recalibration family
suffices, and whether a fitted map transfers across competitions and tracking
modalities.

**Football analytics.** The first reliability diagrams for pitch-control models
that we are aware of; an empirical answer to whether pitch-control values may be
used as probabilities; quantification of how much downstream metrics move under
recalibration; and — potentially the most practically useful output —
quantification of how much the assumed locomotion parameters, rather than the
model structure, drive the answer.

**Practical.** Concrete guidance: whether to recalibrate before using control
values in valuation; whether a global two-parameter map suffices or clubs must
refit per competition; where on the pitch and under what conditions the values
should not be trusted; and how much a club using broadcast tracking should
discount its control-derived metrics.

**Reproducibility.** A tested, runnable implementation of every model, metric and
protocol step, which executes end-to-end without any provider data, together with
a validated calibration estimator, a committed leakage audit, and a manifest for
every result recording the git revision, package versions and a hash of the
configuration.

### 19.1 What counts as a meaningful result if the hypotheses fail

This matters because the study must be worth doing under either outcome.

- **If the physics models turn out well calibrated**, that is a strong positive
  result: it would validate a widely used tool against a test it has not
  previously faced, and would justify the downstream literature's usage. It would
  also be genuinely surprising, given that no parameter is fitted, and the
  natural follow-up — *why* does an unfitted physical model land on calibrated
  probabilities? — is a better question than the one we started with.
- **If the logistic baseline matches the physics model**, the conclusion is that
  the physics buys interpretability and cross-league transfer rather than
  accuracy — useful, and a reason to prefer it in data-poor settings.
- **If recalibration fixes everything**, the recommendation is a one-line fix
  practitioners can adopt immediately, and the paper's value is precisely that it
  is a deflating result. Good papers are often deflating.
- **If the answer is dominated by the assumed `σ`**, the study becomes a paper
  about parameter sensitivity in physics-based sports models, which is arguably
  more important than the calibration question itself.
- **If nothing is statistically distinguishable**, the contribution is the
  protocol, the demonstration that available open corpora are too small for this
  question, and a power analysis telling the field what corpus would be needed.
  That is a legitimate, publishable negative result at a workshop.

---

## 20. Possible results and their interpretation

**No results have been obtained.** The table below states, in advance, how each
pattern would be read — so that the interpretation is fixed before the data
rather than fitted to it.

| # | Pattern | Reading | What must also be checked before believing it |
|---|---|---|---|
| 1 | Physics model well calibrated (`MCB ≈ 0`, `b ≈ 1`) | Strong validation of current practice; downstream uses justified | Is DSC also high? A flat forecast is calibrated and useless. Does it hold out of competition? |
| 2 | Sharp but overconfident (`DSC` high, `b < 1`) | **The expected pattern.** Ranking is sound, scale is wrong; downstream valuations are biased in a signed, accumulating way; recalibration should fix most of it | Is the slope stable across strata? A varying slope means no single fix works |
| 3 | Logistic baseline beats the physics model on MCB with equal DSC | The physics adds no probabilistic information a simple regression lacks; use physics fields as *features*, not as probabilities | Is the baseline's advantage preserved out of competition, where it has more to lose? |
| 4 | Physics model beats the fitted models out of competition despite worse in-distribution calibration | Physics buys transportability; the right recommendation depends on deployment setting | Is the fitted model's failure a support problem (GBM extrapolation) rather than a real advantage for physics? |
| 5 | Calibration varies sharply by zone (e.g. overconfident in the final third) | Zone-specific recalibration is required; any aggregate space metric is biased non-uniformly | Are the zones confounded with pass length and pressure? Multivariable stratification needed |
| 6 | Calibration degrades on broadcast tracking | Clubs without optical tracking should discount control-derived metrics | Does the degradation ablation on optical data reproduce the size of the effect? If not, it is sampling, not measurement |
| 7 | Platt scaling recovers nearly all the lost score | Deflating and useful: a two-parameter fix, deployable today | Does the map transfer across competitions, or must it be refitted? |
| 8 | Isotonic ≫ Platt | The distortion is non-logit-linear, probably tail-specific; a simple fix is not enough | Is the isotonic advantage stable out of sample, or is it PAV overfitting? |
| 9 | Large residual MCB after every map | The model's *ordering* is wrong, not just its scale — the most serious possible finding, and unfixable downstream | Is it concentrated in an identifiable stratum? If so, selective prediction is the remedy |
| 10 | No model transfers across competitions | Pitch control is league-specific; published fields should not be applied to a new competition without local validation | Is the failure in calibration, discrimination, or both? |
| 11 | Chosen and unchosen arrivals show very different calibration | Selection on unobservables is material; the chosen-arrival estimate is optimistic and the unchosen estimate is the better guide | Does it survive matching on flight time, height and congestion? |
| 12 | Reweighted and stratified answers disagree | The weights are fragile; report the stratified result and the disagreement | Check overlap diagnostics and effective sample size |
| 13 | `σ`-sweep range ≥ between-model differences | The model comparison is not identified; the paper's finding is about parameter sensitivity | Does the ranking at least stay stable in *sign* across the sweep? |
| 14 | Downstream EPV shifts materially but net benefit is unchanged | Miscalibration matters for accounting, not for ranking decisions; practitioners should re-tune thresholds rather than expect better rankings | Is that true at *all* thresholds, or only away from the operating point? |
| 15 | Everything is inconclusive with wide intervals | The open corpora are too small; the contribution is the protocol and the power analysis | Is the design effect the binding constraint, or the number of matches? |

Patterns 2, 7 and 14 are, in our judgement, the most likely combination. Stating
that guess in advance is a discipline, not a prediction: if the data say
otherwise, the record shows we were wrong rather than that we always expected it.

---

## 21. Reproducibility plan

### 21.1 Software and environment

| Component | Choice | Justification |
|---|---|---|
| Language | Python 3.10+ | The football-analytics ecosystem (`kloppy`, `socceraction`, `mplsoccer`, `statsbombpy`) is Python-first |
| Arrays / tables | NumPy, pandas | standard |
| Numerics | SciPy | Savitzky–Golay, optimisation, distributions |
| Models and calibration | scikit-learn | `IsotonicRegression`, `LogisticRegression`, `HistGradientBoostingClassifier`, `GroupKFold` — all needed pieces without extra dependencies |
| Figures | Matplotlib | no plotting dependency the study cannot run |
| Config | PyYAML | one hashed config file per experiment |
| Tests | pytest | 120 tests, including instrument validation |

**Optional, with the check applied to each:**

- **`kloppy`** — *appropriate.* Provider-agnostic deserialisers and a common
  coordinate model. Writing bespoke parsers would duplicate maintained work and
  add a second source of coordinate-convention bugs. Recommended for all three
  provider adapters.
- **`statsbombpy`** — *appropriate* for the fallback design specifically.
- **`socceraction`** — *appropriate but narrow.* The reference VAEP
  implementation. Used only in the downstream section, and only if the corpus has
  the event coverage VAEP requires; otherwise the simpler linear EPV of
  Section 16.1 is used, which is sufficient for the propagation argument and does
  not pretend to be VAEP.
- **`mplsoccer`** — *appropriate but optional.* Nicer pitch rendering. The
  repository implements minimal pitch furniture directly so figures render
  without it.
- **PyTorch** — *appropriate only if Model 5 is included.* Omitted from the
  minimum viable study.
- **LightGBM / XGBoost** — *not necessary.* scikit-learn's histogram gradient
  boosting is adequate and removes a compiled dependency. Available as a swap.
- **statsmodels** — *appropriate* for cluster-robust GLM standard errors as a
  cross-check on the bootstrap.
- **Stan / `brms`** — *appropriate* for the hierarchical model of Section 8.11.
  Optional; it introduces a second language toolchain and is not needed for the
  minimum viable study.
- **MLflow / Weights & Biases** — *not recommended here.* This study runs a fixed
  protocol a small number of times, not a hyper-parameter search. A hashed config
  plus a per-result manifest gives the same auditability without a service
  dependency or an account. If the project grows to include Model 5 tuning, MLflow
  (local file backend) becomes worthwhile.

### 21.2 Determinism and provenance

- **Seeds.** A single `random_state` in the config propagates to NumPy, Python's
  `random`, scikit-learn and (if present) PyTorch via `seed_everything`.
- **Manifests.** Every results directory carries `manifest.json` with the script
  name, UTC timestamp, git revision, **whether the working tree was dirty**,
  Python and platform, package versions, the full config and its SHA-256. A
  result produced from an uncommitted tree is a fact the reader should know, so
  it is recorded rather than hidden.
- **Config hashing.** Two result sets can never be silently compared across
  different settings: differing configs produce differing hashes.
- **Tables as CSV.** Results are written as CSV, not pickles: they must be
  readable in ten years without this code.
- **Instrument validation first.** `scripts/09_validate_instrument.py` checks that
  the calibration estimator recovers a known slope, detects an injected
  distortion, shows AUC's blindness to it, and produces wider intervals under
  clustering. `run_all.sh` runs it before anything else.

### 21.3 Data versioning and licensing

- **No provider data is committed.** `data/raw/` is git-ignored. The repository
  ships loaders and documentation, not data.
- **Each source is recorded** with its access route, licence status (unverified
  until read) and a verification checklist in `docs/data_sources.md`.
- **A datasheet** is completed per corpus actually used
  (`docs/datasheet_template.md`), and a **model card** per fitted model
  (`docs/model_card_template.md`).
- **Ethics.** The data concerns professional athletes performing in public.
  Positional tracking of identifiable individuals is nonetheless personal data
  under several regimes; the study reports team-level and aggregate results, does
  not publish individual-level derived performance measures, and does not
  redistribute raw tracking. Where a provider's terms restrict use, the terms
  govern.

### 21.4 What can be released if the data cannot be

This matters, because the most likely corpus is one that cannot be redistributed:

1. **All code**, tested and runnable end-to-end on the simulator.
2. **All derived, aggregated results**: metric tables, decomposition tables,
   reliability-curve coordinates, subgroup tables, ablation grids, bootstrap
   intervals. None of these reconstruct the source data.
3. **The simulator**, so the pipeline is executable by anyone.
4. **The exact preprocessing provenance and leakage audit tables.**
5. **A schema-conformant synthetic replica** matched to the real corpus's
   marginal distributions, so a reader can run the full analysis end-to-end and
   verify the code path, clearly labelled as synthetic.
6. **Per-arrival forecasts and outcomes for the test fold**, *if and only if* the
   licence permits — this is what would let a third party recompute every
   calibration statistic independently, and it is worth asking the provider for
   explicitly.

---

## 22. Proposed paper structure

Target: 8–10 pages plus appendix for a workshop; 25–35 pages for a journal.

**1. Introduction.** Pitch control is used as a probability. The probabilistic
reading has not been tested. It is testable at realised ball arrivals.
Contributions listed as three or four bullets. *(1.5 pp.)*

**2. Related work.** Structured as Section 6 — control models; possession value;
calibration and proper scoring rules; selection. Ends with the gap stated in one
paragraph. *(2 pp.)*

**3. Problem formulation.** The construct (reachability vs possession vs
retention); the operational definition of `Y`; the estimand; the selection
problem with its DAG. This section carries the paper's conceptual contribution
and should not be compressed. *(2 pp.)*

**4. Data.** Sources with availability and licence status; preprocessing with
retention; the arrival typology; quality covariates. Honest about what was
verified and what was not. *(2 pp.)*

**5. Methods.** Models M0–M4 with their equations; the evaluation protocol;
metrics with the CORP choice justified; recalibration; selection handling. *(3 pp.)*

**6. Results.** Overall calibration and reliability curves (the headline figure);
model comparison separating MCB and DSC; subgroup and spatial calibration;
recalibration. *(4 pp.)*

**7. Robustness.** Horizon sensitivity; locomotion-parameter ablations;
preprocessing sweep; tracking-noise injection; the selection analyses. *(2 pp.)*

**8. Practical implications.** Downstream propagation; decision curves; threshold
displacement; concrete guidance for practitioners. *(2 pp.)*

**9. Limitations.** Construct validity; the unidentified counterfactual; corpus
size; generalisation beyond elite men's football. Written as a section, not a
paragraph — this is the study's honesty budget and reviewers will read it first.
*(1 p.)*

**10. Conclusion.** *(0.5 p.)*

**Appendices.** Full metric tables with intervals; per-stratum tables including
suppressed strata with counts; the complete ablation grid; the preprocessing
provenance and leakage audit; the instrument-validation report; model cards.

**Figures, in order of importance.**
1. CORP reliability curves for all models, with forecast histograms.
2. MCB against DSC, one point per model, with bootstrap ellipses — the single
   figure that makes the calibration/discrimination distinction visible.
3. Spatial calibration map with insufficient cells blank.
4. Forest plot of MCB by stratum.
5. Calibration slope against control horizon — the construct-sensitivity figure.
6. Decision curves, raw versus recalibrated.
7. A control-field heatmap paired with its own reliability curve: the paper's
   rhetorical centrepiece, showing an authoritative-looking field beside the
   evidence that its numbers do not mean what they appear to.

---

## 23. Dissertation-level extensions

Three studies that would turn this paper into a programme. Each is a paper.

### Extension 1 — Uncertainty-aware and selectively predictive pitch control

*Motivation.* If the paper finds miscalibration concentrated in identifiable
regions, the constructive response is not a better point estimate but a field
that reports its own reliability.

*Approach.* Replace the point field with a predictive distribution, by (i)
propagating uncertainty in the locomotion parameters and in tracking positions
through the control computation, and (ii) conformal prediction adapted to the
match-clustered, non-exchangeable structure — which is itself a methodological
contribution, since standard conformal guarantees assume exchangeability that
football data violates. Add a selective-prediction layer that abstains where
coverage cannot be guaranteed.

*Deliverable.* A control field with calibrated intervals, and a demonstration
that abstaining where the model is unreliable improves downstream decision
quality more than recalibrating everywhere does.

### Extension 2 — Causal evaluation of off-ball movement

*Motivation.* The most valuable use of pitch control is counterfactual: what
would have happened had the striker made the near-post run? The paper shows that
the field's counterfactual claims are untested. This extension asks what can be
identified.

*Approach.* Treat off-ball movement as a treatment and formalise identification
of `E[Y | do(movement)]` under stated assumptions. Use quasi-experimental
variation — set-piece routines (near-randomised movement patterns), substitutions
as exogenous shocks to personnel, forced tactical changes after dismissals.
Combine with multi-agent behaviour models for a counterfactual distribution over
opponent responses, since a run that assumes a static defence is not a
counterfactual at all.

*Deliverable.* A principled statement of what off-ball-run valuation identifies
and under which assumptions — likely the most theoretically substantial of the
three, and the one most likely to attract a causal-inference audience.

### Extension 3 — Possession control versus player control, via multi-agent models

*Motivation.* Section 7.2's three quantities differ by the physics of the contest
— first touch, orientation, aerial duels. The paper measures the aggregate gap;
this decomposes it.

*Approach.* Model the contest explicitly: a learned contest model conditioned on
approach angle, relative speed, body orientation (from pose estimation) and ball
height, composed with a reachability model. Test whether the composition closes
the calibration gap the paper measures, and whether the two components are
separately calibrated — a composed model can be calibrated overall while both
components are wrong in offsetting ways, which is worth knowing.

*Deliverable.* A decomposition of control into reachability and contest, with
each component separately validated, and an answer to how much of pitch control's
miscalibration is a missing contest model.

### Further directions

- **Transportability across populations.** Women's football, youth football,
  lower divisions — where locomotion envelopes and pressing intensity differ and
  a fixed 5 m/s assumption has no reason to hold. Also the most socially useful
  direction, since these settings have the least analytical infrastructure.
- **Tactical decision-making with calibrated probabilities.** Whether coaches and
  players make better decisions when given calibrated rather than raw values —
  ultimately a behavioural question requiring a field study.
- **Calibration under adversarial adaptation.** If a team knows its opponent uses
  a control model, can it position to induce miscalibration? A game-theoretic
  extension with a genuinely novel flavour.

---

## 24. Publication strategy

No claim is made about acceptance likelihood at any venue.

### 24.1 Sports-analytics venues

| Venue | Fit | Novelty expected |
|---|---|---|
| **MLSA @ ECML PKDD** (Machine Learning and Data Mining for Sports Analytics) | **Primary target.** Exactly the right size and audience; short-paper track; reviewers will grasp the problem in one paragraph; a methodologically careful evaluation paper is squarely in scope | Sound method and a clear empirical answer; a new model is not required |
| **StatsBomb Conference** | Good fit; a practitioner audience that uses these models daily; the downstream section would land well | Applied rigour; practical implications weighted heavily |
| **MIT Sloan Sports Analytics Conference** | Prestigious and visible, but the research track is crowded and slow, and a critique-and-evaluation paper competes against new-method papers | High novelty or a striking result |
| **Barça Sports Analytics Summit** | Strong fit; the venue where several pitch-control papers originated, so the audience knows the models intimately | Applied novelty; domain depth |
| ***Journal of Quantitative Analysis in Sports*** | Good journal home for the extended version; values methodological care | Full rigour; complete robustness reporting |
| ***International Journal of Performance Analysis in Sport*** | Fits the measurement-validity framing | Domain rigour; lighter on statistical novelty |

### 24.2 Machine-learning venues

| Venue | Fit | Novelty expected |
|---|---|---|
| **NeurIPS / ICML workshops** on uncertainty, calibration, or ML for sports | Good fit for the calibration-methodology angle, especially the CORP transfer and the selection-under-choice problem | Methodological insight; an application alone is not enough |
| **AISTATS / UAI** | Only if the selection-and-calibration problem is generalised beyond football — "calibration of counterfactual spatial fields observed only at selected points" is a genuinely general problem | Substantial methodological novelty |
| **KDD applied data science track** | Plausible if framed as measurement validity in a deployed system | Real-world deployment relevance; scale |

*Honest assessment:* as written, this is not a core-ML paper. The methods are
correct applications of existing tools. A core-ML submission would need the
selection problem abstracted and solved in generality, which is Extension 2's
territory.

### 24.3 Statistics and applied-mathematics venues

| Venue | Fit | Novelty expected |
|---|---|---|
| ***Annals of Applied Statistics*** | Good fit if the hierarchical calibration model and the selection analysis are the centrepiece rather than the model comparison | A genuine statistical contribution, not only an application |
| ***Journal of the Royal Statistical Society, Series C*** (Applied Statistics) | Strong fit; JRSS-C publishes exactly this kind of careful applied evaluation | Applied rigour; a real-world problem well solved |
| ***Statistical Modelling*** / ***Electronic Journal of Statistics*** | Possible for a methods-forward version | Methodological novelty |
| ***Significance*** | For an accessible account after the technical paper | Clarity and interest, not novelty |

### 24.4 Recommended sequence

1. **arXiv preprint** on completion — establishes priority and invites correction
   before peer review.
2. **MLSA @ ECML PKDD** short paper — the primary target: right audience, right
   scale, fast feedback.
3. **Extended journal version** (JQAS or JRSS-C) with the full robustness suite,
   the hierarchical model and the downstream analysis.
4. **A practitioner write-up** for a club-analytics audience, focused on the
   decision-displacement result, which is the part a performance analyst can act
   on.

### 24.5 The minimum viable study (9–12 months, one researcher)

**Essential — the paper does not exist without these.**

| Component | Time |
|---|---|
| Data access, licence clearance, and verification of the availability claims in Section 10.3 | 1–2 months *(start immediately; it is the critical path and the only irreducible external dependency)* |
| Preprocessing: synchronisation, arrival detection and typology, labelling | 2 months *(the true time sink; the modelling is not)* |
| Models M0, M1, M2, M2a, M3 | 3 weeks *(already implemented here)* |
| Evaluation harness and calibration analysis | 2 weeks *(already implemented here)* |
| Selection analysis: quasi-exogenous contrast plus reweighting | 1 month |
| Recalibration study (RQ5) | 2 weeks |
| Downstream propagation (RQ6) | 3 weeks |
| Ablations and robustness | 3 weeks |
| Writing | 2 months |

**Optional — improves the paper, is not required for it.**

- Model 4 (GBM) — cheap, worth including.
- Model 5 (Deep Sets) — expensive, omit unless time permits.
- The hierarchical Bayesian model — use the partially pooled estimator instead.
- Cross-competition validation — requires a second competition.
- RQ4 (tracking modality) — requires a second modality.
- Body orientation and pose — requires data most corpora lack.

**The pre-registered fallback.** If tracking access fails, pivot to the
freeze-frame design: event data with freeze frames, velocity-free control models,
imputed arrival times. Same intellectual shape, same clean holdout, larger
sample, weaker construct. The weakening is itself informative, because it
coincides exactly with the "remove velocity" ablation — the fallback study
*measures* what the ablation only simulates.

**The decision point.** Data access must be resolved by month 3. If it is not,
switch to the fallback rather than waiting, because the preprocessing work is not
transferable between the two designs and starting it late is what turns a
twelve-month project into an eighteen-month one.

---

## 25. Final research proposal

*A standalone document for discussion with a supervisor.*

### Are football pitch-control models calibrated probabilities?

**The problem.** Pitch-control models are among the most widely used tools in
football spatial analytics. Given the positions and velocities of twenty-two
players, they return a field over the pitch, `C_A(x, y, t) ∈ [0, 1]`, that is
routinely described — in published papers and in club practice — as the
probability that team A would control the ball were it to arrive at that point
at that time. The probabilistic reading is not decorative. It is load-bearing.
Expected-possession-value frameworks multiply control values by positional
values; space-creation metrics integrate the field over regions; off-ball-run
valuation takes differences of two fields; pass recommendations compare control
values against a threshold. Each of these operations is meaningful only if `C_A`
is a calibrated probability.

That assumption has, as far as I can establish, never been tested. I am not
aware of a published reliability diagram or proper-scoring-rule decomposition
for any pitch-control model. The reason is understandable: control is
counterfactual at almost every point on the pitch, so the field has appeared
unfalsifiable.

**The idea that makes it testable.** The observation this project rests on is
narrow and sufficient. At every pass, cross, deflection, clearance and second
ball, the ball genuinely arrives somewhere. At that point and that moment, the
model's value is an ordinary probability forecast and the realised outcome is an
ordinary draw. A season of tracking data supplies tens of thousands of such
forecast-outcome pairs. Evaluating them is a standard probabilistic forecast
evaluation problem, for which the statistical apparatus — proper scoring rules,
reliability diagrams, score decompositions, recalibration — has been mature in
meteorology for fifty years and simply has not been applied here.

**Why I expect this to find something.** Three structural features make
miscalibration likely in advance, and none is a criticism of the original
authors, who did not claim calibration. First, the leading physics-based models
estimate no parameter from football outcomes; their values follow from an assumed
locomotion envelope, and there is no mechanism by which such a construction would
come out calibrated except by the physics happening to be right. Second, the
parameter that governs the field's sharpness — the arrival-time uncertainty `σ` —
is conventionally fixed at a plausible value; in simulation I find that sweeping
it across a plausible range moves the calibration slope by more than competing
models differ from one another, which means it is not a detail but the thing
being measured. Third, the construct itself is undefined: "control" might mean
first touch, possession a second later, or a retained sequence, and no model can
be calibrated to an undefined quantity.

**Why it would matter.** Most downstream metrics are approximately *linear* in
`C`. That has a sharp consequence: a calibration error with a consistent sign
over a region does not average away when integrated over that region — it
accumulates. A model overconfident by 0.05 in the final third biases every
final-third space metric in the same direction, every match, all season. A random
error would wash out; it is exactly the systematic component, which the
calibration slope measures, that survives aggregation. And because a
miscalibrated forecast displaces every decision threshold, a model whose 0.7 is
really a 0.55 makes systematically over-aggressive recommendations. That is a
football error, not merely a statistical one.

**Design.** I will evaluate five models on identical inputs: a Voronoi
dominant-region baseline; a physics-based control model with ball-control
dynamics; a one-parameter reachability sigmoid (which isolates how much of the
physics model's behaviour is just a squashed time-to-point difference); a
penalised logistic regression on engineered kinematic features; and a
gradient-boosted model. Crucially, the fitted models receive *exactly* the
information the geometric models receive, so any advantage is attributable to
functional form and fitting rather than to extra data.

The outcome is defined operationally and defended: team A in uninterrupted
possession one second after arrival, with stoppages censored rather than coded as
failures. All four candidate definitions and six horizons are reported, because a
conclusion that flips across them is a conclusion that flips.

Splits are by match and, where possible, by competition — never by row. Arrivals
nest within possessions within matches, and a random split would let a model be
rewarded for memorising a match's shape. The primary metric is the
match-cross-fitted miscalibration term of the CORP decomposition of the Brier
score, rather than binned expected calibration error: binned ECE depends on an
arbitrary bin count, is biased, is not a proper scoring rule, and can be driven
near zero simply by coarsening — I have a constructed case in the test suite
where two-bin ECE is under 0.02 and forty-bin ECE is over 0.08 for the same
forecast. All intervals come from a cluster bootstrap resampling whole matches;
the design effect is reported, since on simulated data it is around eight,
implying naive intervals roughly three times too narrow.

**The hard part, and my answer to it.** Passers choose their destinations, so the
evaluation sample is a non-random slice of the pitch, and they choose partly on
information the state vector does not contain. I separate two distinct problems.
Covariate shift — the distribution of destinations differing from the pitch — is
addressable by density-ratio reweighting against a stated reference
distribution, with overlap diagnostics and effective sample size reported.
Selection on unobservables is not addressable by any weighting scheme, and I will
not pretend otherwise. My primary response is a design one: deflections,
clearances and aerial second balls arrive where nobody aimed, so on that
subsample the selection channel is largely closed. Calibration measured there is
my most credible estimate, and the contrast against chosen destinations is itself
a measurement of the selection effect. I will match on observable arrival physics
before making that contrast, and report a sensitivity bound stating how strong
unmeasured selection would have to be to explain the finding away. What remains
unidentified — control where the ball never arrives — I will state as a limit of
observational data rather than dissolve in machinery.

**Data.** The intended primary corpus is a single-tournament optical tracking
release, supplemented by open optical and broadcast-derived samples. I am
treating every dataset property as unverified until checked against the files,
and the repository contains an availability-checking script whose output is
committed alongside results. A fallback design using event data with
freeze-frames is pre-registered: velocity-free control models, imputed arrival
times, a weaker construct on a larger sample. The weakening is informative,
because it coincides exactly with the "remove velocity" ablation.

**Current state.** The full protocol is implemented and tested: all models, the
calibration machinery, leakage-safe splits, the selection layer, the downstream
propagation, and the ablation suite. It runs end-to-end without provider data.
A hundred and twenty tests pass, including eight instrument-validation checks
that confirm the calibration estimator recovers a known injected distortion,
reports near-zero miscalibration for an oracle forecast, demonstrates that AUC is
blind to miscalibration, and produces wider intervals under clustering than a
naive bootstrap. The remaining work is data access and the provider adapters.

**What I want to discuss.** Three things. First, whether the operational
definition of control is defensible, or whether scoring reachability models
against a possession outcome is a category error I should reframe. Second,
whether the quasi-exogenous subsample is large enough to carry the identification
argument, which is a sample-size question I should settle before committing.
Third, whether the right framing is "these models are miscalibrated" or "the
field lacks an instrument for telling" — the second is more defensible and
probably more useful, and it changes how the paper is written.

**Scope discipline.** I will not propose a new pitch-control model. The
contribution is an evaluation protocol and an empirical answer, and a paper that
both criticises existing evaluation and introduces a competitor invites the
reader to assess the competitor instead of the critique. If a better-calibrated
construction falls out, it belongs in a short final section.

*(≈1,320 words.)*

---

## Appendix: repository map

| Path | Contents |
|---|---|
| `src/pcc/geometry.py` | Pitch frame, coordinate normalisation, zoning |
| `src/pcc/kinematics.py` | Smoothing, derivatives, time-to-point models, arrival probability |
| `src/pcc/data/` | Schema contract, labelling, preprocessing, source registry, loaders, simulator |
| `src/pcc/models/` | M0–M5 and the shared interface |
| `src/pcc/calibration/` | Proper scores, CORP, reliability, recalibration, cluster bootstrap |
| `src/pcc/evaluation/` | Leakage-safe splits, the protocol, subgroups, decision curves |
| `src/pcc/selection/` | Candidate arrivals, density-ratio weights, diagnostics, sensitivity |
| `src/pcc/downstream/` | Value surface, EPV, space metrics, decision displacement |
| `src/pcc/viz/` | Reliability diagrams, spatial maps, control fields, decision curves |
| `scripts/` | The numbered pipeline, `run_all.sh` |
| `tests/` | 120 tests, including instrument validation |
| `configs/default.yaml` | Every analyst choice, hashed into each manifest |
| `docs/data_sources.md` | Per-source verification checklists and licensing status |
| `docs/references.bib` | Bibliography, with uncertain entries flagged |
