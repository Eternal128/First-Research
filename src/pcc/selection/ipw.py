"""Selection bias: where ball arrivals actually happen, and what that costs.

The problem
-----------
A pitch-control field is defined over the whole pitch, but the ball only ever
arrives somewhere a player chose to send it. Players preferentially pass into
space they believe they control. Let ``D = 1`` indicate that an arrival at
state/destination ``X`` was observed. What the data identify is

.. math:: P(Y = 1 \\mid X, D = 1),

while the quantity the pitch-control field claims to describe is

.. math:: P(Y = 1 \\mid X).

These coincide if and only if :math:`Y \\perp D \\mid X`. Two distinct
obstacles stand between the two, and they need different treatment:

**(a) Covariate shift.** Even under :math:`Y \\perp D \\mid X`, the *distribution*
of ``X`` among observed arrivals is far from the distribution over the pitch.
Conditional calibration at the level of ``X`` transfers, but *marginal*,
binned calibration does not: a reliability bin aggregates over ``X`` and its
composition changes with the evaluation distribution. This is the part that
reweighting genuinely fixes, and :func:`selection_weights` implements it.

**(b) Selection on unobservables.** The passer knows things the state vector
does not: that the receiver has already started the run, that the defender has
mis-stepped, that the pass will be disguised. Conditional on ``X``, observed
arrivals are then systematically the *easy* ones, and :math:`Y \\not\\perp D
\\mid X`. **No reweighting scheme fixes this**, because the required
information is absent from the data by construction. Inverse propensity
weighting must not be presented as though it did.

The study's response to (b) is therefore threefold, in descending order of
strength:

1. **Quasi-exogenous arrivals.** Deflections, clearances, blocked passes and
   aerial second balls arrive at destinations no one selected. Their
   destination is not chosen conditional on private information about control,
   so on this subsample the selection channel is largely closed. Calibration
   measured here is the study's most credible estimate, and the contrast
   against chosen destinations is a direct, if partial, measurement of the
   selection effect itself.
2. **Reweighting** to a stated target distribution, for (a) only, with overlap
   diagnostics that say where the reweighted estimate is supported and where it
   is extrapolation.
3. **Sensitivity analysis** that reports how strong unmeasured selection would
   have to be to overturn the conclusion (:func:`tipping_point_analysis`).

Nothing here identifies control at locations where the ball never arrives. That
is a genuine limit of observational tracking data and the proposal states it as
such rather than dissolving it in machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pcc.data.schema import ArrivalFrame
from pcc.geometry import Pitch


# ---------------------------------------------------------------------------
# Candidate (counterfactual) arrivals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateConfig:
    """How counterfactual destinations are proposed for the density-ratio fit."""

    n_per_arrival: int = 8
    proposal: str = "reachable"   # 'uniform' | 'reachable' | 'perturb'
    perturb_sd: float = 8.0       # metres, for 'perturb'
    max_pass_length: float = 60.0
    random_state: int = 0


def generate_candidates(
    frames: list[ArrivalFrame],
    config: CandidateConfig | None = None,
    *,
    pitch: Pitch = Pitch(),
) -> tuple[list[ArrivalFrame], np.ndarray]:
    """Propose destinations the passer *could* have chosen but did not.

    Returns the candidate frames and the index of the real arrival each was
    derived from. Candidates carry **no outcome label** - they cannot, because
    the ball never went there. Their only role is to estimate the density of
    chosen destinations relative to a reference proposal, which is what makes
    the reweighting possible.

    Proposal distributions
    ----------------------
    ``uniform``
        Uniform over the playing area. Simple and target-like, but produces
        many candidates that no player would ever attempt (a 90 m cross-field
        pass into the corner), which makes the density ratio extreme and the
        weights unstable.
    ``reachable``
        Uniform over the disc of radius ``max_pass_length`` about the release
        point, intersected with the pitch. The recommended default: it restricts
        attention to the physically plausible option set, which is closer to the
        decision a player actually faces, and keeps the estimated weights in a
        range where the overlap diagnostics are meaningful.
    ``perturb``
        Gaussian jitter about the realised destination. Useful as a local
        sensitivity check (does calibration change over a few metres?), not as a
        target distribution.
    """
    cfg = config or CandidateConfig()
    rng = np.random.default_rng(cfg.random_state)
    out: list[ArrivalFrame] = []
    parent: list[int] = []

    for i, f in enumerate(frames):
        origin = np.asarray(f.meta.get("origin", f.target), dtype=float).reshape(2)
        for _ in range(cfg.n_per_arrival):
            if cfg.proposal == "uniform":
                t = np.array(
                    [
                        rng.uniform(-pitch.half_length, pitch.half_length),
                        rng.uniform(-pitch.half_width, pitch.half_width),
                    ]
                )
            elif cfg.proposal == "reachable":
                r = cfg.max_pass_length * np.sqrt(rng.uniform())
                theta = rng.uniform(0, 2 * np.pi)
                t = origin + r * np.array([np.cos(theta), np.sin(theta)])
                t = pitch.clip(t[None, :])[0]
            elif cfg.proposal == "perturb":
                t = pitch.clip((f.target + rng.normal(0, cfg.perturb_sd, 2))[None, :])[0]
            else:
                raise ValueError(f"unknown proposal {cfg.proposal!r}")

            dist = float(np.linalg.norm(t - origin))
            own_dist = float(np.linalg.norm(f.target - origin))
            speed = own_dist / max(f.flight_time, 1e-6) if own_dist > 1e-6 else 15.0
            out.append(
                ArrivalFrame(
                    att_xy=f.att_xy, att_v=f.att_v, def_xy=f.def_xy, def_v=f.def_v,
                    target=t, flight_time=max(dist / max(speed, 1e-6), 1e-3),
                    att_is_gk=f.att_is_gk, def_is_gk=f.def_is_gk,
                    att_vmax=f.att_vmax, def_vmax=f.def_vmax,
                    meta={**f.meta, "candidate_of": i, "is_candidate": True},
                )
            )
            parent.append(i)
    return out, np.asarray(parent, dtype=int)


# ---------------------------------------------------------------------------
# Density-ratio ("propensity") estimation and weights
# ---------------------------------------------------------------------------
@dataclass
class SelectionModel:
    """Classifier separating chosen destinations from proposed ones.

    Under case-control sampling the fitted classifier estimates
    :math:`P(D=1 \\mid X)` only up to the sampling odds, but the *ratio*
    :math:`p_{\\text{obs}}(X) / q(X)` it implies is identified up to that same
    constant, which cancels when weights are normalised. So the constant does
    not matter for reweighting, and this class does not pretend to estimate an
    absolute propensity.
    """

    model: object = None
    feature_names: list[str] | None = None
    n_real: int = 0
    n_candidate: int = 0

    def fit(self, real_X: pd.DataFrame, cand_X: pd.DataFrame, *, random_state: int = 0) -> "SelectionModel":
        from sklearn.ensemble import HistGradientBoostingClassifier

        X = pd.concat([real_X, cand_X], ignore_index=True)
        d = np.concatenate([np.ones(len(real_X)), np.zeros(len(cand_X))])
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, min_samples_leaf=50, random_state=random_state
        )
        self.model.fit(X, d)
        self.feature_names = list(X.columns)
        self.n_real, self.n_candidate = len(real_X), len(cand_X)
        return self

    def odds(self, X: pd.DataFrame) -> np.ndarray:
        """``P(D=1 | X) / P(D=0 | X)``, proportional to ``p_obs(X) / q(X)``."""
        if self.model is None:
            raise RuntimeError("SelectionModel is not fitted")
        p = np.clip(self.model.predict_proba(X[self.feature_names])[:, 1], 1e-6, 1 - 1e-6)
        return p / (1 - p)

    def discrimination(self, real_X: pd.DataFrame, cand_X: pd.DataFrame) -> float:
        """AUC separating chosen from proposed destinations.

        A diagnostic worth reporting in its own right: it quantifies *how
        strongly* passers select their destinations. An AUC near 0.5 would mean
        destinations are effectively random and the whole selection concern is
        moot; a high AUC means the observed evaluation set is a highly
        non-random slice of the pitch, which is the premise of this section.
        """
        from sklearn.metrics import roc_auc_score

        X = pd.concat([real_X, cand_X], ignore_index=True)
        d = np.concatenate([np.ones(len(real_X)), np.zeros(len(cand_X))])
        return float(roc_auc_score(d, self.model.predict_proba(X[self.feature_names])[:, 1]))


def selection_weights(
    selection_model: SelectionModel,
    real_X: pd.DataFrame,
    *,
    stabilise: bool = True,
    trim_quantile: float = 0.99,
) -> np.ndarray:
    """Weights that reweight observed arrivals toward the proposal distribution.

    .. math::
        w(X) \;\\propto\; \\frac{q(X)}{p_{\\text{obs}}(X)}
        \;\\propto\; \\frac{1}{\\mathrm{odds}(X)}

    so an arrival at a destination that passers strongly favour is
    *down*-weighted, and a rare, contested destination is up-weighted. The
    result answers "how calibrated would this model be if the ball arrived
    according to ``q`` rather than according to player choice?"

    ``stabilise`` normalises the weights to mean one, which removes the
    case-control constant and leaves the effective sample size interpretable.
    ``trim_quantile`` caps the upper tail: a handful of arrivals in regions
    almost never targeted would otherwise dominate every weighted statistic,
    converting a bias problem into a variance problem. Trimming reintroduces a
    small bias, and both the trim point and the share of weight it removes are
    reported.
    """
    w = 1.0 / np.maximum(selection_model.odds(real_X), 1e-9)
    if trim_quantile is not None and 0 < trim_quantile < 1:
        cap = float(np.quantile(w, trim_quantile))
        w = np.minimum(w, cap)
    if stabilise:
        w = w / np.mean(w)
    return w


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size, :math:`(\\sum w)^2 / \\sum w^2`.

    The number to quote whenever a weighted result is reported. If reweighting
    an evaluation set of 40,000 arrivals leaves an effective size of 3,000, the
    weighted confidence intervals must widen accordingly and the study should
    say so rather than reporting the nominal n.
    """
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.maximum((w**2).sum(), 1e-12))


def overlap_diagnostics(
    selection_model: SelectionModel, real_X: pd.DataFrame, cand_X: pd.DataFrame
) -> dict[str, float]:
    """Positivity / common-support diagnostics for the reweighting.

    Reweighting is only credible where both chosen and unchosen destinations
    occur. Regions of ``X`` in which essentially every point is chosen (or
    none is) have no overlap, the density ratio there is an extrapolation, and
    conclusions about them rest on the classifier's functional form rather than
    on data. The reported quantities are the share of observed arrivals in
    low-overlap regions and the weight concentration.
    """
    odds_real = selection_model.odds(real_X)
    odds_cand = selection_model.odds(cand_X)
    p_real = odds_real / (1 + odds_real)
    p_cand = odds_cand / (1 + odds_cand)

    lo, hi = float(np.quantile(p_cand, 0.01)), float(np.quantile(p_cand, 0.99))
    outside = float(np.mean((p_real < lo) | (p_real > hi)))
    w = 1.0 / np.maximum(odds_real, 1e-9)
    w = w / w.mean()
    order = np.sort(w)[::-1]
    top1 = float(order[: max(1, len(order) // 100)].sum() / order.sum())

    return {
        "auc_selection": selection_model.discrimination(real_X, cand_X),
        "frac_real_outside_candidate_support": outside,
        "weight_share_top_1pct": top1,
        "effective_sample_size": effective_sample_size(w),
        "ess_ratio": effective_sample_size(w) / len(w),
        "max_weight": float(w.max()),
        "median_weight": float(np.median(w)),
    }


# ---------------------------------------------------------------------------
# Stratification, matching and sensitivity
# ---------------------------------------------------------------------------
def propensity_strata(odds: np.ndarray, *, n_strata: int = 5) -> np.ndarray:
    """Quintiles of the estimated selection score.

    Stratified reporting is the transparent alternative to weighting: instead
    of one reweighted number whose construction the reader must trust, five
    numbers with their own sample sizes. The study reports both, and treats a
    disagreement between them as evidence that the weights are doing something
    fragile.
    """
    odds = np.asarray(odds, dtype=float)
    score = odds / (1 + odds)
    edges = np.quantile(score, np.linspace(0, 1, n_strata + 1))
    edges = np.unique(edges)
    return np.clip(np.digitize(score, edges[1:-1]), 0, max(edges.size - 2, 0))


def negative_control_comparison(
    df: pd.DataFrame,
    *,
    prob_col: str,
    outcome_col: str,
    type_col: str = "arrival_type",
    exogenous_types=("deflection", "clearance", "second_ball"),
    cluster_col: str = "match_id",
    n_boot: int = 400,
) -> pd.DataFrame:
    """Calibration on chosen versus unchosen destinations.

    The study's primary evidence about selection. If a model is well calibrated
    on deflections and clearances but overconfident on chosen passes, the
    excess confidence is attributable to what passers know and the model does
    not - a substantive finding about the construct, not a nuisance.

    The comparison is not clean in one respect that must be stated: the two
    subsamples also differ in *football* - deflections are faster, more often
    aerial and more often contested - so the contrast confounds selection with
    arrival physics. The proposal's remedy is to match on the observable
    differences (flight time, height, pitch zone, local player density) before
    comparing, which :func:`matched_subsample` provides.
    """
    from pcc.calibration.bootstrap import cluster_bootstrap
    from pcc.calibration.metrics import calibration_slope_intercept, corp_decomposition

    rows = []
    df = df.copy()
    df["_exogenous"] = df[type_col].isin(list(exogenous_types))
    for flag, sub in df.groupby("_exogenous"):
        if len(sub) < 100 or sub[outcome_col].nunique() < 2:
            rows.append({"subsample": "unchosen" if flag else "chosen", "n": len(sub), "suppressed": True})
            continue
        y, p = sub[outcome_col].to_numpy(), sub[prob_col].to_numpy()
        corp = corp_decomposition(y, p)
        slope = calibration_slope_intercept(y, p)
        work = pd.DataFrame({"y": y, "p": p, cluster_col: sub[cluster_col].to_numpy()})
        boot = cluster_bootstrap(
            lambda d: corp_decomposition(d["y"], d["p"]).miscalibration,
            work, cluster_col=cluster_col, n_boot=n_boot,
        ) if work[cluster_col].nunique() >= 5 else None
        rows.append(
            {
                "subsample": "unchosen" if flag else "chosen",
                "n": len(sub),
                "suppressed": False,
                "base_rate": float(np.mean(y)),
                "mean_forecast": float(np.mean(p)),
                "corp_mcb": corp.miscalibration,
                "corp_dsc": corp.discrimination,
                "calibration_slope": slope["slope"],
                "mcb_lo": boot.lo if boot else np.nan,
                "mcb_hi": boot.hi if boot else np.nan,
            }
        )
    return pd.DataFrame(rows)


def matched_subsample(
    df: pd.DataFrame,
    *,
    treat_col: str,
    match_cols: list[str],
    caliper: float = 0.2,
    random_state: int = 0,
) -> pd.DataFrame:
    """1:1 nearest-neighbour match on a propensity score built from ``match_cols``.

    Used to make the chosen/unchosen contrast comparable on observable arrival
    physics. Returns the matched rows with a ``match_pair`` column. Balance
    after matching must be reported (standardised mean differences); an
    unbalanced match is not a match.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    work = df.dropna(subset=match_cols + [treat_col]).copy()
    X = work[match_cols]
    t = work[treat_col].astype(int).to_numpy()
    if len(np.unique(t)) < 2:
        return work.iloc[0:0]

    clf = HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=40, random_state=random_state)
    clf.fit(X, t)
    logit = np.log(np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6) / np.clip(1 - clf.predict_proba(X)[:, 1], 1e-6, 1))
    work["_ps_logit"] = logit
    sd = float(np.std(logit)) or 1.0

    treated = work[t == 1].sort_values("_ps_logit")
    control = work[t == 0].sort_values("_ps_logit").reset_index(drop=True)
    if treated.empty or control.empty:
        return work.iloc[0:0]

    ctrl_vals = control["_ps_logit"].to_numpy()
    used = np.zeros(len(control), dtype=bool)
    pairs = []
    for pair_id, (_, row) in enumerate(treated.iterrows()):
        diffs = np.abs(ctrl_vals - row["_ps_logit"])
        diffs[used] = np.inf
        j = int(np.argmin(diffs))
        if not np.isfinite(diffs[j]) or diffs[j] > caliper * sd:
            continue
        used[j] = True
        pairs.append(row.to_frame().T.assign(match_pair=pair_id))
        pairs.append(control.iloc[[j]].assign(match_pair=pair_id))
    return pd.concat(pairs, ignore_index=True) if pairs else work.iloc[0:0]


def standardised_mean_differences(df: pd.DataFrame, *, treat_col: str, cols: list[str]) -> pd.DataFrame:
    """Covariate balance before/after matching. |SMD| < 0.1 is the usual target."""
    rows = []
    a = df[df[treat_col].astype(bool)]
    b = df[~df[treat_col].astype(bool)]
    for c in cols:
        ma, mb = a[c].mean(), b[c].mean()
        sa, sb = a[c].std(), b[c].std()
        pooled = np.sqrt((sa**2 + sb**2) / 2)
        rows.append(
            {"variable": c, "mean_treated": ma, "mean_control": mb,
             "smd": float((ma - mb) / pooled) if pooled > 0 else np.nan}
        )
    return pd.DataFrame(rows)


def tipping_point_analysis(
    y_true, y_prob, *, gamma_grid=None, sample_weight=None
) -> pd.DataFrame:
    """How strong would unmeasured selection have to be to erase the finding?

    Models unmeasured selection as a multiplicative bias on the odds of the
    outcome among observed arrivals: for a sensitivity parameter
    :math:`\\Gamma \\ge 1`, the "true" probability at a given state is bounded by

    .. math::
        p^{\\pm}(x) = \\frac{\\Gamma^{\\pm 1}\\, p(x)}
        {\\Gamma^{\\pm 1} p(x) + (1 - p(x))} .

    Recomputing the calibration statistic under the deflated bound
    :math:`p^-` answers: *if observed arrivals are systematically easier than
    the state alone implies, by an odds factor of :math:`\\Gamma`, would the
    model still look miscalibrated?* The reported quantity is the smallest
    :math:`\\Gamma` at which the miscalibration statistic would be explained
    away.

    This is a bounding argument in the spirit of Rosenbaum sensitivity
    analysis, not an identification result. It says how bad the unmeasured
    selection would have to be; it cannot say how bad it is.
    """
    from pcc.calibration.metrics import calibration_slope_intercept, corp_decomposition

    y = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    if gamma_grid is None:
        gamma_grid = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    rows = []
    for g in gamma_grid:
        for direction, factor in (("deflated", 1.0 / g), ("inflated", g)):
            odds = factor * p / (1 - p)
            p_adj = odds / (1 + odds)
            corp = corp_decomposition(y, p_adj, sample_weight)
            slope = calibration_slope_intercept(y, p_adj, sample_weight)
            rows.append(
                {
                    "gamma": float(g),
                    "direction": direction,
                    "corp_mcb": corp.miscalibration,
                    "calibration_slope": slope["slope"],
                    "calibration_in_the_large": slope["intercept_in_the_large"],
                    "mean_forecast": float(np.mean(p_adj)),
                    "base_rate": float(np.mean(y)),
                }
            )
    return pd.DataFrame(rows)
