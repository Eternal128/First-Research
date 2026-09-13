"""Proper scores, calibration statistics and score decompositions.

Why this module is the centre of the study
------------------------------------------
A pitch-control value is a claim about a probability. There are exactly two
ways such a claim can be wrong: it can be *miscalibrated* (0.7 does not happen
70% of the time) or *unsharp* (it says 0.5 everywhere and is technically
honest but useless). A strictly proper scoring rule is the only evaluation that
cannot be gamed by either failure alone, and its decomposition into
miscalibration, discrimination and uncertainty is what separates the two.

Every statistic here accepts ``sample_weight`` so that the same code can be run
on the inverse-propensity-weighted evaluation distribution of Section 14.

Primary metric recommendation
-----------------------------
The study's primary calibration statistic is the **MCB term of the CORP
decomposition** of the Brier score (:func:`corp_decomposition`), not a binned
ECE. Binned ECE depends on an arbitrary bin count, is a biased estimator whose
bias shrinks with the bin width in an uncontrolled way, and can be driven to
zero by coarsening. The CORP construction replaces binning with isotonic
regression (pool-adjacent-violators), which is binning-free, reproducible, and
yields a decomposition that is exactly consistent with the reported Brier
score. Binned ECE, MCE and ACE are retained as secondary, comparability-with-
the-literature metrics and are always reported with their bin count.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

EPS = 1e-12


def _prep(y_true, y_prob, sample_weight=None):
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape}, y_prob {p.shape}")
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
    if w.shape != y.shape:
        raise ValueError(f"sample_weight shape {w.shape} does not match {y.shape}")
    finite = np.isfinite(y) & np.isfinite(p) & np.isfinite(w)
    if not finite.all():
        y, p, w = y[finite], p[finite], w[finite]
    if y.size == 0:
        raise ValueError("no finite observations")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be binary 0/1")
    return y, np.clip(p, 0.0, 1.0), w


# ---------------------------------------------------------------------------
# Proper scores
# ---------------------------------------------------------------------------
def brier_score(y_true, y_prob, sample_weight=None) -> float:
    """Mean squared error of the probability forecast; strictly proper."""
    y, p, w = _prep(y_true, y_prob, sample_weight)
    return float(np.average((p - y) ** 2, weights=w))


def log_loss(y_true, y_prob, sample_weight=None, *, eps: float = 1e-6) -> float:
    """Negative log-likelihood per observation; strictly proper, unbounded.

    ``eps`` clipping is unavoidable for models (M1) that emit hard 0/1
    forecasts. Because the clip value changes the number, the log loss is
    reported as a secondary metric and the clip is stated alongside it. The
    Brier score is the primary proper score for exactly this reason.
    """
    y, p, w = _prep(y_true, y_prob, sample_weight)
    p = np.clip(p, eps, 1.0 - eps)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def spherical_score(y_true, y_prob, sample_weight=None) -> float:
    """Spherical score (higher is better); a second proper rule as robustness.

    Reported because conclusions that flip between the Brier and the spherical
    score are conclusions about the loss function rather than about the model.
    """
    y, p, w = _prep(y_true, y_prob, sample_weight)
    num = np.where(y == 1, p, 1 - p)
    den = np.sqrt(p**2 + (1 - p) ** 2)
    return float(np.average(num / np.maximum(den, EPS), weights=w))


# ---------------------------------------------------------------------------
# Isotonic (binning-free) calibration and the CORP decomposition
# ---------------------------------------------------------------------------
def isotonic_recalibrate(y_true, y_prob, sample_weight=None) -> np.ndarray:
    """Pool-adjacent-violators fit of ``y`` on ``p``; the CORP recalibration.

    Returns the in-sample isotonic fit evaluated at each forecast. This is the
    conditional event frequency given the forecast, subject only to a
    monotonicity assumption, and is what the reliability curve *should* be if
    the forecast were calibrated.
    """
    from sklearn.isotonic import IsotonicRegression

    y, p, w = _prep(y_true, y_prob, sample_weight)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    return np.asarray(iso.fit_transform(p, y, sample_weight=w), dtype=float)


@dataclass(frozen=True)
class CORPDecomposition:
    """CORP decomposition of the mean Brier score: ``score = MCB - DSC + UNC``."""

    score: float
    miscalibration: float   # MCB; 0 iff the forecast is (isotonically) calibrated
    discrimination: float   # DSC; 0 iff the forecast carries no information
    uncertainty: float      # UNC; the score of the climatological forecast
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def corp_decomposition(y_true, y_prob, sample_weight=None) -> CORPDecomposition:
    """Binning-free decomposition of the Brier score via isotonic regression.

    .. math::
        \\bar S(x) = \\underbrace{\\bar S(x) - \\bar S(\\hat x)}_{\\mathrm{MCB}}
        - \\underbrace{\\bar S(\\bar y) - \\bar S(\\hat x)}_{\\mathrm{DSC}}
        + \\underbrace{\\bar S(\\bar y)}_{\\mathrm{UNC}}

    where :math:`\\hat x` is the PAV-recalibrated forecast and :math:`\\bar y`
    the (weighted) base rate.

    Interpretation for this study:

    * ``MCB`` is the Brier-score cost of the forecast's miscalibration. It is
      the headline number for RQ1. It is non-negative by construction and is
      on the same scale as the score itself, which makes "how much does
      miscalibration cost?" answerable in the units the field already reports.
    * ``DSC`` is sharpness/resolution. A model may reduce MCB by flattening
      toward the base rate, which shows up immediately as a fall in DSC. The
      two must be read together; reporting MCB alone reproduces the mistake the
      study is criticising.
    * ``UNC`` depends only on the evaluation sample and is therefore the number
      that must be held fixed when comparing models, and reported whenever
      comparing across subgroups (a zone with base rate 0.9 has small UNC and
      small scores for trivial reasons).

    Caveat: MCB estimated in-sample is biased **upward**, not downward. The PAV
    fit is the best monotone fit *on this sample*, so it absorbs sampling noise
    as well as real miscalibration; ``S(x_hat)`` is therefore too low and the
    difference ``S(x) - S(x_hat)`` too high. A perfectly calibrated forecast
    will show a small positive in-sample MCB purely from this effect, and the
    bias grows as the sample shrinks. Use :func:`corp_decomposition_cv` for any
    headline claim: cross-fitting removes the bias, at the cost of allowing
    slightly negative estimates, which should be reported as such rather than
    truncated at zero.
    """
    y, p, w = _prep(y_true, y_prob, sample_weight)
    fitted = isotonic_recalibrate(y, p, w)
    base = float(np.average(y, weights=w))

    s_x = float(np.average((p - y) ** 2, weights=w))
    s_hat = float(np.average((fitted - y) ** 2, weights=w))
    s_bar = float(np.average((base - y) ** 2, weights=w))

    return CORPDecomposition(
        score=s_x,
        miscalibration=max(s_x - s_hat, 0.0),
        discrimination=max(s_bar - s_hat, 0.0),
        uncertainty=s_bar,
        n=int(y.size),
    )


def corp_decomposition_cv(
    y_true, y_prob, groups, sample_weight=None, *, n_splits: int = 5, random_state: int = 0
) -> CORPDecomposition:
    """Cross-fitted CORP decomposition, with folds grouped by ``groups``.

    The isotonic recalibration is fitted on out-of-fold data and applied to the
    held-out fold, so ``MCB`` is not inflated by PAV's in-sample flexibility.
    ``groups`` should be the match identifier: arrivals from the same match are
    dependent, and splitting within a match would let the recalibration see the
    same possession on both sides of the split.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold

    y, p, w = _prep(y_true, y_prob, sample_weight)
    g = np.asarray(groups).ravel()[: y.size]
    n_groups = len(np.unique(g))
    n_splits = int(min(n_splits, max(2, n_groups)))
    if n_groups < 2:
        return corp_decomposition(y, p, w)

    fitted = np.empty_like(p)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(p, y, groups=g):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        iso.fit(p[train_idx], y[train_idx], sample_weight=w[train_idx])
        fitted[test_idx] = iso.predict(p[test_idx])

    base = float(np.average(y, weights=w))
    s_x = float(np.average((p - y) ** 2, weights=w))
    s_hat = float(np.average((fitted - y) ** 2, weights=w))
    s_bar = float(np.average((base - y) ** 2, weights=w))
    return CORPDecomposition(
        score=s_x,
        miscalibration=s_x - s_hat,
        discrimination=s_bar - s_hat,
        uncertainty=s_bar,
        n=int(y.size),
    )


def murphy_decomposition(y_true, y_prob, sample_weight=None, *, n_bins: int = 10) -> dict[str, float]:
    """Classical binned decomposition: ``Brier = REL - RES + UNC + residual``.

    Provided for comparability with the meteorological and sports-analytics
    literature that reports it. It is *not* the study's primary statistic, for
    two reasons.

    First, the reliability and resolution terms both depend on the binning, and
    the dependence is not monotone, so two papers using different bin counts
    are not comparable.

    Second - and this is usually left unsaid - the three-term identity is exact
    only when the forecast takes finitely many distinct values, one per bin.
    For a continuous forecast there is a residual

    .. math::
        \\sum_b w_b \\bigl[\\mathrm{Var}_b(p) - 2\\,\\mathrm{Cov}_b(p, y)\\bigr],

    from the spread of forecasts *within* each bin. It is returned as
    ``residual`` so that the identity can be checked rather than assumed; a
    large residual means the bins are too coarse for the decomposition to be
    meaningful at all.
    """
    y, p, w = _prep(y_true, y_prob, sample_weight)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    base = float(np.average(y, weights=w))
    total_w = w.sum()

    rel = res = residual = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        pb = float(np.average(p[m], weights=w[m]))
        yb = float(np.average(y[m], weights=w[m]))
        rel += wb * (pb - yb) ** 2
        res += wb * (yb - base) ** 2
        var_p = float(np.average((p[m] - pb) ** 2, weights=w[m]))
        cov_py = float(np.average((p[m] - pb) * (y[m] - yb), weights=w[m]))
        residual += wb * (var_p - 2.0 * cov_py)

    return {
        "reliability": rel / total_w,
        "resolution": res / total_w,
        "uncertainty": base * (1.0 - base),
        "residual": residual / total_w,
        "n_bins": float(n_bins),
    }


# ---------------------------------------------------------------------------
# Binned calibration errors (secondary metrics)
# ---------------------------------------------------------------------------
def _bin_stats(p, y, w, edges):
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        wb = float(w[m].sum())
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": int(m.sum()),
                "weight": wb,
                "mean_pred": float(np.average(p[m], weights=w[m])),
                "obs_freq": float(np.average(y[m], weights=w[m])),
            }
        )
    return rows


def expected_calibration_error(
    y_true, y_prob, sample_weight=None, *, n_bins: int = 15, strategy: str = "uniform"
) -> float:
    """Weighted mean absolute gap between forecast and observed frequency.

    ``strategy='uniform'`` gives equal-width bins (the standard ECE);
    ``strategy='quantile'`` gives equal-mass bins, which is the adaptive
    calibration error (ACE). ACE is preferred when the forecast distribution is
    highly concentrated - which pitch-control outputs are, since most arrivals
    occur in space the passing team already dominates - because equal-width
    bins then leave most bins nearly empty and the ECE is dominated by noise in
    a handful of them.

    Known limitations, stated here because the study must not lean on this
    number: ECE is a biased estimator of the true calibration error, the bias
    depends on ``n_bins`` and sample size, and it is not a proper scoring rule,
    so a model can lower its ECE without improving as a forecast.
    """
    y, p, w = _prep(y_true, y_prob, sample_weight)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        edges[0], edges[-1] = 0.0, 1.0
        if edges.size < 3:
            edges = np.linspace(0.0, 1.0, 3)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    rows = _bin_stats(p, y, w, edges)
    total_w = w.sum()
    return float(sum(r["weight"] * abs(r["mean_pred"] - r["obs_freq"]) for r in rows) / total_w)


def maximum_calibration_error(y_true, y_prob, sample_weight=None, *, n_bins: int = 15, min_count: int = 30) -> float:
    """Largest absolute bin-level gap, ignoring bins below ``min_count``.

    The ``min_count`` guard matters: without it the MCE is almost always
    reported from the emptiest bin and is an estimate of sampling noise.
    """
    y, p, w = _prep(y_true, y_prob, sample_weight)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = [r for r in _bin_stats(p, y, w, edges) if r["n"] >= min_count]
    if not rows:
        return float("nan")
    return float(max(abs(r["mean_pred"] - r["obs_freq"]) for r in rows))


def adaptive_calibration_error(y_true, y_prob, sample_weight=None, *, n_bins: int = 15) -> float:
    """Equal-mass-bin calibration error (ACE)."""
    return expected_calibration_error(y_true, y_prob, sample_weight, n_bins=n_bins, strategy="quantile")


# ---------------------------------------------------------------------------
# Calibration slope and intercept (the "calibration hierarchy" view)
# ---------------------------------------------------------------------------
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))



def _unpenalised_logistic(cls):
    """Construct an unpenalised ``LogisticRegression`` across scikit-learn versions.

    The recalibration regression must be unpenalised: any shrinkage would bias
    the estimated calibration slope toward zero and so would manufacture
    exactly the overconfidence the study is trying to measure. scikit-learn
    spells "no penalty" as ``penalty=None`` up to 1.7 and as ``C=np.inf`` from
    1.8, so both are attempted.
    """
    try:
        return cls(C=np.inf, max_iter=1000)
    except (TypeError, ValueError):
        return cls(penalty=None, max_iter=1000)


def calibration_slope_intercept(y_true, y_prob, sample_weight=None, *, eps: float = 1e-6) -> dict[str, float]:
    """Weak-calibration diagnostics from a logistic recalibration regression.

    Fits :math:`\\operatorname{logit} P(Y=1) = a + b \\operatorname{logit}(p)`.

    * ``slope`` :math:`b`: 1 under calibration. :math:`b < 1` is the signature
      of **overconfidence** - the forecast spreads further toward 0 and 1 than
      the evidence supports. This is the single most likely finding for an
      unfitted geometric model, and it is the one that most directly
      contaminates downstream possession-value sums, which are approximately
      linear in ``C``.
    * ``intercept_in_the_large`` :math:`a_0`: fitted with the slope fixed at 1
      (an offset model), it measures systematic over- or under-forecasting of
      the base rate.

    Both are reported with cluster-bootstrap intervals; a slope confidence
    interval excluding 1 is the formal test for H1b.
    """
    from sklearn.linear_model import LogisticRegression

    y, p, w = _prep(y_true, y_prob, sample_weight)
    z = _logit(p, eps)
    out: dict[str, float] = {}

    if np.ptp(z) < 1e-9 or len(np.unique(y)) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "intercept_in_the_large": float("nan")}

    lr = _unpenalised_logistic(LogisticRegression)
    lr.fit(z.reshape(-1, 1), y.astype(int), sample_weight=w)
    out["slope"] = float(lr.coef_.ravel()[0])
    out["intercept"] = float(lr.intercept_[0])

    # Calibration-in-the-large: slope fixed at 1, intercept free. Solved
    # directly by one-dimensional root finding on the score equation
    # sum_i w_i (y_i - sigmoid(a + z_i)) = 0, which is monotone in a.
    from scipy.optimize import brentq

    def score_eq(a: float) -> float:
        s = 1.0 / (1.0 + np.exp(-np.clip(a + z, -500, 500)))
        return float(np.sum(w * (y - s)))

    lo, hi = -20.0, 20.0
    try:
        out["intercept_in_the_large"] = float(brentq(score_eq, lo, hi, xtol=1e-8))
    except ValueError:
        out["intercept_in_the_large"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Discrimination (reported alongside, never instead of, calibration)
# ---------------------------------------------------------------------------
def roc_auc(y_true, y_prob, sample_weight=None) -> float:
    """Area under the ROC curve. Invariant to any monotone recalibration.

    That invariance is the point: AUC cannot detect miscalibration at all. It
    is reported so that a reader can see that two models with identical
    discrimination can differ arbitrarily in calibration, which is the study's
    central methodological message.
    """
    from sklearn.metrics import roc_auc_score

    y, p, w = _prep(y_true, y_prob, sample_weight)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p, sample_weight=w))


def average_precision(y_true, y_prob, sample_weight=None) -> float:
    """Area under the precision-recall curve; informative when classes are skewed."""
    from sklearn.metrics import average_precision_score

    y, p, w = _prep(y_true, y_prob, sample_weight)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p, sample_weight=w))


def sharpness(y_prob, sample_weight=None) -> dict[str, float]:
    """Dispersion of the forecast distribution, independent of outcomes.

    Reported because a calibration comparison without a sharpness comparison
    cannot distinguish "honest" from "uninformative".
    """
    p = np.asarray(y_prob, dtype=float).ravel()
    w = np.ones_like(p) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
    mean = float(np.average(p, weights=w))
    var = float(np.average((p - mean) ** 2, weights=w))
    return {
        "mean_forecast": mean,
        "forecast_variance": var,
        "forecast_sd": float(np.sqrt(var)),
        "frac_below_0.05": float(np.average(p < 0.05, weights=w)),
        "frac_above_0.95": float(np.average(p > 0.95, weights=w)),
    }


# ---------------------------------------------------------------------------
# Convenience: the full report for one model on one evaluation set
# ---------------------------------------------------------------------------
def full_report(
    y_true,
    y_prob,
    sample_weight=None,
    groups=None,
    *,
    n_bins: int = 15,
    cv_corp: bool = True,
) -> dict[str, float]:
    """All scalar metrics for one (model, evaluation set) pair."""
    y, p, w = _prep(y_true, y_prob, sample_weight)
    corp = (
        corp_decomposition_cv(y, p, groups, w)
        if (cv_corp and groups is not None)
        else corp_decomposition(y, p, w)
    )
    murphy = murphy_decomposition(y, p, w, n_bins=10)
    slope = calibration_slope_intercept(y, p, w)

    return {
        "n": float(y.size),
        "base_rate": float(np.average(y, weights=w)),
        "brier": brier_score(y, p, w),
        "log_loss": log_loss(y, p, w),
        "spherical": spherical_score(y, p, w),
        "corp_mcb": corp.miscalibration,
        "corp_dsc": corp.discrimination,
        "corp_unc": corp.uncertainty,
        "murphy_reliability": murphy["reliability"],
        "murphy_resolution": murphy["resolution"],
        "ece_uniform": expected_calibration_error(y, p, w, n_bins=n_bins, strategy="uniform"),
        "ace_quantile": adaptive_calibration_error(y, p, w, n_bins=n_bins),
        "mce": maximum_calibration_error(y, p, w, n_bins=n_bins),
        "calibration_slope": slope["slope"],
        "calibration_intercept": slope["intercept"],
        "calibration_in_the_large": slope["intercept_in_the_large"],
        "auc": roc_auc(y, p, w),
        "average_precision": average_precision(y, p, w),
        **sharpness(p, w),
    }
