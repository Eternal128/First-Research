"""Decision-curve analysis and cost-sensitive utility.

The step that separates a statistics paper from a football paper. A model can
improve its Brier score by a statistically clear but practically irrelevant
amount. Decision-curve analysis asks a different question: at a given
*threshold probability* - the level of control probability above which a
decision-maker would act - does using the model produce more net benefit than
the default policies of always acting or never acting?

.. math::
    \\mathrm{NB}(p_t) = \\frac{\\mathrm{TP}(p_t)}{N}
    - \\frac{\\mathrm{FP}(p_t)}{N} \\cdot \\frac{p_t}{1 - p_t}

The threshold encodes the decision-maker's exchange rate between a missed
opportunity and a turnover, so the whole curve is reported rather than a single
operating point.

Football reading of the threshold in this study
-----------------------------------------------
``p_t`` is the control probability at which a player should attempt a pass into
a contested area rather than a safe one. A coach who treats a turnover in
midfield as roughly twice as costly as a foregone progression is operating near
``p_t = 2/3``. Decision-curve analysis is the right frame precisely because
**miscalibration relocates the decision threshold**: if a model's 0.7 is really
a 0.55, every threshold-based recommendation it makes is systematically too
aggressive, and that is a football error, not merely a scoring-rule error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def net_benefit(y_true, y_prob, thresholds=None, sample_weight=None) -> pd.DataFrame:
    """Net benefit of the model, and of treat-all/treat-none, across thresholds."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    n = w.sum()
    prevalence = float(np.average(y, weights=w))
    rows = []
    for pt in np.asarray(thresholds, dtype=float):
        act = p >= pt
        tp = float(w[act & (y == 1)].sum())
        fp = float(w[act & (y == 0)].sum())
        odds = pt / max(1.0 - pt, 1e-12)
        rows.append(
            {
                "threshold": float(pt),
                "net_benefit_model": tp / n - (fp / n) * odds,
                "net_benefit_all": prevalence - (1 - prevalence) * odds,
                "net_benefit_none": 0.0,
                "frac_acted": float(w[act].sum() / n),
            }
        )
    out = pd.DataFrame(rows)
    out["net_benefit_best_default"] = out[["net_benefit_all", "net_benefit_none"]].max(axis=1)
    out["delta_vs_default"] = out["net_benefit_model"] - out["net_benefit_best_default"]
    return out


def standardised_net_benefit(y_true, y_prob, thresholds=None, sample_weight=None) -> pd.DataFrame:
    """Net benefit rescaled by prevalence, comparable across subgroups.

    Necessary for the subgroup analysis: the final third and the defensive
    third have very different control base rates, so raw net benefit is not
    comparable between them.
    """
    nb = net_benefit(y_true, y_prob, thresholds, sample_weight)
    y = np.asarray(y_true, dtype=float).ravel()
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
    prevalence = float(np.average(y, weights=w))
    for col in ("net_benefit_model", "net_benefit_all", "delta_vs_default"):
        nb[f"std_{col}"] = nb[col] / max(prevalence, 1e-12)
    return nb


def expected_cost(y_true, y_prob, *, threshold: float, cost_fp: float, cost_fn: float,
                  sample_weight=None) -> float:
    """Expected cost of a threshold policy under an explicit cost ratio.

    Provided so the study can state a football-meaningful loss directly - for
    instance, costing a turnover in the middle third by the opponent's
    resulting expected threat - rather than relying on the implicit costs
    encoded by a threshold.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
    act = p >= threshold
    fp = float(w[act & (y == 0)].sum())
    fn = float(w[~act & (y == 1)].sum())
    return (cost_fp * fp + cost_fn * fn) / w.sum()


def threshold_displacement(y_true, y_prob_raw, y_prob_cal, thresholds=None, sample_weight=None) -> pd.DataFrame:
    """How many decisions change when a model is recalibrated.

    For each threshold, the fraction of arrivals on which the raw and
    recalibrated forecasts fall on opposite sides. This is the study's most
    direct translation of "the number was miscalibrated" into "a coach would
    have decided differently", and it is reported for the thresholds a
    practitioner would plausibly use rather than averaged over all of them.
    """
    raw = np.asarray(y_prob_raw, dtype=float).ravel()
    cal = np.asarray(y_prob_cal, dtype=float).ravel()
    y = np.asarray(y_true, dtype=float).ravel()
    w = np.ones_like(raw) if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
    if thresholds is None:
        thresholds = np.array([0.4, 0.5, 0.6, 0.7, 0.8])

    rows = []
    for pt in np.asarray(thresholds, dtype=float):
        a, b = raw >= pt, cal >= pt
        flipped = a != b
        rows.append(
            {
                "threshold": float(pt),
                "frac_decisions_changed": float(w[flipped].sum() / w.sum()),
                "raw_act_rate": float(w[a].sum() / w.sum()),
                "cal_act_rate": float(w[b].sum() / w.sum()),
                "accuracy_raw": float(w[(a == (y == 1))].sum() / w.sum()),
                "accuracy_cal": float(w[(b == (y == 1))].sum() / w.sum()),
            }
        )
    return pd.DataFrame(rows)
