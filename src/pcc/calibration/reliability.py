"""Reliability curves: binned and binning-free (CORP).

A reliability diagram plots observed event frequency against forecast
probability. The published pitch-control literature rarely shows one; producing
them for the standard models is the study's most immediately legible output.

Two constructions are supplied:

* :func:`binned_reliability` - the familiar version, with equal-width or
  equal-mass bins. Its shape is an artefact of the bin count as much as of the
  forecast, so bin counts are always reported and a sensitivity sweep is run.
* :func:`corp_reliability` - the isotonic (PAV) reliability curve. It requires
  no bin choice, is a consistent estimator of the conditional event frequency
  under monotonicity, and is what the study uses for its headline figure.

Both return tidy frames so the same object feeds plots, tables and the
cluster bootstrap without reshaping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcc.calibration.metrics import _prep, isotonic_recalibrate


def binned_reliability(
    y_true,
    y_prob,
    sample_weight=None,
    *,
    n_bins: int = 15,
    strategy: str = "quantile",
    min_count: int = 1,
) -> pd.DataFrame:
    """Binned reliability table.

    Returns columns ``lower, upper, n, weight, mean_pred, obs_freq, gap,
    se_wilson_lo, se_wilson_hi``.

    The per-bin interval is a Wilson score interval on the *unweighted* count,
    which understates uncertainty because arrivals within a bin are clustered
    by match. It is shown as a within-bin guide only; all inferential
    statements in the study come from the cluster bootstrap in
    :mod:`pcc.calibration.bootstrap`.
    """
    y, p, w = _prep(y_true, y_prob, sample_weight)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 3:
            edges = np.linspace(0.0, 1.0, 3)
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        m = idx == b
        if m.sum() < min_count:
            continue
        k = float(y[m].sum())
        n = float(m.sum())
        lo, hi = _wilson(k, n)
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": int(n),
                "weight": float(w[m].sum()),
                "mean_pred": float(np.average(p[m], weights=w[m])),
                "obs_freq": float(np.average(y[m], weights=w[m])),
                "wilson_lo": lo,
                "wilson_hi": hi,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["gap"] = out["obs_freq"] - out["mean_pred"]
    return out


def _wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def corp_reliability(y_true, y_prob, sample_weight=None) -> pd.DataFrame:
    """Binning-free reliability curve from the pool-adjacent-violators fit.

    Returns one row per distinct forecast level with the isotonic conditional
    event frequency and the number of observations supporting it. Plotting
    ``obs_freq`` against ``mean_pred`` gives a monotone step function whose
    departure from the diagonal is exactly the miscalibration charged by the
    MCB term of :func:`~pcc.calibration.metrics.corp_decomposition`.
    """
    y, p, w = _prep(y_true, y_prob, sample_weight)
    fitted = isotonic_recalibrate(y, p, w)
    df = pd.DataFrame({"mean_pred": p, "obs_freq": fitted, "y": y, "weight": w})
    grouped = (
        df.groupby("obs_freq", as_index=False)
        .agg(
            mean_pred=("mean_pred", "mean"),
            pred_min=("mean_pred", "min"),
            pred_max=("mean_pred", "max"),
            n=("y", "size"),
            weight=("weight", "sum"),
            emp_freq=("y", "mean"),
        )
        .sort_values("mean_pred")
        .reset_index(drop=True)
    )
    grouped["gap"] = grouped["obs_freq"] - grouped["mean_pred"]
    return grouped


def reliability_by_group(
    df: pd.DataFrame,
    *,
    prob_col: str,
    outcome_col: str,
    group_col: str,
    weight_col: str | None = None,
    n_bins: int = 10,
    min_group_n: int = 200,
) -> pd.DataFrame:
    """Stacked reliability tables for every level of ``group_col``.

    Used for the subgroup analysis (RQ3): pitch zone, pass-length band,
    pressure band, game state, tracking source. ``min_group_n`` suppresses
    strata too small to say anything about; the suppressed strata are still
    listed with their counts so that the reader can see what was dropped rather
    than inferring it from a gap in a figure.
    """
    out = []
    for level, sub in df.groupby(group_col, observed=True):
        if len(sub) < min_group_n:
            out.append(pd.DataFrame([{group_col: level, "n": len(sub), "suppressed": True}]))
            continue
        tab = binned_reliability(
            sub[outcome_col].to_numpy(),
            sub[prob_col].to_numpy(),
            None if weight_col is None else sub[weight_col].to_numpy(),
            n_bins=n_bins,
            strategy="quantile",
        )
        tab[group_col] = level
        tab["suppressed"] = False
        out.append(tab)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
