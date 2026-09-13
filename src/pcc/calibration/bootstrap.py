"""Uncertainty quantification under clustered dependence.

Arrivals are not independent. Two passes from the same possession share a
defensive shape; two possessions from the same match share a team, a referee, a
pitch and a tactical plan. Treating ~40,000 arrivals as 40,000 independent
observations would shrink every confidence interval by roughly the square root
of the design effect and would make trivial differences between models look
decisive. That mistake is common enough in applied football analytics that
avoiding it is part of this study's methodological contribution.

The default inferential tool is the **cluster (block) bootstrap resampling
whole matches with replacement**. Matches, not possessions, are the unit:
possession-level resampling would still treat within-match correlation as
independent noise. Where a statistic is compared *between* two models on the
same data, the bootstrap is paired - both models are recomputed on the same
resampled matches - so the interval is for the difference and absorbs the
shared match-level variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lo: float
    hi: float
    se: float
    n_boot: int
    n_clusters: int
    level: float

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}point": self.point,
            f"{prefix}lo": self.lo,
            f"{prefix}hi": self.hi,
            f"{prefix}se": self.se,
            f"{prefix}n_boot": float(self.n_boot),
            f"{prefix}n_clusters": float(self.n_clusters),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


def cluster_bootstrap(
    statistic: Callable[[pd.DataFrame], float],
    df: pd.DataFrame,
    *,
    cluster_col: str = "match_id",
    n_boot: int = 1000,
    level: float = 0.95,
    random_state: int = 0,
    tag_replicates: bool = False,
) -> BootstrapResult:
    """Percentile cluster bootstrap of an arbitrary statistic.

    Parameters
    ----------
    statistic
        Callable mapping a resampled dataframe to a scalar. It must be
        computable on a resample in which the same match appears several times.
        Statistics that need to tell repeated draws apart should be run with
        ``tag_replicates=True``, which adds a ``_boot_rep`` column at a
        noticeable cost; none of the statistics used in this study need it.
    cluster_col
        Column holding the resampling unit. Use ``match_id``.
    n_boot
        1000 replicates is adequate for a percentile interval at the 95% level;
        the study reports the replicate count with every interval.
    tag_replicates
        Off by default. Building the tag requires a per-cluster copy on every
        replicate, which dominates the runtime; without it a replicate is a
        single positional take.

    Notes
    -----
    The percentile interval is used rather than BCa. BCa's acceleration
    estimate needs a cluster jackknife, which with 60-100 matches is itself
    noisy; the percentile interval's coverage error is second-order and is
    dominated here by the small number of clusters. With fewer than about 30
    matches - the situation for Metrica and SkillCorner - the interval should
    be read as indicative and the study says so rather than reporting a number
    with false precision.
    """
    if cluster_col not in df.columns:
        raise KeyError(f"cluster column {cluster_col!r} not in dataframe")

    rng = np.random.default_rng(random_state)
    clusters = df[cluster_col].to_numpy()
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}

    point = float(statistic(df))
    replicates = np.empty(n_boot, dtype=float)
    n_fail = 0
    for b in range(n_boot):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        positions = np.concatenate([index_by_cluster[c] for c in drawn])
        resample = df.take(positions)
        if tag_replicates:
            tags = np.concatenate(
                [np.full(index_by_cluster[c].size, rep) for rep, c in enumerate(drawn)]
            )
            resample = resample.assign(_boot_rep=tags)
        try:
            replicates[b] = float(statistic(resample))
        except Exception:
            replicates[b] = np.nan
            n_fail += 1

    good = replicates[np.isfinite(replicates)]
    if good.size < max(20, n_boot // 10):
        return BootstrapResult(point, float("nan"), float("nan"), float("nan"), int(good.size), unique.size, level)

    alpha = (1.0 - level) / 2.0
    return BootstrapResult(
        point=point,
        lo=float(np.quantile(good, alpha)),
        hi=float(np.quantile(good, 1.0 - alpha)),
        se=float(np.std(good, ddof=1)),
        n_boot=int(good.size),
        n_clusters=int(unique.size),
        level=level,
    )


def paired_cluster_bootstrap(
    statistic: Callable[[pd.DataFrame], float],
    df: pd.DataFrame,
    *,
    col_a: str,
    col_b: str,
    prob_col: str = "p",
    cluster_col: str = "match_id",
    n_boot: int = 1000,
    level: float = 0.95,
    random_state: int = 0,
) -> BootstrapResult:
    """Interval for ``statistic(model A) - statistic(model B)`` on shared matches.

    Both models' forecasts must already be columns of ``df``. Each replicate
    recomputes both statistics on the *same* resampled matches, so the interval
    is for the paired difference. This is the correct test for H2 (physics
    model versus logistic baseline): the models see identical situations, and
    an unpaired comparison would waste that and inflate the interval.
    """

    def diff(sub: pd.DataFrame) -> float:
        a = statistic(sub.assign(**{prob_col: sub[col_a]}))
        b = statistic(sub.assign(**{prob_col: sub[col_b]}))
        return float(a - b)

    return cluster_bootstrap(
        diff, df, cluster_col=cluster_col, n_boot=n_boot, level=level, random_state=random_state
    )


def design_effect(df: pd.DataFrame, *, value_col: str, cluster_col: str = "match_id") -> dict[str, float]:
    """Intra-cluster correlation and the resulting variance inflation.

    .. math::
        \\mathrm{ICC} = \\frac{\\sigma^2_{between}}
        {\\sigma^2_{between} + \\sigma^2_{within}}, \\qquad
        \\mathrm{DEFF} = 1 + (\\bar m - 1)\\,\\mathrm{ICC}

    ``DEFF`` is the factor by which naive independent-observation standard
    errors are too small. Reporting it makes the clustering correction
    auditable and gives a one-number answer to "how much would ignoring this
    have mattered?" - which the study should state explicitly, since much
    published football analytics does ignore it.
    """
    grouped = df.groupby(cluster_col)[value_col]
    means = grouped.mean()
    sizes = grouped.size()
    grand = float(df[value_col].mean())

    between = float(np.average((means - grand) ** 2, weights=sizes))
    within = float(df.groupby(cluster_col)[value_col].transform("mean").sub(df[value_col]).pow(2).mean())
    icc = between / (between + within) if (between + within) > 0 else float("nan")
    mbar = float(sizes.mean())
    return {
        "icc": icc,
        "mean_cluster_size": mbar,
        "design_effect": 1.0 + (mbar - 1.0) * icc if np.isfinite(icc) else float("nan"),
        "n_clusters": float(sizes.size),
    }


def bootstrap_reliability_band(
    df: pd.DataFrame,
    *,
    prob_col: str,
    outcome_col: str,
    cluster_col: str = "match_id",
    n_bins: int = 10,
    n_boot: int = 500,
    level: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Cluster-bootstrap confidence band for a binned reliability curve.

    Bin edges are held fixed at the edges computed on the full sample, so that
    replicates are comparable bin by bin; re-deriving quantile edges inside
    each replicate would mix variation in the curve with variation in the
    binning and produce an uninterpretable band.
    """
    from pcc.calibration.reliability import binned_reliability

    p_all = df[prob_col].to_numpy()
    edges = np.unique(np.quantile(p_all, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        edges = np.linspace(0.0, 1.0, 3)
    edges[0], edges[-1] = 0.0, 1.0

    def curve(sub: pd.DataFrame) -> np.ndarray:
        p = sub[prob_col].to_numpy()
        y = sub[outcome_col].to_numpy()
        idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, edges.size - 2)
        out = np.full(edges.size - 1, np.nan)
        for b in range(edges.size - 1):
            m = idx == b
            if m.any():
                out[b] = y[m].mean()
        return out

    rng = np.random.default_rng(random_state)
    clusters = df[cluster_col].to_numpy()
    unique = np.unique(clusters)
    index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}

    reps = np.full((n_boot, edges.size - 1), np.nan)
    for b in range(n_boot):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        idx = np.concatenate([index_by_cluster[c] for c in drawn])
        reps[b] = curve(df.iloc[idx])

    alpha = (1 - level) / 2
    base = binned_reliability(
        df[outcome_col].to_numpy(), df[prob_col].to_numpy(), n_bins=n_bins, strategy="quantile"
    )
    with np.errstate(invalid="ignore"):
        lo = np.nanquantile(reps, alpha, axis=0)
        hi = np.nanquantile(reps, 1 - alpha, axis=0)
    if not base.empty:
        bins = base["bin"].to_numpy()
        base = base.assign(boot_lo=lo[bins], boot_hi=hi[bins], n_boot=n_boot)
    return base
