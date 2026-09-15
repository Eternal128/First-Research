"""Subgroup construction and stratified calibration analysis (RQ3).

Subgroup definitions are fixed here, *before* any results are seen, and the
bands are chosen on substantive football grounds rather than on the data. A
subgroup analysis whose cut-points were chosen after inspecting the outcome is
not a test, and with six subgrouping variables the multiplicity problem is real
- so the module also carries the multiplicity adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcc.geometry import Pitch, zone_index, zone_label

#: Pre-registered pass-length bands (metres). Short passes are dominated by the
#: passer's immediate options; long passes give defenders time to converge, so
#: the two are expected a priori to have different calibration.
PASS_LENGTH_BANDS = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 200)]

#: Opponents within 5 m of the release point.
PRESSURE_BANDS = [(0, 1), (1, 2), (2, 3), (3, 99)]

#: Score difference from the perspective of the team in possession.
GAME_STATES = {"trailing": (-99, -1), "level": (0, 0), "leading": (1, 99)}


def _band(values: np.ndarray, bands, labels=None) -> pd.Categorical:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, "unknown", dtype=object)
    names = labels or [f"[{lo},{hi})" for lo, hi in bands]
    for (lo, hi), name in zip(bands, names):
        out[(values >= lo) & (values < hi)] = name
    return pd.Categorical(out, categories=list(names) + ["unknown"])


def add_subgroups(df: pd.DataFrame, *, pitch: Pitch = Pitch()) -> pd.DataFrame:
    """Attach every pre-registered subgroup column to an arrivals table."""
    out = df.copy()
    xy = out[["dest_x", "dest_y"]].to_numpy()
    zi = zone_index(xy, pitch=pitch, n_x=5, n_y=3)
    out["zone_index"] = zi
    out["zone"] = [zone_label(int(i), n_x=5, n_y=3) for i in zi]
    out["third"] = pd.cut(
        out["dest_x"], bins=[-np.inf, -pitch.half_length / 3, pitch.half_length / 3, np.inf],
        labels=["defensive_third", "middle_third", "attacking_third"],
    )
    out["pass_length_band"] = _band(
        out["pass_length"], PASS_LENGTH_BANDS,
        ["0-10m", "10-20m", "20-30m", "30-45m", "45m+"],
    )
    if "pressure_index" in out.columns:
        out["pressure_band"] = _band(
            out["pressure_index"], PRESSURE_BANDS, ["0", "1", "2", "3+"]
        )
    else:
        out["pressure_band"] = "unknown"
    if "score_diff" in out.columns:
        out["game_state"] = np.select(
            [out["score_diff"] < 0, out["score_diff"] == 0, out["score_diff"] > 0],
            ["trailing", "level", "leading"],
            default="unknown",
        )
    else:
        out["game_state"] = "unknown"
    # Tracking completeness. The primary within-corpus axis for RQ4: comparing
    # well-observed against poorly-observed arrivals inside one corpus holds the
    # competition, the provider and the season fixed, which the across-corpus
    # optical-versus-broadcast contrast cannot do.
    if "frame_completeness" in out.columns:
        out["completeness_band"] = _band(
            out["frame_completeness"], [(0.0, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)],
            ["<60%", "60-75%", "75-90%", "90%+"],
        )
    else:
        out["completeness_band"] = "unknown"

    out["flight_band"] = _band(
        out["flight_time"], [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 99)],
        ["<0.5s", "0.5-1s", "1-1.5s", "1.5s+"],
    )
    return out


def stratified_metrics(
    df: pd.DataFrame,
    *,
    prob_col: str,
    outcome_col: str,
    group_col: str,
    cluster_col: str = "match_id",
    weight_col: str | None = None,
    min_n: int = 200,
    n_boot: int = 400,
    random_state: int = 0,
) -> pd.DataFrame:
    """Per-stratum calibration metrics with cluster-bootstrap intervals.

    Strata below ``min_n`` are reported with their counts and ``suppressed =
    True`` rather than dropped, so a reader can see the whole partition. This
    matters for the spatial analysis, where the sparsely targeted zones (deep
    in the defensive corner, inside the opposition six-yard box) are precisely
    the ones where a physics model is least constrained by data and most likely
    to be wrong.
    """
    from pcc.calibration.bootstrap import cluster_bootstrap
    from pcc.calibration.metrics import brier_score, calibration_slope_intercept, corp_decomposition

    rows = []
    for level, sub in df.groupby(group_col, observed=True):
        n = len(sub)
        if n < min_n or sub[outcome_col].nunique() < 2:
            rows.append({group_col: level, "n": n, "suppressed": True})
            continue
        y = sub[outcome_col].to_numpy()
        p = sub[prob_col].to_numpy()
        w = sub[weight_col].to_numpy() if weight_col else None
        corp = corp_decomposition(y, p, w)
        slope = calibration_slope_intercept(y, p, w)

        work = pd.DataFrame({"y": y, "p": p, cluster_col: sub[cluster_col].to_numpy()})
        n_clusters = work[cluster_col].nunique()
        boot = (
            cluster_bootstrap(
                lambda d: corp_decomposition(d["y"], d["p"]).miscalibration,
                work, cluster_col=cluster_col, n_boot=n_boot, random_state=random_state,
            )
            if n_clusters >= 5
            else None
        )
        rows.append(
            {
                group_col: level,
                "n": n,
                "n_clusters": n_clusters,
                "suppressed": False,
                "base_rate": float(np.mean(y)),
                "mean_forecast": float(np.mean(p)),
                "brier": brier_score(y, p, w),
                "corp_mcb": corp.miscalibration,
                "corp_dsc": corp.discrimination,
                "corp_unc": corp.uncertainty,
                "calibration_slope": slope["slope"],
                "calibration_in_the_large": slope["intercept_in_the_large"],
                "mcb_lo": boot.lo if boot else np.nan,
                "mcb_hi": boot.hi if boot else np.nan,
            }
        )
    return pd.DataFrame(rows)


def adjust_multiplicity(df: pd.DataFrame, *, pvalue_col: str, method: str = "fdr_bh") -> pd.DataFrame:
    """Benjamini-Hochberg (or Bonferroni) adjustment across strata.

    With six subgrouping variables and up to fifteen pitch zones, some stratum
    will show a nominally significant calibration gap by chance. The study
    controls the false discovery rate across the full pre-registered subgroup
    family and reports both raw and adjusted values, so that an unadjusted
    reader can still see what was found.
    """
    out = df.copy()
    p = out[pvalue_col].to_numpy(dtype=float)
    finite = np.isfinite(p)
    adj = np.full_like(p, np.nan)
    pf = p[finite]
    m = pf.size
    if m == 0:
        out[f"{pvalue_col}_adj"] = adj
        return out

    if method == "bonferroni":
        adj[finite] = np.minimum(pf * m, 1.0)
    elif method == "fdr_bh":
        order = np.argsort(pf)
        ranked = pf[order] * m / (np.arange(m) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        tmp = np.empty(m)
        tmp[order] = np.minimum(ranked, 1.0)
        adj[finite] = tmp
    else:
        raise ValueError(f"unknown method {method!r}")
    out[f"{pvalue_col}_adj"] = adj
    out["multiplicity_method"] = method
    return out


def spatial_calibration_map(
    df: pd.DataFrame,
    *,
    prob_col: str,
    outcome_col: str,
    pitch: Pitch = Pitch(),
    n_x: int = 12,
    n_y: int = 8,
    min_n: int = 50,
) -> pd.DataFrame:
    """Cell-level mean forecast, observed frequency and gap, for the spatial map.

    The deliverable is a map of ``E[Y | cell] - E[C | cell]``. Note carefully
    what it is *not*: it is not the miscalibration of the model at every point
    on the pitch, because it is estimated only where balls actually arrived.
    Empty and sparse cells are returned with ``n`` so the figure can mask them
    instead of interpolating confidence into regions with no data - the visual
    equivalent of the selection problem the study is about.
    """
    x_edges = np.linspace(-pitch.half_length, pitch.half_length, n_x + 1)
    y_edges = np.linspace(-pitch.half_width, pitch.half_width, n_y + 1)
    ix = np.clip(np.digitize(df["dest_x"].to_numpy(), x_edges[1:-1]), 0, n_x - 1)
    iy = np.clip(np.digitize(df["dest_y"].to_numpy(), y_edges[1:-1]), 0, n_y - 1)

    work = df.assign(_ix=ix, _iy=iy)
    agg = (
        work.groupby(["_ix", "_iy"], observed=True)
        .agg(n=(outcome_col, "size"), obs_freq=(outcome_col, "mean"), mean_pred=(prob_col, "mean"))
        .reset_index()
    )
    agg["gap"] = agg["obs_freq"] - agg["mean_pred"]
    agg["x_centre"] = (x_edges[agg["_ix"]] + x_edges[agg["_ix"] + 1]) / 2
    agg["y_centre"] = (y_edges[agg["_iy"]] + y_edges[agg["_iy"] + 1]) / 2
    agg["sufficient"] = agg["n"] >= min_n
    return agg
