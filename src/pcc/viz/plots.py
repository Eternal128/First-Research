"""Figures. Matplotlib only, so the study has no plotting dependency it cannot run.

Every function returns the ``Figure`` rather than showing or saving it, so the
scripts control output paths and the tests can call them headlessly.

Design rule for this study's figures: **a reliability diagram must always show
the forecast distribution**. A calibration curve without a histogram of where
the forecasts actually live invites the reader to weigh a badly-calibrated
region containing 1% of the data equally with a well-calibrated region
containing 60% of it. That omission is common and this module makes it
impossible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pcc.geometry import Pitch


def _mpl():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def reliability_diagram(
    y_true,
    forecasts: dict[str, np.ndarray],
    *,
    n_bins: int = 10,
    title: str = "Reliability",
    show_corp: bool = True,
    figsize: tuple[float, float] = (6.0, 7.0),
):
    """Reliability curves for several models, with the forecast histogram below.

    ``forecasts`` maps a model label to its probability vector. The CORP
    (isotonic) curve is overlaid as a dashed line when ``show_corp`` is set,
    which lets the reader see how much of the binned curve's shape is a
    binning artefact.
    """
    from pcc.calibration.reliability import binned_reliability, corp_reliability

    plt = _mpl()
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot([0, 1], [0, 1], color="0.4", lw=1, ls=":", label="perfect calibration", zorder=1)

    for label, p in forecasts.items():
        p = np.asarray(p, dtype=float)
        tab = binned_reliability(y_true, p, n_bins=n_bins, strategy="quantile")
        if not tab.empty:
            line, = ax.plot(tab["mean_pred"], tab["obs_freq"], marker="o", ms=4, lw=1.5, label=label, zorder=3)
            ax.vlines(tab["mean_pred"], tab["wilson_lo"], tab["wilson_hi"],
                      color=line.get_color(), alpha=0.45, lw=1, zorder=2)
            if show_corp:
                corp = corp_reliability(y_true, p)
                ax.plot(corp["mean_pred"], corp["obs_freq"], lw=1, ls="--",
                        color=line.get_color(), alpha=0.7, zorder=2)
        axh.hist(p, bins=np.linspace(0, 1, 41), histtype="step", lw=1.2, label=label)

    ax.set_ylabel("observed frequency of control")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    axh.set_xlabel("forecast control probability $C_A$")
    axh.set_ylabel("count")
    axh.set_yscale("log")
    axh.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def calibration_gap_by_group(
    df: pd.DataFrame, *, group_col: str, value_col: str = "corp_mcb",
    lo_col: str = "mcb_lo", hi_col: str = "mcb_hi", title: str | None = None,
    figsize: tuple[float, float] = (7.0, 5.0),
):
    """Forest plot of a per-stratum calibration statistic with intervals."""
    plt = _mpl()
    work = df[~df.get("suppressed", pd.Series(False, index=df.index)).astype(bool)].copy()
    work = work.sort_values(value_col)
    y = np.arange(len(work))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color="0.4", lw=1, ls=":")
    if lo_col in work and hi_col in work:
        ax.hlines(y, work[lo_col], work[hi_col], color="C0", lw=2, alpha=0.6)
    ax.plot(work[value_col], y, "o", color="C0", ms=5)
    ax.set_yticks(y)
    ax.set_yticklabels(work[group_col].astype(str), fontsize=8)
    ax.set_xlabel(value_col)
    ax.set_title(title or f"{value_col} by {group_col}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def spatial_gap_map(
    cells: pd.DataFrame,
    *,
    pitch: Pitch = Pitch(),
    value_col: str = "gap",
    mask_insufficient: bool = True,
    vmax: float | None = None,
    title: str = "Observed minus forecast control",
    figsize: tuple[float, float] = (8.0, 5.5),
):
    """Pitch map of ``E[Y | cell] - E[C | cell]``.

    Cells below the sufficiency threshold are left blank rather than shaded.
    That blankness is a finding, not a gap in the figure: it marks the regions
    of the pitch where the ball essentially never arrives and where, therefore,
    a pitch-control model's claims are empirically untestable on observational
    data.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=figsize)
    _draw_pitch(ax, pitch)

    work = cells.copy()
    if mask_insufficient and "sufficient" in work:
        work = work[work["sufficient"]]
    if work.empty:
        ax.set_title(f"{title} (no cell met the sufficiency threshold)")
        return fig

    v = float(vmax if vmax is not None else np.nanpercentile(np.abs(work[value_col]), 95))
    sc = ax.scatter(
        work["x_centre"], work["y_centre"], c=work[value_col], s=260, marker="s",
        cmap="RdBu_r", vmin=-v, vmax=v, edgecolors="none", alpha=0.85, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("observed - forecast")
    ax.set_title(f"{title}\n(blank cells: fewer arrivals than the sufficiency threshold)", fontsize=10)
    fig.tight_layout()
    return fig


def control_field(
    model,
    frame,
    *,
    pitch: Pitch = Pitch(),
    n_x: int = 70,
    n_y: int = 46,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 5.5),
):
    """Render one model's control field for a single arrival frame.

    Included mainly to make a rhetorical point available as a figure: the same
    field that looks authoritative as a heatmap may be badly calibrated. Pairing
    this figure with the reliability diagram for the same model is the paper's
    core visual argument.
    """
    plt = _mpl()
    XX, YY = pitch.grid(n_x, n_y)
    targets = np.column_stack([XX.ravel(), YY.ravel()])
    C = np.asarray(model.control(frame, targets), dtype=float).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(C, extent=pitch.extent, origin="lower", cmap="RdBu_r",
                   vmin=0, vmax=1, alpha=0.85, aspect="equal", zorder=0)
    _draw_pitch(ax, pitch)
    ax.scatter(frame.att_xy[:, 0], frame.att_xy[:, 1], c="crimson", s=45, zorder=4, label="team A")
    ax.scatter(frame.def_xy[:, 0], frame.def_xy[:, 1], c="navy", s=45, zorder=4, label="team B")
    ax.quiver(frame.att_xy[:, 0], frame.att_xy[:, 1], frame.att_v[:, 0], frame.att_v[:, 1],
              color="crimson", alpha=0.6, width=0.003, zorder=4)
    ax.quiver(frame.def_xy[:, 0], frame.def_xy[:, 1], frame.def_v[:, 0], frame.def_v[:, 1],
              color="navy", alpha=0.6, width=0.003, zorder=4)
    ax.scatter(*frame.target, marker="X", s=140, c="k", zorder=5, label="ball destination")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("$C_A$")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_title(title or f"{getattr(model, 'name', type(model).__name__)} control field")
    fig.tight_layout()
    return fig


def decision_curve(nb: pd.DataFrame, *, title: str = "Decision curve", figsize=(6.5, 4.5)):
    """Net benefit against threshold probability, with both default policies."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(nb["threshold"], nb["net_benefit_model"], lw=1.8, label="model")
    ax.plot(nb["threshold"], nb["net_benefit_all"], lw=1.2, ls="--", color="0.4", label="always act")
    ax.plot(nb["threshold"], nb["net_benefit_none"], lw=1.2, ls=":", color="0.4", label="never act")
    lo = float(np.nanpercentile(nb["net_benefit_model"], 5))
    ax.set_ylim(min(lo, 0) - 0.02, float(nb["net_benefit_model"].max()) + 0.02)
    ax.set_xlabel("threshold control probability $p_t$")
    ax.set_ylabel("net benefit")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _draw_pitch(ax, pitch: Pitch) -> None:
    """Minimal pitch furniture; ``mplsoccer`` is optional and not required."""
    import matplotlib.patches as patches

    hl, hw = pitch.half_length, pitch.half_width
    ax.add_patch(patches.Rectangle((-hl, -hw), pitch.length, pitch.width,
                                   fill=False, color="0.3", lw=1.2, zorder=2))
    ax.plot([0, 0], [-hw, hw], color="0.3", lw=1.2, zorder=2)
    ax.add_patch(patches.Circle((0, 0), 9.15, fill=False, color="0.3", lw=1.2, zorder=2))
    for sign in (-1, 1):
        ax.add_patch(patches.Rectangle((sign * hl - sign * 16.5, -20.16), sign * 16.5, 40.32,
                                       fill=False, color="0.3", lw=1.2, zorder=2))
        ax.add_patch(patches.Rectangle((sign * hl - sign * 5.5, -9.16), sign * 5.5, 18.32,
                                       fill=False, color="0.3", lw=1.0, zorder=2))
    ax.set_xlim(-hl - 2, hl + 2)
    ax.set_ylim(-hw - 2, hw + 2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
