#!/usr/bin/env python3
"""Build every figure in the LaTeX paper, driven from results/.

Figures are generated from the result tables rather than hand-drawn, so a
re-run of the analysis updates the paper's pictures along with its numbers.
Output is vector PDF into ``paper/figures/``.

Design notes (following the project's visualisation guidance):
  * Diverging blue-red with a neutral grey midpoint wherever the quantity has a
    *polarity* (which team controls the space; observed minus forecast).
  * A single-hue blue ramp, light to dark, wherever it has only *magnitude*.
  * Categorical series use a fixed, colour-vision-deficiency-validated order
    (blue, orange, aqua) and every series is directly labelled, so identity is
    never carried by colour alone.
  * No chart uses two y-axes. Quantities on different scales get separate panels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, banner

sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

RESULTS = REPO_ROOT / "results"
FIGDIR = REPO_ROOT / "paper" / "figures"
SOURCE = "statsbomb_open"

# --- palette (light surface) ------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GREY = "#e34948", "#f0efec"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE = "#ffffff"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": INK3,
        "axes.linewidth": 0.7,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    FIGDIR.mkdir(parents=True, exist_ok=True)


def save(fig, name, tight=True):
    """Write PDF (for LaTeX) and PNG (for quick inspection).

    ``tight=False`` keeps a hand-tuned ``subplots_adjust`` layout; the bounding
    box is still cropped to the drawn content, it is simply not recomputed.
    """
    path = FIGDIR / f"{name}.pdf"
    kw = dict(bbox_inches="tight", pad_inches=0.02) if tight else dict(pad_inches=0.02)
    fig.savefig(path, **kw)
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, **kw)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def table(rel):
    return pd.read_csv(RESULTS / rel)


def draw_pitch(ax, length=105.0, width=68.0, lc=INK3, lw=0.8):
    hl, hw = length / 2, width / 2
    ax.add_patch(Rectangle((-hl, -hw), length, width, fill=False, color=lc, lw=lw, zorder=3))
    ax.plot([0, 0], [-hw, hw], color=lc, lw=lw, zorder=3)
    ax.add_patch(Circle((0, 0), 9.15, fill=False, color=lc, lw=lw, zorder=3))
    for s in (-1, 1):
        ax.add_patch(Rectangle((s * hl - s * 16.5, -20.16), s * 16.5, 40.32,
                               fill=False, color=lc, lw=lw, zorder=3))
        ax.add_patch(Rectangle((s * hl - s * 5.5, -9.16), s * 5.5, 18.32,
                               fill=False, color=lc, lw=lw * 0.8, zorder=3))
    ax.set_xlim(-hl - 2, hl + 2); ax.set_ylim(-hw - 2, hw + 2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------
def fig01_what_is_pitch_control():
    """What the model outputs, and what 'calibrated' would mean."""
    from pcc.data import load_source
    from pcc.models import PhysicalControl
    from pcc.geometry import Pitch

    root = REPO_ROOT / "data" / "raw" / "metrica_sample"
    frames, _df = load_source("metrica_sample", root=str(root))
    pitch = Pitch()

    # Pick a frame that reads clearly: both teams on the pitch, well spread, the
    # ball heading somewhere central rather than into a corner.
    def legibility(f):
        if f.n_att < 10 or f.n_def < 10:
            return -1e9
        tx, ty = f.target
        if abs(tx) > 34 or abs(ty) > 22:
            return -1e9
        spread = float(np.std(np.vstack([f.att_xy, f.def_xy])[:, 0]))
        origin = np.asarray(f.meta.get("origin", f.target), dtype=float)
        travel = float(np.linalg.norm(f.target - origin))
        return spread + 0.8 * travel
    frame = max(frames[:1200], key=legibility)
    model = PhysicalControl()

    XX, YY = pitch.grid(140, 92)
    field = np.asarray(model.control(frame, np.column_stack([XX.ravel(), YY.ravel()]))).reshape(XX.shape)

    fig = plt.figure(figsize=(6.9, 3.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.16)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(field, extent=pitch.extent, origin="lower", cmap="RdBu_r",
                   vmin=0, vmax=1, alpha=0.92, aspect="equal", zorder=0)
    draw_pitch(ax, lc="#ffffff", lw=1.0)
    origin = np.asarray(frame.meta.get("origin", frame.target), dtype=float)
    ax.annotate("", xy=tuple(frame.target), xytext=tuple(origin),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3,
                                shrinkA=4, shrinkB=6), zorder=6)
    ax.scatter(frame.att_xy[:, 0], frame.att_xy[:, 1], s=30, c=RED,
               edgecolors="white", linewidths=0.7, zorder=5)
    ax.scatter(frame.def_xy[:, 0], frame.def_xy[:, 1], s=30, c=BLUE,
               edgecolors="white", linewidths=0.7, zorder=5)
    ax.scatter(*frame.target, marker="X", s=95, c=INK, edgecolors="white",
               linewidths=0.9, zorder=7)
    ax.text(0, 39.5, "pass played $\\rightarrow$ ball lands on the $\\times$",
            ha="center", fontsize=8, color=INK)
    ax.text(-52.5, -40.5, "red attacks right   $\\bullet$ red players   $\\bullet$ blue players",
            fontsize=7.2, color=INK2, ha="left")
    ax.set_ylim(-46, 44)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("model's claim: chance red controls it", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.set_title("(a) What a pitch-control model outputs", loc="left", color=INK, pad=14)

    ax2 = fig.add_subplot(gs[0, 1])
    bins = [0.9, 0.7, 0.5, 0.3, 0.1]
    n = 10
    for i, b in enumerate(bins):
        y = len(bins) - 1 - i
        kept = int(round(b * n))
        cols = [AQUA] * kept + [ORANGE] * (n - kept)
        ax2.scatter(np.linspace(0.34, 0.86, n), np.full(n, y), s=34, c=cols,
                    edgecolors="white", linewidths=0.5, zorder=3)
        ax2.text(0.29, y, f"{b:.1f}", ha="right", va="center", fontsize=9, color=INK)
        ax2.text(0.92, y, f"{kept} of {n}", ha="left", va="center", fontsize=8.5, color=INK2)
    ax2.text(0.29, len(bins) - 0.42, "model\nsays", ha="right", va="bottom",
             fontsize=7.5, color=INK2, linespacing=1.2)
    ax2.text(0.92, len(bins) - 0.42, "should keep\nthe ball", ha="left", va="bottom",
             fontsize=7.5, color=INK2, linespacing=1.2)
    ax2.scatter([], [], s=34, c=AQUA, label="kept the ball")
    ax2.scatter([], [], s=34, c=ORANGE, label="lost it")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.52, 0.10), ncol=2,
               handletextpad=0.3, columnspacing=1.2, frameon=False, fontsize=8.5)
    ax2.set_xlim(0.10, 1.30); ax2.set_ylim(-0.90, len(bins) + 0.15)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_title("(b) What that number would have to mean",
                  loc="left", color=INK, pad=6)

    save(fig, "fig01_concept")


def fig02_calibration_vs_discrimination():
    """Same ranking, same AUC, different probabilities."""
    from pcc.calibration import roc_auc

    preds = table(f"main/{SOURCE}/tables/predictions.csv")
    y = preds["y"].to_numpy()
    raw = preds["M2_physical__raw"].to_numpy()
    cal = preds["M2_physical__isotonic"].to_numpy()
    auc_raw, auc_cal = roc_auc(y, raw), roc_auc(y, cal)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))

    ax = axes[0]
    order = np.argsort(raw)
    sample = order[:: max(1, len(order) // 900)]
    ax.scatter(raw[sample], cal[sample], s=5, c=BLUE, alpha=0.35, edgecolors="none")
    ax.plot([0, 1], [0, 1], color=INK3, lw=0.8, ls=":")
    ax.set_xlabel("original value"); ax.set_ylabel("corrected value")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.5)
    ax.set_title("(a) Correction only rescales", loc="left", color=INK)
    ax.text(0.05, 0.90, "order of passes\nis unchanged", fontsize=7.5, color=INK2)

    ax = axes[1]
    for label, p, colour, marker in [("original", raw, ORANGE, "o"), ("corrected", cal, BLUE, "s")]:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() > 50:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, marker=marker, ms=4, lw=1.6, color=colour, label=label, zorder=3)
    ax.plot([0, 1], [0, 1], color=INK3, lw=0.8, ls=":", label="perfect", zorder=2)
    ax.set_xlabel("what the model says"); ax.set_ylabel("what happened")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.5)
    ax.legend(loc="upper left")
    ax.set_title("(b) But the probabilities change", loc="left", color=INK)
    ax.text(0.42, 0.06,
            f"AUC {auc_raw:.3f} vs {auc_cal:.3f}\n(identical — AUC cannot see this)",
            fontsize=7.5, color=INK2)

    save(fig, "fig02_calibration_vs_discrimination")


def fig03_reliability():
    """The headline: reliability curves with the forecast distribution beneath.

    Note on reading direction. The curves lie ABOVE the diagonal, so the models
    *understate* the chance of keeping the ball. That is not in tension with a
    calibration slope below one: the forecasts are both shifted down and spread
    too widely. The annotation says what the picture shows rather than what a
    slope statistic is conventionally called.
    """
    preds = table(f"main/{SOURCE}/tables/predictions.csv")
    y = preds["y"].to_numpy()

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(4.6, 4.8), sharex=True,
                                  gridspec_kw={"height_ratios": [2.7, 1.0], "hspace": 0.10})
    ax.plot([0, 1], [0, 1], color=INK3, lw=0.9, ls=":", zorder=2, label="a perfectly honest model")

    # Voronoi emits only 0 or 1, so quantile bins collapse; show its two levels.
    p1 = preds["M1_voronoi__raw"].to_numpy()
    lo, hi = p1 < 0.5, p1 >= 0.5
    ax.plot([p1[lo].mean(), p1[hi].mean()], [y[lo].mean(), y[hi].mean()],
            marker="^", ms=6, lw=1.7, color=ORANGE, label="Voronoi (M1)", zorder=4)
    axh.hist(p1, bins=np.linspace(0, 1, 46), histtype="step", lw=1.2, color=ORANGE)

    for label, col, colour, marker in [("Physics (M2)", "M2_physical__raw", BLUE, "o"),
                                       ("Logistic (M3)", "M3_logistic__raw", AQUA, "s")]:
        p = preds[col].to_numpy()
        edges = np.unique(np.quantile(p, np.linspace(0, 1, 13)))
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() > 80:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, marker=marker, ms=4.6, lw=1.8, color=colour, label=label, zorder=4)
        axh.hist(p, bins=np.linspace(0, 1, 46), histtype="step", lw=1.2, color=colour)

    ax.set_ylabel("share that actually kept the ball")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.annotate("when the model says 0.4,\nthe ball is kept 8 times in 10",
                xy=(0.40, 0.795), xytext=(0.09, 0.47), fontsize=7.8, color=INK2,
                linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8,
                                connectionstyle="arc3,rad=0.15"))
    ax.text(0.035, 0.955, "above the line = the model understates the chance",
            fontsize=7.8, color=INK, va="top")
    axh.set_xlabel("what the model says the chance is")
    axh.set_ylabel("passes")
    axh.set_yscale("log"); axh.grid(alpha=0.5)
    axh.text(0.03, axh.get_ylim()[1] * 0.35,
             "Voronoi only ever answers 0 or 1", fontsize=7.2, color=INK2)
    save(fig, "fig03_reliability")


def fig04_score_decomposition():
    """Where each model's score goes, against the do-nothing baseline."""
    from matplotlib.patches import Patch

    m = table(f"main/{SOURCE}/tables/metrics.csv")
    raw = m[m.variant == "raw"].set_index("model")
    names = {"M1_voronoi": "Voronoi", "M2_physical": "Physics", "M2a_reach_sigmoid": "Reachability",
             "M0_marginal": "Base rate", "M3_logistic": "Logistic", "M4_gbm": "Boosted trees"}
    order = ["M1_voronoi", "M2_physical", "M2a_reach_sigmoid", "M0_marginal", "M3_logistic", "M4_gbm"]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.15), gridspec_kw={"width_ratios": [1.55, 1.0]})
    fig.subplots_adjust(left=0.135, right=0.985, wspace=0.10, bottom=0.30, top=0.88)

    ax = axes[0]
    ys = np.arange(len(order))[::-1]
    unc = float(raw.loc[order[0], "corp_unc"])
    for i, k in zip(ys, order):
        mcb = max(float(raw.loc[k, "corp_mcb"]), 0.0)
        dsc = max(float(raw.loc[k, "corp_dsc"]), 0.0)
        ax.barh(i, unc, color=GREY, height=0.52, zorder=2)
        ax.barh(i, -dsc, left=unc, color=AQUA, height=0.52, zorder=3)
        ax.barh(i, mcb, left=unc - dsc, color=ORANGE, height=0.52, zorder=3)
        ax.text(max(unc - dsc + mcb, unc) + 0.008, i, f"{float(raw.loc[k, 'brier']):.3f}",
                va="center", fontsize=7.5, color=INK)
    ax.axvline(unc, color=INK2, lw=0.9, ls="--", zorder=4)
    ax.annotate("score of just guessing the average",
                xy=(unc, -0.30), xytext=(unc + 0.014, -0.55),
                fontsize=7, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_yticks(ys); ax.set_yticklabels([names[k] for k in order])
    ax.set_xlabel("Brier score  (lower is better)")
    ax.set_xlim(0, 0.275); ax.set_ylim(-0.95, len(order) - 0.20)
    ax.grid(axis="x", alpha=0.5)
    ax.set_title("(a) What each model's score is made of", loc="left", color=INK)

    ax = axes[1]
    aucs = [float(raw.loc[k, "auc"]) for k in order]
    ax.barh(ys, aucs, color=[BLUE if k != "M0_marginal" else "#c9c8c3" for k in order],
            height=0.52, zorder=3)
    for i, a in zip(ys, aucs):
        ax.text(a + 0.010, i, f"{a:.3f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(0.5, color=INK2, lw=0.9, ls="--")
    ax.set_yticks(ys); ax.set_yticklabels([])
    ax.set_xlabel("AUC  (higher is better)")
    ax.set_xlim(0.45, 0.97); ax.set_ylim(-0.95, len(order) - 0.20)
    ax.grid(axis="x", alpha=0.5)
    ax.set_title("(b) \u2026and its AUC", loc="left", color=INK)

    fig.legend(handles=[Patch(facecolor=GREY, label="how hard the question is"),
                        Patch(facecolor=AQUA, label="credit for useful information"),
                        Patch(facecolor=ORANGE, label="penalty for wrong probabilities")],
               loc="lower center", ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    save(fig, "fig04_decomposition", tight=False)


def fig05_where_it_breaks():
    """Pitch map of the error, plus the pass-length gradient."""
    sg = table(f"subgroups/{SOURCE}/tables/subgroup_metrics.csv")
    zones = sg[(sg.subgroup_variable == "zone") & (~sg.suppressed.fillna(False))].copy()

    xnames = ["def-fifth", "def-mid", "middle", "att-mid", "att-fifth"]
    ynames = ["right", "centre", "left"]
    grid = np.full((3, 5), np.nan)
    for _, r in zones.iterrows():
        xs, ys = str(r["level"]).split("/")
        grid[ynames.index(ys), xnames.index(xs)] = r["corp_mcb"]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.2), gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.subplots_adjust(wspace=0.30, bottom=0.22, top=0.88)
    ax = axes[0]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ)
    im = ax.imshow(grid, extent=(-52.5, 52.5, -34, 34), origin="lower", cmap=cmap,
                   vmin=0, vmax=np.nanmax(grid), aspect="equal", alpha=0.95, zorder=0)
    draw_pitch(ax, lc="#ffffff", lw=1.0)
    for iy in range(3):
        for ix in range(5):
            if np.isfinite(grid[iy, ix]):
                v = grid[iy, ix]
                ax.text(-52.5 + 21 * ix + 10.5, -34 + 22.67 * iy + 11.3, f"{v:.02f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 * np.nanmax(grid) else INK)
    ax.text(0, -39.5, "attacking direction $\\rightarrow$", ha="center", fontsize=7.5, color=INK2)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.11,
                      aspect=34)
    cb.set_label("size of the error", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    ax.set_title("(a) The error is worst in front of goal", loc="left", color=INK)

    ax = axes[1]
    pl = sg[(sg.subgroup_variable == "pass_length_band") & (~sg.suppressed.fillna(False))]
    labels = ["0-10m", "10-20m", "20-30m", "30-45m", "45m+"]
    pretty = {"0-10m": "0\u201310", "10-20m": "10\u201320", "20-30m": "20\u201330",
              "30-45m": "30\u201345", "45m+": "45+"}
    pl = pl.set_index("level").reindex(labels).dropna(subset=["corp_mcb"])
    xs = np.arange(len(pl))
    ax.bar(xs, pl["corp_mcb"], color=BLUE, width=0.6, zorder=3)
    for x, v in zip(xs, pl["corp_mcb"]):
        ax.text(x, v + 0.004, f"{v:.03f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([pretty[i] for i in pl.index], fontsize=8)
    ax.set_xlabel("how far the pass travelled (metres)")
    ax.set_ylabel("size of the error")
    ax.set_ylim(0, float(pl["corp_mcb"].max()) * 1.22)
    ax.grid(axis="y", alpha=0.5)
    ax.set_title("(b) …and grows with distance", loc="left", color=INK)
    save(fig, "fig05_where_it_breaks", tight=False)


def fig06_observability():
    """More of the pitch seen, better calibrated - but not fixed."""
    sg = table(f"subgroups/{SOURCE}/tables/subgroup_metrics.csv")
    cb = sg[(sg.subgroup_variable == "completeness_band") & (~sg.suppressed.fillna(False))]
    order = ["<60%", "60-75%", "75-90%", "90%+"]
    cb = cb.set_index("level").reindex(order).dropna(subset=["corp_mcb"])
    xs = np.arange(len(cb))

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharex=True)
    ax = axes[0]
    ax.bar(xs, cb["corp_mcb"], color=[SEQ[1], SEQ[2], SEQ[3], SEQ[5]], width=0.6, zorder=3)
    for x, v in zip(xs, cb["corp_mcb"]):
        ax.text(x, v + 0.002, f"{v:.03f}", ha="center", fontsize=7.5, color=INK)
    ax.set_ylabel("size of the error"); ax.grid(axis="y", alpha=0.5)
    ax.set_ylim(0, float(cb["corp_mcb"].max()) * 1.2)
    ax.set_title("(a) Error shrinks as the camera sees more", loc="left", color=INK)

    ax = axes[1]
    ax.bar(xs, cb["calibration_slope"], color=[SEQ[1], SEQ[2], SEQ[3], SEQ[5]], width=0.6, zorder=3)
    ax.axhline(1.0, color=INK2, lw=0.9, ls="--", zorder=4)
    ax.text(len(xs) - 0.55, 1.02, "a calibrated model would sit here",
            fontsize=7, color=INK2, ha="right")
    for x, v in zip(xs, cb["calibration_slope"]):
        ax.text(x, v + 0.02, f"{v:.02f}", ha="center", fontsize=7.5, color=INK)
    ax.set_ylabel("calibration slope (1.0 = honest)")
    ax.set_ylim(0, 1.2); ax.grid(axis="y", alpha=0.5)
    ax.set_title("(b) …but never reaches honest", loc="left", color=INK)
    for a in axes:
        a.set_xticks(xs); a.set_xticklabels(cb.index, fontsize=8)
        a.set_xlabel("share of the 22 players visible")
    save(fig, "fig06_observability")


def fig07_transport_and_fix():
    """It travels, and one correction travels with it."""
    cp = table(f"transportability/{SOURCE}/tables/crossing_penalty.csv").set_index("model")
    tr = table(f"transportability/{SOURCE}/tables/recalibration_transfer.csv")
    loco = table(f"transportability/{SOURCE}/tables/leave_one_competition_out.csv")
    loco = loco[(loco.model == "M2_physical") & (loco.variant == "raw")]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.subplots_adjust(left=0.105, right=0.985, wspace=0.28, bottom=0.40, top=0.90)
    ax = axes[0]
    labels = [str(x).replace("FIFA ", "").replace("Women's ", "Women's ") for x in loco["held_out"]]
    xs = np.arange(len(labels))
    within = float(cp.loc["M2_physical", "mcb_within_competition"])
    ax.bar(xs, loco["corp_mcb"], color=BLUE, width=0.55, zorder=3,
           label="tested on a competition it never saw")
    ax.axhline(within, color=ORANGE, lw=1.4, ls="--", zorder=4,
               label="tested inside the competitions it learned on")
    for x, v in zip(xs, loco["corp_mcb"]):
        ax.text(x, v + 0.002, f"{v:.03f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.2, rotation=20, ha="right")
    ax.set_ylabel("size of the error"); ax.set_ylim(0, 0.115); ax.grid(axis="y", alpha=0.5)
    handles, lbls = ax.get_legend_handles_labels()
    ax.set_title("(a) The error travels between competitions", loc="left", color=INK)

    ax = axes[1]
    t2 = tr[tr.model == "M2_physical"].groupby("map")["mcb_removed_frac"].mean()
    maps = ["platt", "beta", "isotonic"]
    names = ["simple\n(2 numbers)", "flexible\n(3 numbers)", "free-form"]
    ys = np.arange(len(maps))[::-1]
    ax.barh(ys, [t2[m] * 100 for m in maps], color=[SEQ[2], SEQ[3], SEQ[5]], height=0.5, zorder=3)
    for yy, m in zip(ys, maps):
        ax.text(t2[m] * 100 - 2, yy, f"{t2[m]*100:.1f}%", va="center", ha="right",
                fontsize=8, color="white", fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 105); ax.set_xlabel("share of the error removed")
    ax.grid(axis="x", alpha=0.5)
    ax.set_title("(b) …and one correction fixes it", loc="left", color=INK)
    fig.legend(handles, lbls, loc="lower center", bbox_to_anchor=(0.30, 0.005),
               fontsize=7.2, frameon=False, ncol=1, handlelength=1.6)
    save(fig, "fig07_transport", tight=False)


def fig08_downstream():
    """What it costs in football terms."""
    dd = table(f"downstream/{SOURCE}/tables/decision_displacement.csv")
    epv = table(f"downstream/{SOURCE}/tables/epv_by_variant.csv")
    e = epv[epv.loss_discount == 0.35].set_index("variant")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.2, 1.0]})
    fig.subplots_adjust(left=0.105, right=0.985, wspace=0.30, bottom=0.24, top=0.89)
    ax = axes[0]
    xs = np.arange(len(dd))
    ax.bar(xs, dd["frac_decisions_changed"] * 100, color=BLUE, width=0.55, zorder=3)
    for x, v in zip(xs, dd["frac_decisions_changed"] * 100):
        ax.text(x, v + 0.6, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([f"{t:.0%}" for t in dd["threshold"]], fontsize=8)
    ax.set_xlabel("how sure a coach wants to be")
    ax.set_ylabel("% of recommendations that change")
    ax.set_ylim(0, 42); ax.grid(axis="y", alpha=0.5)
    ax.set_title("(a) Up to a third of pass advice flips", loc="left", color=INK)

    ax = axes[1]
    vals = [float(e.loc["raw", "mean_epv"]), float(e.loc["isotonic", "mean_epv"])]
    ax.bar([0, 1], vals, color=[ORANGE, BLUE], width=0.5, zorder=3)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.annotate("", xy=(0.80, vals[1] + 0.055), xytext=(0.20, vals[0] + 0.055),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0,
                                connectionstyle="arc3,rad=-0.22"))
    ax.text(0.5, max(vals) * 0.40, f"{(1 - vals[0] / vals[1]) * 100:.0f}%\nunder-valued",
            ha="center", fontsize=8, color=INK2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["original\nvalues", "corrected\nvalues"], fontsize=8)
    ax.set_ylabel("average value of a pass")
    ax.set_ylim(0, max(vals) * 1.25); ax.grid(axis="y", alpha=0.5)
    ax.set_title("(b) Possession is under-valued", loc="left", color=INK)
    save(fig, "fig08_downstream", tight=False)


def main() -> int:
    setup()
    banner("Building paper figures")
    for fn in (fig01_what_is_pitch_control, fig02_calibration_vs_discrimination,
               fig03_reliability, fig04_score_decomposition, fig05_where_it_breaks,
               fig06_observability, fig07_transport_and_fix, fig08_downstream):
        try:
            fn()
        except Exception as exc:
            print(f"  {fn.__name__}: FAILED - {type(exc).__name__}: {exc}")
            raise
    print(f"\n  {len(list(FIGDIR.glob('*.pdf')))} figures in {FIGDIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
