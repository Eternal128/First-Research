"""Does miscalibration change the football numbers people actually use?

A calibration paper that stops at the Brier score has not shown that anything
matters. This module propagates the raw and recalibrated control forecasts
through the metrics that pitch control is used to build, and measures how much
they move.

The logic of the propagation
----------------------------
Most pitch-control-derived quantities are approximately *linear* in ``C``:

.. math::
    \\mathrm{EPV}(\\text{pass to } \\mathbf{x})
    \\approx C_A(\\mathbf{x})\\, V(\\mathbf{x}) + (1 - C_A(\\mathbf{x}))\\, \\bar V_{\\text{loss}}(\\mathbf{x})

space-creation and off-ball-run metrics are differences of control fields, and
team-compactness or pressure summaries are integrals of them. Linearity has a
sharp consequence the paper should state explicitly: **a calibration error that
is constant in sign over a region does not average away when integrated over
that region - it accumulates.** A model that is overconfident by 0.05 in the
final third biases every final-third space metric in the same direction, all
season. Conversely, a purely random error would average out, so the
*systematic* part measured by the calibration slope is exactly the part that
propagates.

What is deliberately not provided
---------------------------------
No positional value surface is hard-coded. Published expected-threat style
grids are the property of their authors and their values depend on the league
and season they were fitted on. :class:`PositionalValueSurface` must be fitted
from data or supplied explicitly; a distance-to-goal placeholder is available
*only* for pipeline testing and refuses to be used without acknowledgement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pcc.geometry import Pitch, distance_to_goal


@dataclass
class PositionalValueSurface:
    """A grid of positional value ``V(x, y)``: the value of having the ball there.

    Fit it with :meth:`fit_from_outcomes` on possession sequences, or construct
    it from an externally supplied grid. It is intentionally awkward to
    fabricate, because a fabricated value surface would let the downstream
    analysis report whatever the analyst wanted.
    """

    grid: np.ndarray               # (n_y, n_x)
    pitch: Pitch = Pitch()
    provenance: str = "unspecified"

    @classmethod
    def fit_from_outcomes(
        cls,
        locations: np.ndarray,
        outcomes: np.ndarray,
        *,
        pitch: Pitch = Pitch(),
        n_x: int = 16,
        n_y: int = 12,
        prior_strength: float = 20.0,
        provenance: str = "fitted from possession outcomes",
    ) -> "PositionalValueSurface":
        """Empirical-Bayes cell means of a binary outcome (e.g. goal within k actions).

        Cells are shrunk toward the global mean with strength ``prior_strength``
        because many cells are sparse; without shrinkage the corner cells would
        carry values estimated from a handful of possessions and would dominate
        any integral over the pitch.
        """
        locations = np.atleast_2d(np.asarray(locations, dtype=float))
        outcomes = np.asarray(outcomes, dtype=float).ravel()
        x_edges = np.linspace(-pitch.half_length, pitch.half_length, n_x + 1)
        y_edges = np.linspace(-pitch.half_width, pitch.half_width, n_y + 1)
        ix = np.clip(np.digitize(locations[:, 0], x_edges[1:-1]), 0, n_x - 1)
        iy = np.clip(np.digitize(locations[:, 1], y_edges[1:-1]), 0, n_y - 1)

        prior = float(outcomes.mean()) if outcomes.size else 0.0
        grid = np.full((n_y, n_x), prior)
        for gy in range(n_y):
            for gx in range(n_x):
                m = (ix == gx) & (iy == gy)
                n = int(m.sum())
                if n:
                    grid[gy, gx] = (outcomes[m].sum() + prior_strength * prior) / (n + prior_strength)
        return cls(grid=grid, pitch=pitch, provenance=provenance)

    @classmethod
    def distance_placeholder(cls, *, pitch: Pitch = Pitch(), n_x: int = 16, n_y: int = 12,
                             acknowledge_placeholder: bool = False) -> "PositionalValueSurface":
        """A monotone-in-distance-to-goal stand-in. **Pipeline testing only.**

        It is not a model of anything. It exists so the downstream code can be
        exercised before a real value surface has been fitted, and it refuses to
        construct unless the caller states that it knows this.
        """
        if not acknowledge_placeholder:
            raise ValueError(
                "distance_placeholder produces fabricated values and must not appear in "
                "any reported result. Pass acknowledge_placeholder=True to use it for "
                "pipeline testing only."
            )
        XX, YY = pitch.grid(n_x, n_y)
        d = distance_to_goal(np.column_stack([XX.ravel(), YY.ravel()]), pitch=pitch).reshape(XX.shape)
        grid = np.exp(-d / 25.0)
        return cls(grid=grid / grid.max(), pitch=pitch, provenance="PLACEHOLDER - not a fitted surface")

    def value_at(self, xy: np.ndarray) -> np.ndarray:
        """Look up ``V`` by nearest cell. Non-finite coordinates return ``NaN``.

        Returning ``NaN`` rather than silently clipping a missing coordinate to
        a corner cell keeps a data defect visible instead of converting it into
        a plausible-looking value.
        """
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        n_y, n_x = self.grid.shape
        finite = np.isfinite(xy).all(axis=1)
        out = np.full(xy.shape[0], np.nan)
        if not finite.any():
            return out
        good = xy[finite]
        ix = np.clip(
            ((good[:, 0] + self.pitch.half_length) / self.pitch.length * n_x).astype(int), 0, n_x - 1
        )
        iy = np.clip(
            ((good[:, 1] + self.pitch.half_width) / self.pitch.width * n_y).astype(int), 0, n_y - 1
        )
        out[finite] = self.grid[iy, ix]
        return out


def expected_possession_value(
    control: np.ndarray,
    destination: np.ndarray,
    surface: PositionalValueSurface,
    *,
    loss_discount: float = 0.35,
) -> np.ndarray:
    """A minimal EPV for a single pass, linear in the control probability.

    .. math::
        \\mathrm{EPV} = C_A V(\\mathbf{x}) - (1 - C_A)\\,\\delta\\, V_{B}(\\mathbf{x})

    where :math:`V_B` is the mirrored surface (the value the *opponent* gains
    from winning the ball there) and :math:`\\delta` discounts it. The specific
    functional form is not the contribution; its linearity in ``C`` is, because
    that is what makes the propagation of calibration error tractable and
    signed.

    ``loss_discount`` is a stated assumption. The study reports the downstream
    effect across a range of it rather than at one value, since the conclusion
    "recalibration changes EPV by X" would otherwise be partly a statement
    about a number chosen by the analyst.
    """
    control = np.asarray(control, dtype=float).ravel()
    destination = np.atleast_2d(np.asarray(destination, dtype=float))
    v_a = surface.value_at(destination)
    v_b = surface.value_at(destination * np.array([-1.0, -1.0]))
    return control * v_a - (1.0 - control) * loss_discount * v_b


def pass_value_comparison(
    df: pd.DataFrame,
    surface: PositionalValueSurface,
    *,
    control_cols: dict[str, str],
    loss_discounts=(0.2, 0.35, 0.5),
) -> pd.DataFrame:
    """Aggregate EPV computed from each control variant, across discount values.

    ``control_cols`` maps a label ("raw", "platt", "isotonic", "baseline") to the
    column holding that variant's forecast. The comparison reports the mean EPV,
    the total over the corpus, and the mean absolute per-pass difference from
    the recalibrated reference - the last being the one that tells a
    practitioner whether the difference would ever be visible in a single
    match's report.
    """
    dest = df[["dest_x", "dest_y"]].to_numpy()
    rows = []
    for delta in loss_discounts:
        values = {
            label: expected_possession_value(df[col].to_numpy(), dest, surface, loss_discount=delta)
            for label, col in control_cols.items()
        }
        reference = values.get("isotonic", next(iter(values.values())))
        for label, v in values.items():
            rows.append(
                {
                    "variant": label,
                    "loss_discount": delta,
                    "mean_epv": float(np.mean(v)),
                    "total_epv": float(np.sum(v)),
                    "mean_abs_diff_vs_reference": float(np.mean(np.abs(v - reference))),
                    "mean_signed_diff_vs_reference": float(np.mean(v - reference)),
                    "n": int(len(v)),
                }
            )
    return pd.DataFrame(rows)


def space_metrics(model, frame, *, pitch: Pitch = Pitch(), n_x: int = 40, n_y: int = 26) -> dict[str, float]:
    """Integral summaries of a control field: space owned, compactness, pressure.

    * ``space_owned`` - the area of the pitch with :math:`C_A > 0.5`, in m^2.
      The classical "space created" quantity.
    * ``soft_space_owned`` - :math:`\\int C_A \\,\\mathrm{d}A`. The
      probability-weighted version, which is the one that *should* be used if
      ``C`` really is a probability, and which differs from the hard version by
      an amount that depends entirely on calibration.
    * ``dangerous_space`` - the same integral restricted to the attacking third.
    * ``control_entropy`` - mean binary entropy of the field; high values mean a
      contested pitch. This is the summary most sensitive to miscalibration,
      because entropy is maximised at 0.5 and an overconfident model pushes mass
      to the extremes, systematically understating contestedness.

    The ``space_owned`` versus ``soft_space_owned`` gap is a clean, reportable
    demonstration of why calibration matters for a metric practitioners already
    compute.
    """
    XX, YY = pitch.grid(n_x, n_y)
    targets = np.column_stack([XX.ravel(), YY.ravel()])
    C = np.clip(np.asarray(model.control(frame, targets), dtype=float), 1e-9, 1 - 1e-9)
    cell_area = (pitch.length / n_x) * (pitch.width / n_y)
    att_third = targets[:, 0] > pitch.half_length / 3

    entropy = -(C * np.log2(C) + (1 - C) * np.log2(1 - C))
    return {
        "space_owned_m2": float((C > 0.5).sum() * cell_area),
        "soft_space_owned_m2": float(C.sum() * cell_area),
        "hard_minus_soft_m2": float(((C > 0.5).sum() - C.sum()) * cell_area),
        "dangerous_space_m2": float(C[att_third].sum() * cell_area),
        "control_entropy": float(entropy.mean()),
        "mean_control": float(C.mean()),
    }


def off_ball_run_value(
    model,
    frame,
    *,
    player_index: int,
    displacement: np.ndarray,
    surface: PositionalValueSurface,
    pitch: Pitch = Pitch(),
    n_x: int = 40,
    n_y: int = 26,
) -> dict[str, float]:
    """Value created by moving one attacker, measured as a change in weighted control.

    .. math::
        \\Delta = \\int \\bigl(C_A'(\\mathbf{x}) - C_A(\\mathbf{x})\\bigr)
                  V(\\mathbf{x}) \\,\\mathrm{d}\\mathbf{x}

    This is the standard construction behind off-ball-run and space-creation
    metrics. Note what it inherits: it is a *difference* of two control fields,
    so a calibration error that is constant cancels, but a calibration error
    that varies with the situation - which is precisely what a slope below one
    produces - does not. The study should therefore expect off-ball metrics to
    be *less* sensitive to calibration-in-the-large and *more* sensitive to
    slope error than EPV is, and it can test that directly.
    """
    from dataclasses import replace as _replace

    XX, YY = pitch.grid(n_x, n_y)
    targets = np.column_stack([XX.ravel(), YY.ravel()])
    v = surface.value_at(targets)
    cell_area = (pitch.length / n_x) * (pitch.width / n_y)

    base = np.asarray(model.control(frame, targets), dtype=float)
    moved_xy = frame.att_xy.copy()
    moved_xy[player_index] = pitch.clip((moved_xy[player_index] + np.asarray(displacement))[None, :])[0]
    moved = _replace(frame, att_xy=moved_xy)
    after = np.asarray(model.control(moved, targets), dtype=float)

    diff = after - base
    return {
        "delta_soft_space_m2": float(diff.sum() * cell_area),
        "delta_weighted_value": float((diff * v).sum() * cell_area),
        "max_local_gain": float(diff.max()),
        "max_local_loss": float(diff.min()),
    }


def downstream_sensitivity(
    df: pd.DataFrame,
    *,
    raw_col: str,
    calibrated_col: str,
    surface: PositionalValueSurface,
    thresholds=(0.5, 0.6, 0.7),
) -> pd.DataFrame:
    """Summarise how far recalibration moves the decisions and the valuations.

    Combines the continuous effect (change in EPV) with the discrete one
    (fraction of pass recommendations that flip). The paper's practical claim
    should rest on the second: a mean EPV shift of 0.002 is a statistic, but
    "8% of the passes this model would have recommended, it no longer
    recommends" is a consequence.
    """
    from pcc.evaluation.decision_curve import threshold_displacement

    dest = df[["dest_x", "dest_y"]].to_numpy()
    raw = df[raw_col].to_numpy()
    cal = df[calibrated_col].to_numpy()
    epv_raw = expected_possession_value(raw, dest, surface)
    epv_cal = expected_possession_value(cal, dest, surface)

    disp = threshold_displacement(df["y_control"].to_numpy(), raw, cal, thresholds=thresholds)
    disp["mean_epv_raw"] = float(np.mean(epv_raw))
    disp["mean_epv_calibrated"] = float(np.mean(epv_cal))
    disp["mean_abs_epv_shift"] = float(np.mean(np.abs(epv_raw - epv_cal)))
    disp["corr_epv"] = float(np.corrcoef(epv_raw, epv_cal)[0, 1])
    disp["value_surface_provenance"] = surface.provenance
    return disp
