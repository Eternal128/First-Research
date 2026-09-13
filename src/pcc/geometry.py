"""Pitch frame, coordinate normalisation and zoning.

Canonical frame used everywhere downstream of :mod:`pcc.data.preprocess`:

* origin at the centre spot;
* ``x`` in metres along the long axis, increasing toward the goal that the
  *team in possession is attacking*;
* ``y`` in metres along the short axis;
* pitch spans ``[-L/2, +L/2] x [-W/2, +W/2]``.

Attacking-direction normalisation matters for this study specifically: pitch
control is asymmetric in ``x`` (offside line, goalkeeper sweeping, defensive
block depth), so a model evaluated in raw provider coordinates would mix two
mirror-image populations and blur every spatial calibration statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# IFAB permits 90-120 m x 45-90 m; UEFA club competitions standardise on
# 105 x 68 m, which most providers also use as their normalised frame.
DEFAULT_LENGTH_M: float = 105.0
DEFAULT_WIDTH_M: float = 68.0


@dataclass(frozen=True)
class Pitch:
    """Physical pitch dimensions in metres."""

    length: float = DEFAULT_LENGTH_M
    width: float = DEFAULT_WIDTH_M

    @property
    def half_length(self) -> float:
        return self.length / 2.0

    @property
    def half_width(self) -> float:
        return self.width / 2.0

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` for matplotlib ``imshow``/``extent``."""
        return (-self.half_length, self.half_length, -self.half_width, self.half_width)

    def attacking_goal(self) -> np.ndarray:
        """Centre of the goal being attacked, in the canonical frame."""
        return np.array([self.half_length, 0.0])

    def defending_goal(self) -> np.ndarray:
        return np.array([-self.half_length, 0.0])

    def contains(self, xy: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Boolean mask for points inside the playing area (within ``tol`` m)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        return (np.abs(xy[:, 0]) <= self.half_length + tol) & (
            np.abs(xy[:, 1]) <= self.half_width + tol
        )

    def clip(self, xy: np.ndarray) -> np.ndarray:
        """Clip points onto the playing area."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float)).copy()
        xy[:, 0] = np.clip(xy[:, 0], -self.half_length, self.half_length)
        xy[:, 1] = np.clip(xy[:, 1], -self.half_width, self.half_width)
        return xy

    def grid(self, n_x: int = 105, n_y: int = 68) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centred evaluation grid, returned as ``(XX, YY)`` meshgrids."""
        xs = np.linspace(-self.half_length, self.half_length, n_x)
        ys = np.linspace(-self.half_width, self.half_width, n_y)
        return np.meshgrid(xs, ys, indexing="xy")


def to_canonical(
    xy: np.ndarray,
    *,
    source_extent: tuple[float, float, float, float],
    pitch: Pitch = Pitch(),
    attacking_left_to_right: bool = True,
) -> np.ndarray:
    """Map provider coordinates onto the canonical centred, metric frame.

    Parameters
    ----------
    xy
        ``(n, 2)`` array of provider coordinates.
    source_extent
        ``(xmin, xmax, ymin, ymax)`` of the provider frame. Examples:
        StatsBomb ``(0, 120, 0, 80)`` (units are yards-like, not metres),
        Metrica normalised ``(0, 1, 0, 1)``, a centred metric frame
        ``(-52.5, 52.5, -34, 34)``.
    attacking_left_to_right
        If ``False`` the frame is rotated 180 degrees so that the team in
        possession always attacks ``+x``.

    Notes
    -----
    The mapping is affine and *rescales* the provider frame onto
    ``pitch``. For providers whose frame is a fixed template rather than true
    metres (StatsBomb), this is an approximation: the real pitch may not be
    105 x 68 m, and the resulting metric error propagates into every
    time-to-point calculation. :mod:`pcc.data.preprocess` records the
    assumption per match so the ablation in ``scripts/08_ablations.py`` can
    quantify its effect.
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    x0, x1, y0, y1 = source_extent
    if x1 == x0 or y1 == y0:
        raise ValueError(f"degenerate source_extent: {source_extent!r}")

    u = (xy[:, 0] - x0) / (x1 - x0)  # -> [0, 1]
    v = (xy[:, 1] - y0) / (y1 - y0)

    out = np.empty_like(xy)
    out[:, 0] = (u - 0.5) * pitch.length
    out[:, 1] = (v - 0.5) * pitch.width

    if not attacking_left_to_right:
        out *= -1.0
    return out


def zone_index(
    xy: np.ndarray,
    *,
    pitch: Pitch = Pitch(),
    n_x: int = 5,
    n_y: int = 3,
) -> np.ndarray:
    """Assign points to a coarse ``n_x`` x ``n_y`` zone lattice.

    A 5 x 3 lattice (defensive/def-mid/mid/att-mid/attacking third-fifths by
    left/centre/right channel) is the default because it is the coarsest
    partition that still separates the regions where pitch-control models are
    expected to behave differently: own box, build-up, midfield, half-space,
    final third. Finer lattices are available but split the sample thinly,
    which matters because every calibration statistic reported per zone needs
    its own cluster bootstrap.
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    u = np.clip((xy[:, 0] + pitch.half_length) / pitch.length, 0.0, 1.0 - 1e-12)
    v = np.clip((xy[:, 1] + pitch.half_width) / pitch.width, 0.0, 1.0 - 1e-12)
    ix = np.floor(u * n_x).astype(int)
    iy = np.floor(v * n_y).astype(int)
    return ix * n_y + iy


def zone_label(idx: int, *, n_x: int = 5, n_y: int = 3) -> str:
    """Human-readable label for a :func:`zone_index` value."""
    x_names = {
        5: ["def-fifth", "def-mid", "middle", "att-mid", "att-fifth"],
        3: ["def-third", "mid-third", "att-third"],
    }.get(n_x, [f"x{i}" for i in range(n_x)])
    y_names = {
        3: ["right", "centre", "left"],
        5: ["wide-right", "half-right", "centre", "half-left", "wide-left"],
    }.get(n_y, [f"y{i}" for i in range(n_y)])
    ix, iy = divmod(int(idx), n_y)
    return f"{x_names[ix]}/{y_names[iy]}"


def distance_to_goal(xy: np.ndarray, *, pitch: Pitch = Pitch()) -> np.ndarray:
    """Euclidean distance from each point to the centre of the attacked goal."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    return np.linalg.norm(xy - pitch.attacking_goal(), axis=1)


def distance_to_touchline(xy: np.ndarray, *, pitch: Pitch = Pitch()) -> np.ndarray:
    """Distance to the nearer touchline (0 on the line, W/2 in the centre)."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    return pitch.half_width - np.abs(xy[:, 1])
