"""Smoothing, derivative estimation and player time-to-point models.

Two things in this module dominate the sensitivity of every pitch-control
model downstream, and both are estimation choices rather than measurements:

1. **Velocity and acceleration are never observed.** Providers deliver
   positions; velocity is a numerical derivative of a noisy signal and is
   therefore a function of the filter. Section 17 of the proposal treats the
   filter cutoff as an ablation axis rather than a fixed preprocessing detail.
2. **Time-to-point is a model, not a fact.** The constant-``v_max`` form used
   by the standard physics models ignores acceleration limits and is
   optimistic for short displacements and pessimistic for long ones. Both the
   constant-speed and the bounded-acceleration forms are implemented so the
   difference can be quantified rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Player locomotion parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LocomotionParams:
    """Kinematic envelope assumed for an outfield player.

    Defaults are round numbers in the range commonly assumed by published
    pitch-control implementations; they are *assumptions of this study*, not
    measurements, and ``scripts/08_ablations.py`` perturbs each of them.
    Where per-player estimation is feasible (see
    :func:`estimate_player_vmax`), the empirical value should be preferred and
    the assumed value retained only as a prior mean.
    """

    v_max: float = 5.0          # m/s, sustained sprint speed used for interception
    a_max: float = 7.0          # m/s^2, forward acceleration bound
    d_max: float = 7.0          # m/s^2, braking bound (unused by constant-speed model)
    reaction_time: float = 0.7  # s, delay before the velocity vector may change
    sigma_tti: float = 0.45     # s, scale of the arrival-time uncertainty logistic
    lambda_control: float = 4.3  # 1/s, rate of converting presence into control
    lambda_control_gk: float = 4.3 * 3.0  # goalkeepers may use hands

    def replace(self, **kwargs) -> "LocomotionParams":
        from dataclasses import replace as _replace

        return _replace(self, **kwargs)


# ---------------------------------------------------------------------------
# Derivative estimation
# ---------------------------------------------------------------------------
def smooth_positions(
    xy: np.ndarray,
    *,
    fps: float,
    window_seconds: float = 0.4,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky-Golay smoothing of a ``(T, 2)`` position track.

    Savitzky-Golay is chosen over a moving average because it preserves the
    local polynomial structure of a turn, which a boxcar filter flattens; the
    flattening biases speed downward exactly at the moments (changes of
    direction) where pitch-control models are most discriminative.

    ``NaN`` gaps are preserved: they are filled, filtered, then re-masked, so
    that an occlusion is never silently interpolated into a confident track.
    Use :func:`gap_mask` to decide which frames are trustworthy.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (T, 2)")

    window = int(round(window_seconds * fps))
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 2 + (polyorder % 2))
    if xy.shape[0] < window:
        return xy.copy()

    nan_mask = ~np.isfinite(xy).all(axis=1)
    filled = xy.copy()
    if nan_mask.any():
        if nan_mask.all():
            return xy.copy()
        idx = np.arange(xy.shape[0])
        for c in range(2):
            col = filled[:, c]
            good = np.isfinite(col)
            col[~good] = np.interp(idx[~good], idx[good], col[good])
            filled[:, c] = col

    out = np.column_stack(
        [savgol_filter(filled[:, c], window_length=window, polyorder=polyorder) for c in range(2)]
    )
    out[nan_mask] = np.nan
    return out


def derivative(series: np.ndarray, *, fps: float, window_seconds: float = 0.4, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay first derivative of a ``(T, 2)`` track, in units/s.

    Computing the derivative analytically from the fitted local polynomial
    (``deriv=1``) rather than differencing a smoothed track avoids compounding
    two filters and keeps the effective bandwidth explicit.
    """
    series = np.asarray(series, dtype=float)
    window = int(round(window_seconds * fps))
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 2 + (polyorder % 2))
    if series.shape[0] < window:
        d = np.gradient(np.nan_to_num(series, nan=0.0), 1.0 / fps, axis=0)
        d[~np.isfinite(series).all(axis=1)] = np.nan
        return d

    nan_mask = ~np.isfinite(series).all(axis=1)
    filled = series.copy()
    if nan_mask.any() and not nan_mask.all():
        idx = np.arange(series.shape[0])
        for c in range(2):
            col = filled[:, c]
            good = np.isfinite(col)
            col[~good] = np.interp(idx[~good], idx[good], col[good])
            filled[:, c] = col
    elif nan_mask.all():
        return np.full_like(series, np.nan)

    out = np.column_stack(
        [
            savgol_filter(filled[:, c], window_length=window, polyorder=polyorder, deriv=1, delta=1.0 / fps)
            for c in range(2)
        ]
    )
    out[nan_mask] = np.nan
    return out


def gap_mask(xy: np.ndarray, *, fps: float, max_gap_seconds: float = 0.5) -> np.ndarray:
    """Mark frames that sit inside an occlusion gap longer than the tolerance.

    Returns ``True`` where the frame should be treated as *not observed* even
    after interpolation. Broadcast-derived tracking (SkillCorner) produces long
    off-camera gaps; silently interpolating across them manufactures players
    who are not known to be where the model places them, which is precisely
    the failure mode RQ4 is meant to detect.
    """
    xy = np.asarray(xy, dtype=float)
    bad = ~np.isfinite(xy).all(axis=1)
    if not bad.any():
        return np.zeros(xy.shape[0], dtype=bool)

    max_gap = max(1, int(round(max_gap_seconds * fps)))
    out = np.zeros(xy.shape[0], dtype=bool)
    start = None
    for i, b in enumerate(bad):
        if b and start is None:
            start = i
        elif not b and start is not None:
            if i - start > max_gap:
                out[start:i] = True
            start = None
    if start is not None and xy.shape[0] - start > max_gap:
        out[start:] = True
    return out


def cap_speed(velocity: np.ndarray, *, v_cap: float = 12.0) -> np.ndarray:
    """Clip implausible speeds produced by tracking identity swaps.

    A 12 m/s cap sits above the fastest recorded human sprint speeds, so any
    frame above it is a tracking artefact rather than a fast player. The count
    of capped frames is a tracking-quality indicator and is reported, not
    discarded silently.
    """
    velocity = np.asarray(velocity, dtype=float)
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    scale = np.where(speed > v_cap, v_cap / np.maximum(speed, 1e-9), 1.0)
    return velocity * scale


def estimate_player_vmax(speeds: np.ndarray, *, quantile: float = 0.995, floor: float = 4.0) -> float:
    """Per-player sprint-speed estimate from the upper tail of observed speed.

    A high quantile rather than the maximum is used because the maximum of a
    noisy derivative is an estimate of the noise, not of the athlete. The floor
    protects players with too few high-intensity frames (substitutes,
    goalkeepers) from receiving an implausibly low envelope.
    """
    speeds = np.asarray(speeds, dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    if speeds.size < 50:
        return float("nan")
    return float(max(floor, np.quantile(speeds, quantile)))


# ---------------------------------------------------------------------------
# Time-to-point models
# ---------------------------------------------------------------------------
def time_to_point_constant_speed(
    positions: np.ndarray,
    velocities: np.ndarray,
    target: np.ndarray,
    params: LocomotionParams,
    *,
    v_max: np.ndarray | None = None,
) -> np.ndarray:
    """Time for each player to reach ``target``, constant-speed formulation.

    During the reaction window the player is assumed to continue on the current
    velocity vector; thereafter they travel in a straight line at ``v_max``:

    .. math::
        \\tau_j = t_r + \\frac{\\lVert \\mathbf{x} -
            (\\mathbf{r}_j + \\mathbf{v}_j t_r)\\rVert}{v_{\\max}}

    This is the form used by the standard physics-based implementations. Its
    known defect is that it ignores the acceleration bound, so it understates
    the time to cover short distances from a standing start.

    Parameters
    ----------
    positions, velocities
        ``(n_players, 2)`` arrays.
    target
        ``(2,)`` or ``(n_targets, 2)``.
    v_max
        Optional per-player override, shape ``(n_players,)``.

    Returns
    -------
    ndarray
        ``(n_players,)`` if ``target`` is a single point, else
        ``(n_targets, n_players)``.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))
    target = np.asarray(target, dtype=float)
    single = target.ndim == 1
    targets = np.atleast_2d(target)

    vm = np.full(positions.shape[0], params.v_max) if v_max is None else np.asarray(v_max, dtype=float)
    vm = np.where(np.isfinite(vm) & (vm > 0), vm, params.v_max)

    tr = params.reaction_time
    origin = positions + velocities * tr                      # (n_players, 2)
    delta = targets[:, None, :] - origin[None, :, :]           # (n_targets, n_players, 2)
    dist = np.linalg.norm(delta, axis=-1)
    tau = tr + dist / vm[None, :]
    return tau[0] if single else tau


def time_to_point_bounded_accel(
    positions: np.ndarray,
    velocities: np.ndarray,
    target: np.ndarray,
    params: LocomotionParams,
    *,
    v_max: np.ndarray | None = None,
) -> np.ndarray:
    """Time-to-point under a bang-bang acceleration bound.

    After the reaction delay the player is assumed to accelerate at
    ``a_max`` along the straight line to the target, starting from the
    component of current velocity along that line, until ``v_max`` is reached
    and thereafter to cruise. Solving
    ``d = u t + a t^2 / 2`` for the accelerating phase and adding the cruise
    remainder gives, with ``d_acc = (v_max^2 - u^2) / (2 a)``:

    .. math::
        \\tau = t_r + \\begin{cases}
            \\dfrac{-u + \\sqrt{u^2 + 2 a d}}{a} & d \\le d_{acc}\\\\[6pt]
            \\dfrac{v_{\\max}-u}{a} + \\dfrac{d - d_{acc}}{v_{\\max}} & d > d_{acc}
        \\end{cases}

    The initial along-path speed ``u`` is clipped to ``[0, v_max]``: a player
    running away from the target is credited with zero useful initial speed
    rather than a negative one, which would otherwise reward the model for
    assuming instantaneous reversal.

    This formulation is strictly more conservative than the constant-speed one
    for short displacements and is the recommended default; the constant-speed
    variant is retained because it is what the published models use and the
    comparison is itself a result (proposal Section 17).
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    velocities = np.atleast_2d(np.asarray(velocities, dtype=float))
    target = np.asarray(target, dtype=float)
    single = target.ndim == 1
    targets = np.atleast_2d(target)

    vm = np.full(positions.shape[0], params.v_max) if v_max is None else np.asarray(v_max, dtype=float)
    vm = np.where(np.isfinite(vm) & (vm > 0), vm, params.v_max)
    a = float(params.a_max)
    tr = params.reaction_time

    origin = positions + velocities * tr
    delta = targets[:, None, :] - origin[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(dist[..., None], 1e-9)

    u = np.einsum("tpc,pc->tp", direction, velocities)
    u = np.clip(u, 0.0, vm[None, :])

    d_acc = (vm[None, :] ** 2 - u**2) / (2.0 * a)
    t_acc_phase = (-u + np.sqrt(np.maximum(u**2 + 2.0 * a * dist, 0.0))) / a
    t_cruise = (vm[None, :] - u) / a + (dist - d_acc) / vm[None, :]
    tau = tr + np.where(dist <= d_acc, t_acc_phase, t_cruise)
    return tau[0] if single else tau


TTI_MODELS = {
    "constant_speed": time_to_point_constant_speed,
    "bounded_accel": time_to_point_bounded_accel,
}


def arrival_probability(tau: np.ndarray, t: float | np.ndarray, params: LocomotionParams) -> np.ndarray:
    """Probability a player with expected arrival ``tau`` has arrived by ``t``.

    The logistic form

    .. math::
        f(t \\mid \\tau) = \\left[1 + \\exp\\!\\left(
            -\\frac{\\pi (t - \\tau)}{\\sqrt{3}\\,\\sigma}\\right)\\right]^{-1}

    is the standard choice; the ``pi / sqrt(3)`` factor makes ``sigma`` the
    standard deviation of the underlying logistic, so the parameter is
    interpretable as "seconds of uncertainty about arrival time" rather than an
    arbitrary temperature. ``sigma`` absorbs path curvature, contact, and
    tracking error, and is therefore a nuisance parameter that must be either
    fitted or ablated - never quietly fixed and then reported as physics.
    """
    tau = np.asarray(tau, dtype=float)
    t = np.asarray(t, dtype=float)
    z = np.pi * (t - tau) / (np.sqrt(3.0) * max(params.sigma_tti, 1e-6))
    # Numerically stable logistic.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def ball_travel_time(
    origin: np.ndarray,
    destination: np.ndarray,
    *,
    speed: float | np.ndarray | None = None,
    observed_time: float | np.ndarray | None = None,
) -> np.ndarray:
    """Ball flight time, preferring the observed value over a speed model.

    When ball tracking is available the flight time is measured directly and
    ``observed_time`` should be passed. Only when it is missing (event-only
    data, or ball-track dropout) is the constant-speed fallback used. Which
    branch produced each row is recorded by the caller as a data-quality flag,
    because an imputed flight time makes the model's conditioning argument
    partly circular.
    """
    origin = np.atleast_2d(np.asarray(origin, dtype=float))
    destination = np.atleast_2d(np.asarray(destination, dtype=float))
    dist = np.linalg.norm(destination - origin, axis=1)

    if observed_time is not None:
        obs = np.asarray(observed_time, dtype=float)
        obs = np.broadcast_to(obs, dist.shape).astype(float)
    else:
        obs = np.full(dist.shape, np.nan)

    if speed is None:
        speed = 15.0  # m/s, a plausible mean ground-pass speed; ablated, not asserted
    spd = np.asarray(speed, dtype=float)
    spd = np.broadcast_to(spd, dist.shape).astype(float)
    spd = np.where(np.isfinite(spd) & (spd > 0.5), spd, 15.0)

    return np.where(np.isfinite(obs) & (obs > 0), obs, dist / spd)
