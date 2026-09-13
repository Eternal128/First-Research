"""Models 1 and 2: the Voronoi baseline and the physics-based control model.

Both are *unfitted*: every number they produce follows from a locomotion
envelope and a geometric rule. That is the property under test. A model with no
free parameters estimated on football outcomes has no mechanism by which its
output could have become a calibrated probability, other than the physics
happening to be right - which is exactly the hypothesis (H1) at issue.
"""

from __future__ import annotations

import numpy as np

from pcc.data.schema import ArrivalFrame
from pcc.kinematics import LocomotionParams, arrival_probability
from pcc.models.base import GeometricModel


class VoronoiControl(GeometricModel):
    """Model 1: dominant-region / nearest-player baseline.

    .. math::
        C_A(\\mathbf{x}) = \\mathbb{1}\\!\\left[
            \\min_{j \\in A} \\tau_j(\\mathbf{x}) <
            \\min_{k \\in B} \\tau_k(\\mathbf{x})\\right]

    With ``velocity_aware=False`` and the constant-speed time-to-point this
    reduces to the classical Voronoi tessellation of the pitch by Euclidean
    distance; with velocities it becomes a motion-model dominant region.

    As a *probability* forecast this model is maximally sharp and, almost
    certainly, badly calibrated: it can only ever say 0 or 1, so any error is
    maximally penalised by a proper score. It is included not as a serious
    competitor but as the reference point that quantifies how much of a
    physics-based model's skill comes from softening a hard boundary. It is
    also the model implicitly used whenever an analyst shades a pitch by
    "whose space this is".
    """

    name = "M1_voronoi"

    def __init__(
        self,
        params: LocomotionParams | None = None,
        *,
        tti_model: str = "bounded_accel",
        velocity_aware: bool = True,
        epsilon: float = 1e-3,
    ):
        super().__init__(params, tti_model=tti_model)
        self.velocity_aware = velocity_aware
        # Hard 0/1 forecasts have infinite log loss when wrong. Clipping to
        # [epsilon, 1 - epsilon] is a *charitable* modification made explicit
        # here rather than hidden in the scorer; the Brier score, which is
        # finite either way, is the primary metric for this reason.
        self.epsilon = float(epsilon)

    def control(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        targets = frame.target[None, :] if targets is None else np.atleast_2d(targets)
        if self.velocity_aware:
            tau_a, tau_b = self._tau(frame, targets)
        else:
            zeros_a = np.zeros_like(frame.att_v)
            zeros_b = np.zeros_like(frame.def_v)
            stub = ArrivalFrame(
                att_xy=frame.att_xy, att_v=zeros_a, def_xy=frame.def_xy, def_v=zeros_b,
                target=frame.target, flight_time=frame.flight_time,
                att_vmax=frame.att_vmax, def_vmax=frame.def_vmax,
            )
            tau_a, tau_b = self._tau(stub, targets)

        a_min = np.nanmin(tau_a, axis=1)
        b_min = np.nanmin(tau_b, axis=1)
        hard = (a_min < b_min).astype(float)
        return np.clip(hard, self.epsilon, 1.0 - self.epsilon)


class ReachabilitySigmoid(GeometricModel):
    """A deliberately minimal soft baseline: a logistic in the reachability gap.

    .. math::
        C_A(\\mathbf{x}) = \\sigma\\!\\left(
            \\frac{\\min_k \\tau_k - \\min_j \\tau_j}{s}\\right)

    One free parameter (``scale`` ``s``), no ball-control dynamics, no
    multi-player accumulation. It exists to answer a specific question that the
    headline model comparison cannot: how much of any physics-based model's
    calibration is attributable to its dynamics, and how much simply to
    monotonically squashing the time-to-point difference? If this model matches
    the physics model, the dynamics are decorative.

    ``fit`` estimates ``s`` by minimising the Brier score on the training
    arrivals; it is therefore the only "geometric" model here that touches
    outcome data, and it is reported as a fitted model.
    """

    name = "M2a_reach_sigmoid"
    requires_fitting = False

    def __init__(
        self,
        params: LocomotionParams | None = None,
        *,
        tti_model: str = "bounded_accel",
        scale: float = 1.0,
    ):
        super().__init__(params, tti_model=tti_model)
        self.scale = float(scale)

    def _gap(self, frame: ArrivalFrame, targets: np.ndarray) -> np.ndarray:
        tau_a, tau_b = self._tau(frame, targets)
        return np.nanmin(tau_b, axis=1) - np.nanmin(tau_a, axis=1)

    def control(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        targets = frame.target[None, :] if targets is None else np.atleast_2d(targets)
        z = self._gap(frame, targets) / max(self.scale, 1e-6)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, frames, y, *, sample_weight=None):
        from scipy.optimize import minimize_scalar

        gaps = np.array([float(self._gap(f, f.target[None, :])[0]) for f in frames])
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        def brier(log_s: float) -> float:
            p = 1.0 / (1.0 + np.exp(-np.clip(gaps / np.exp(log_s), -50, 50)))
            return float(np.average((p - y) ** 2, weights=w))

        res = minimize_scalar(brier, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        self.scale = float(np.exp(res.x))
        return self


class PhysicalControl(GeometricModel):
    """Model 2: physics-based pitch control with ball-control dynamics.

    Follows the published formulation in which each player's probability of
    having taken control accumulates over time after arrival, competing for the
    remaining probability mass:

    .. math::
        \\frac{\\mathrm{d}\\mathrm{PPC}_j}{\\mathrm{d}t}
        = \\Bigl(1 - \\sum_k \\mathrm{PPC}_k(t)\\Bigr)\\,
          f_j(t \\mid \\tau_j)\\, \\lambda_j ,

    with :math:`f_j` the logistic arrival probability from
    :func:`~pcc.kinematics.arrival_probability` and :math:`\\lambda_j` the rate
    at which a present player converts presence into control. Integrating from
    the ball's arrival until the mass is exhausted and summing over team A
    gives :math:`C_A`.

    Free parameters: ``sigma_tti``, ``lambda_control``, ``v_max``,
    ``reaction_time``. None is estimated from outcome data in the default
    configuration, which is the point: the resulting number is asserted to be a
    probability on physical grounds alone.

    Implementation notes
    --------------------
    * Integration starts at the ball's arrival time and runs forward. Players
      who cannot arrive before the ball still contribute, with the logistic
      giving them small weight - this is what produces the smooth field.
    * The integration is explicit Euler with a small step; the step size is an
      ablation axis (``dt``) because too coarse a step systematically inflates
      the total mass assigned per step.
    * The returned value is renormalised by the total mass actually
      accumulated. Without renormalisation a truncated integration leaves mass
      unassigned and the "probability" does not sum to one across teams - a
      quiet violation of the probabilistic reading that is easy to miss.
    """

    name = "M2_physical"

    def __init__(
        self,
        params: LocomotionParams | None = None,
        *,
        tti_model: str = "bounded_accel",
        dt: float = 0.04,
        t_max: float = 10.0,
        mass_tolerance: float = 0.995,
        renormalise: bool = True,
    ):
        super().__init__(params, tti_model=tti_model)
        self.dt = float(dt)
        self.t_max = float(t_max)
        self.mass_tolerance = float(mass_tolerance)
        self.renormalise = bool(renormalise)

    def control(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        if targets is None:
            targets = frame.target[None, :]
            flight = np.array([frame.flight_time], dtype=float)
        else:
            targets = np.atleast_2d(np.asarray(targets, dtype=float))
            flight = self._flight_times(frame, targets)

        tau_a, tau_b = self._tau(frame, targets)          # (n_t, n_a), (n_t, n_b)
        n_t, n_a = tau_a.shape
        n_b = tau_b.shape[1]

        lam_a = np.where(frame.att_is_gk, self.params.lambda_control_gk, self.params.lambda_control)
        lam_b = np.where(frame.def_is_gk, self.params.lambda_control_gk, self.params.lambda_control)

        ppc_a = np.zeros((n_t, n_a))
        ppc_b = np.zeros((n_t, n_b))
        active = np.ones(n_t, dtype=bool)

        n_steps = int(np.ceil(self.t_max / self.dt))
        for step in range(n_steps):
            if not active.any():
                break
            t = flight + (step + 0.5) * self.dt  # midpoint of the step
            remaining = 1.0 - ppc_a.sum(axis=1) - ppc_b.sum(axis=1)
            remaining = np.clip(remaining, 0.0, 1.0)

            f_a = arrival_probability(tau_a, t[:, None], self.params)
            f_b = arrival_probability(tau_b, t[:, None], self.params)

            d_a = remaining[:, None] * f_a * lam_a[None, :] * self.dt
            d_b = remaining[:, None] * f_b * lam_b[None, :] * self.dt

            # Guard against a step that would allocate more than the remaining
            # mass, which explicit Euler can do when lambda * dt is not small.
            total = d_a.sum(axis=1) + d_b.sum(axis=1)
            scale = np.where(total > remaining, remaining / np.maximum(total, 1e-12), 1.0)
            ppc_a += d_a * scale[:, None] * active[:, None]
            ppc_b += d_b * scale[:, None] * active[:, None]

            active = (ppc_a.sum(axis=1) + ppc_b.sum(axis=1)) < self.mass_tolerance

        mass_a = ppc_a.sum(axis=1)
        mass_b = ppc_b.sum(axis=1)
        total = mass_a + mass_b
        if self.renormalise:
            out = np.where(total > 1e-9, mass_a / np.maximum(total, 1e-12), 0.5)
        else:
            out = mass_a
        return np.clip(out, 0.0, 1.0)

    def unassigned_mass(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        """Probability mass left unallocated by the truncated integration.

        Reported as a diagnostic: a model whose output needs substantial
        renormalisation before it sums to one across teams is not delivering a
        proper probability distribution over the two outcomes, independently of
        whether the renormalised value happens to be well calibrated.
        """
        saved = self.renormalise
        try:
            self.renormalise = False
            mass_a = self.control(frame, targets)
        finally:
            self.renormalise = saved
        # Recompute team B mass via the mirrored frame.
        mirrored = ArrivalFrame(
            att_xy=frame.def_xy, att_v=frame.def_v, def_xy=frame.att_xy, def_v=frame.att_v,
            target=frame.target, flight_time=frame.flight_time,
            att_is_gk=frame.def_is_gk, def_is_gk=frame.att_is_gk,
            att_vmax=frame.def_vmax, def_vmax=frame.att_vmax, meta=frame.meta,
        )
        try:
            self.renormalise = False
            mass_b = self.control(mirrored, targets)
        finally:
            self.renormalise = saved
        return np.clip(1.0 - mass_a - mass_b, 0.0, 1.0)
