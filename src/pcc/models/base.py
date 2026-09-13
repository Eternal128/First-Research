"""Common interface for the candidate control models.

All five models answer the same question and are therefore directly
comparable: given the state at ball release and a destination, what is the
probability that team A is in control at ``t_arrival + h``? They differ only in
how much they are told and how much they are allowed to learn.

The interface deliberately forces geometric and learned models into the same
signature. That is what makes the comparison fair: a learned model sees exactly
the state an unlearned one sees, so any advantage it shows is attributable to
functional form and fitting, not to extra information.
"""

from __future__ import annotations

import abc
from typing import Iterable, Sequence

import numpy as np

from pcc.data.schema import ArrivalFrame
from pcc.kinematics import LocomotionParams


class ControlModel(abc.ABC):
    """Base class for every model that emits ``C_A`` in ``[0, 1]``."""

    #: Short identifier used in result tables and filenames.
    name: str = "base"
    #: Whether :meth:`fit` must be called before :meth:`control`.
    requires_fitting: bool = False
    #: Whether the model can evaluate an arbitrary grid of destinations (needed
    #: for spatial maps and for synthetic candidate arrivals).
    supports_grid: bool = True

    @abc.abstractmethod
    def control(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        """Return ``C_A`` at ``targets`` (default: the frame's own destination).

        Parameters
        ----------
        frame
            State at ball release.
        targets
            ``(n, 2)`` destinations. When ``None``, ``frame.target`` is used and
            the flight time is ``frame.flight_time``. When a grid is supplied,
            each destination's flight time is recomputed from the ball origin
            unless the model is time-invariant.

        Returns
        -------
        ndarray
            Shape ``(n,)``, values in ``[0, 1]``.
        """

    def fit(
        self,
        frames: Sequence[ArrivalFrame],
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> "ControlModel":
        """Fit the model. Geometric models are no-ops and return ``self``."""
        return self

    def predict(self, frames: Iterable[ArrivalFrame]) -> np.ndarray:
        """Vector of ``C_A`` evaluated at each frame's own destination."""
        return np.array([float(np.atleast_1d(self.control(f))[0]) for f in frames], dtype=float)

    # -- helpers shared by the geometric models ----------------------------
    @staticmethod
    def _flight_times(frame: ArrivalFrame, targets: np.ndarray) -> np.ndarray:
        """Flight time to each target, scaled from the frame's own flight time.

        For the frame's own destination this reproduces ``frame.flight_time``
        exactly. For other grid points it assumes the same mean ball speed,
        which is the least-assumption extension available without a ball-flight
        model; it is only used for visualisation and for synthetic candidate
        arrivals, never for the primary evaluation.
        """
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        origin = np.asarray(frame.meta.get("origin", frame.target), dtype=float).reshape(2)
        own_dist = float(np.linalg.norm(frame.target - origin))
        dists = np.linalg.norm(targets - origin[None, :], axis=1)
        if own_dist < 1e-6 or frame.flight_time <= 0:
            return np.full(dists.shape, max(frame.flight_time, 1e-6))
        speed = own_dist / frame.flight_time
        return np.maximum(dists / speed, 1e-6)


class GeometricModel(ControlModel):
    """Mixin for models parameterised by a :class:`LocomotionParams` envelope."""

    def __init__(self, params: LocomotionParams | None = None, *, tti_model: str = "bounded_accel"):
        self.params = params or LocomotionParams()
        if tti_model not in ("constant_speed", "bounded_accel"):
            raise ValueError(f"unknown tti_model {tti_model!r}")
        self.tti_model = tti_model

    def _tau(self, frame: ArrivalFrame, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Time-to-point matrices ``(n_targets, n_att)`` and ``(n_targets, n_def)``."""
        from pcc.kinematics import TTI_MODELS

        fn = TTI_MODELS[self.tti_model]
        tau_a = np.atleast_2d(fn(frame.att_xy, frame.att_v, targets, self.params, v_max=frame.att_vmax))
        tau_b = np.atleast_2d(fn(frame.def_xy, frame.def_v, targets, self.params, v_max=frame.def_vmax))
        return tau_a, tau_b

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(tti_model={self.tti_model!r}, params={self.params})"
