"""Post-hoc recalibration maps (RQ5).

Each class learns a map :math:`g : [0,1] \\to [0,1]` applied to an existing
forecast. The scientific use is diagnostic rather than cosmetic: if a
one-parameter map recovers most of the lost score, the underlying model's
*ordering* of situations was sound and only its scale was wrong; if it does
not, the model is wrong about which situations differ, which is a much more
serious defect and cannot be patched downstream.

Critical protocol point: a recalibration map must be fitted on data disjoint
from the data used to evaluate it, and the split must be by match. Fitting and
evaluating a flexible map such as isotonic regression on the same arrivals will
show near-perfect calibration for any model whatsoever, which is a well-known
artefact and would invalidate the entire RQ5 analysis.
"""

from __future__ import annotations

import abc

import numpy as np

EPS = 1e-6


def _logit(p, eps: float = EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -500, 500)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class Recalibrator(abc.ABC):
    name = "identity"
    n_parameters = 0

    @abc.abstractmethod
    def fit(self, p, y, sample_weight=None) -> "Recalibrator": ...

    @abc.abstractmethod
    def transform(self, p) -> np.ndarray: ...

    def fit_transform(self, p, y, sample_weight=None) -> np.ndarray:
        return self.fit(p, y, sample_weight).transform(p)


class IdentityRecalibrator(Recalibrator):
    """No-op, for holding the "raw model" row in the same table shape."""

    name = "identity"

    def fit(self, p, y, sample_weight=None):
        return self

    def transform(self, p):
        return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)


class PlattScaling(Recalibrator):
    """Two-parameter logistic recalibration: ``g(p) = sigma(a + b logit p)``.

    The classical Platt map, here in its logit-linear form. ``b`` is exactly
    the calibration slope, so fitting this map and reading ``b`` is the same
    diagnostic as :func:`~pcc.calibration.metrics.calibration_slope_intercept`
    - which is a useful internal consistency check and is asserted in the
    tests.

    Strength: two parameters, so it cannot overfit a match-level split and
    generalises across competitions. Limitation: it can only fix monotone,
    logit-linear distortion. A model whose reliability curve is S-shaped in
    logit space (plausible for a physics model that is well behaved in the
    middle and wrong at the extremes) will not be fixed by it, and the residual
    miscalibration after Platt scaling is therefore itself informative.
    """

    name = "platt"
    n_parameters = 2

    def __init__(self) -> None:
        self.a: float = 0.0
        self.b: float = 1.0

    def fit(self, p, y, sample_weight=None):
        from sklearn.linear_model import LogisticRegression

        from pcc.calibration.metrics import _unpenalised_logistic

        z = _logit(p).reshape(-1, 1)
        y = np.asarray(y, dtype=int).ravel()
        if len(np.unique(y)) < 2 or np.ptp(z) < 1e-9:
            self.a, self.b = 0.0, 1.0
            return self
        lr = _unpenalised_logistic(LogisticRegression)
        lr.fit(z, y, sample_weight=sample_weight)
        self.b = float(lr.coef_.ravel()[0])
        self.a = float(lr.intercept_[0])
        return self

    def transform(self, p):
        return _sigmoid(self.a + self.b * _logit(p))


class IsotonicRecalibrator(Recalibrator):
    """Non-parametric monotone recalibration by pool-adjacent-violators.

    Maximum flexibility subject to monotonicity, and therefore the natural
    upper bound on what post-hoc recalibration can achieve for a given model's
    ordering. Its weaknesses are the mirror image of Platt's: it overfits small
    samples, produces a step function that extrapolates poorly to forecast
    values unseen in training, and is unstable at the tails where pitch-control
    forecasts pile up.

    The gap between the isotonic and Platt results is itself a reportable
    quantity: it measures how much of the miscalibration is non-logit-linear.
    """

    name = "isotonic"
    n_parameters = float("inf")

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds
        self._iso = None

    def fit(self, p, y, sample_weight=None):
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds=self.out_of_bounds
        )
        self._iso.fit(np.asarray(p, dtype=float).ravel(), np.asarray(y, dtype=float).ravel(),
                      sample_weight=sample_weight)
        return self

    def transform(self, p):
        if self._iso is None:
            raise RuntimeError("IsotonicRecalibrator is not fitted")
        return np.clip(self._iso.predict(np.asarray(p, dtype=float).ravel()), 0.0, 1.0)


class BetaCalibration(Recalibrator):
    """Three-parameter beta calibration.

    .. math::
        g(p) = \\sigma\\bigl(c + a \\ln p - b \\ln (1 - p)\\bigr)

    Fitted as a logistic regression on the two features :math:`\\ln p` and
    :math:`-\\ln(1-p)`. Unlike Platt scaling it is not forced to map 0.5 to a
    fixed point and can represent asymmetric distortion at the two tails, which
    is the expected shape for a control model that is confident-and-wrong when
    it claims near-certain possession but conservative near zero.

    Included because it is the natural intermediate between the two-parameter
    and the non-parametric map, and because a finding that it suffices would be
    a clean, transferable recommendation for practitioners.
    """

    name = "beta"
    n_parameters = 3

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 1.0
        self.c: float = 0.0

    @staticmethod
    def _design(p):
        p = np.clip(np.asarray(p, dtype=float).ravel(), EPS, 1 - EPS)
        return np.column_stack([np.log(p), -np.log(1 - p)])

    def fit(self, p, y, sample_weight=None):
        from sklearn.linear_model import LogisticRegression

        from pcc.calibration.metrics import _unpenalised_logistic

        X = self._design(p)
        y = np.asarray(y, dtype=int).ravel()
        if len(np.unique(y)) < 2:
            self.a, self.b, self.c = 1.0, 1.0, 0.0
            return self
        lr = _unpenalised_logistic(LogisticRegression)
        lr.fit(X, y, sample_weight=sample_weight)
        self.a, self.b = (float(v) for v in lr.coef_.ravel())
        self.c = float(lr.intercept_[0])
        return self

    def transform(self, p):
        X = self._design(p)
        return _sigmoid(self.c + X @ np.array([self.a, self.b]))


class HierarchicalRecalibrator(Recalibrator):
    """Partially pooled per-stratum Platt scaling.

    Motivation: calibration may differ by pitch zone, competition or tracking
    source (RQ3, RQ4), but many strata are small. Fitting an independent map
    per stratum is noisy; fitting one global map ignores real heterogeneity.
    This estimator shrinks each stratum's Platt parameters toward the pooled
    estimate with a shrinkage weight

    .. math:: \\omega_s = \\frac{n_s}{n_s + \\kappa},

    which is the empirical-Bayes posterior mean under a normal-normal model
    with a variance ratio implied by ``kappa``. ``kappa`` is set on a grouped
    validation split rather than chosen by hand.

    This is deliberately a lightweight stand-in for a fully Bayesian
    hierarchical model. A properly specified random-effects logistic model
    (stratum-varying intercept and slope, fitted by MCMC) is the more defensible
    version and is listed as an optional extension in the proposal; it is not
    required for the minimum viable study, and the shrinkage estimator here
    keeps the whole pipeline dependency-free and fast enough to bootstrap.
    """

    name = "hierarchical_platt"

    def __init__(self, kappa: float = 500.0) -> None:
        self.kappa = float(kappa)
        self.global_map = PlattScaling()
        self.strata: dict = {}

    def fit(self, p, y, sample_weight=None, *, strata=None):
        p = np.asarray(p, dtype=float).ravel()
        y = np.asarray(y, dtype=int).ravel()
        self.global_map.fit(p, y, sample_weight)
        self.strata = {}
        if strata is None:
            return self
        strata = np.asarray(strata).ravel()
        for s in np.unique(strata):
            m = strata == s
            n_s = int(m.sum())
            local = PlattScaling().fit(p[m], y[m], None if sample_weight is None else np.asarray(sample_weight)[m])
            omega = n_s / (n_s + self.kappa)
            self.strata[s] = (
                omega * local.a + (1 - omega) * self.global_map.a,
                omega * local.b + (1 - omega) * self.global_map.b,
                n_s,
                omega,
            )
        return self

    def transform(self, p, strata=None):
        p = np.asarray(p, dtype=float).ravel()
        if strata is None or not self.strata:
            return self.global_map.transform(p)
        strata = np.asarray(strata).ravel()
        out = self.global_map.transform(p)
        for s, (a, b, _n, _w) in self.strata.items():
            m = strata == s
            if m.any():
                out[m] = _sigmoid(a + b * _logit(p[m]))
        return out


RECALIBRATORS = {
    "identity": IdentityRecalibrator,
    "platt": PlattScaling,
    "isotonic": IsotonicRecalibrator,
    "beta": BetaCalibration,
    "hierarchical_platt": HierarchicalRecalibrator,
}
