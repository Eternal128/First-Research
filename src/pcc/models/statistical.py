"""Models 3-5: fitted statistical and neural baselines.

These models are the adversary in the study's central comparison. If a
penalised logistic regression on a handful of kinematic features is better
calibrated *and* no less sharp than a physics-based pitch-control model, then
the physics model's principal claim - that its output is a probability because
it was derived from mechanics - is not doing work that a two-feature
regression could not do.

A deliberate asymmetry: the fitted models are given the chance to be
well-calibrated by construction (a logistic link fitted by maximum likelihood
is calibrated in-sample by the score equations). The interesting question is
therefore *out-of-sample and out-of-distribution* calibration, which is why the
evaluation protocol in :mod:`pcc.evaluation` never scores a model on data from
a match it was fitted on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pcc.data.schema import ArrivalFrame
from pcc.models.base import ControlModel
from pcc.models.features import FeatureConfig, build_feature_matrix


class LogisticControl(ControlModel):
    """Model 3: penalised logistic regression on engineered features.

    .. math::
        \\operatorname{logit} P(Y = 1 \\mid \\mathbf{z}) = \\beta_0 +
        \\boldsymbol{\\beta}^\\top \\mathbf{z}

    Features are standardised and an L2 penalty is applied. The penalty is a
    small but real threat to calibration - shrinkage flattens the linear
    predictor and therefore pushes the calibration slope above one - so the
    penalty strength is selected by *grouped* cross-validation on log loss,
    not on accuracy, and the resulting slope is reported rather than assumed.
    """

    name = "M3_logistic"
    requires_fitting = True

    def __init__(
        self,
        *,
        feature_config: FeatureConfig | None = None,
        C: float = 1.0,
        max_iter: int = 2000,
        random_state: int = 0,
    ):
        self.feature_config = feature_config or FeatureConfig()
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._pipeline = None

    def _build(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.C, max_iter=self.max_iter, random_state=self.random_state
                    ),
                ),
            ]
        )

    def fit(self, frames: Sequence[ArrivalFrame], y, *, sample_weight=None):
        X = build_feature_matrix(frames, self.feature_config)
        self._pipeline = self._build()
        kw = {}
        if sample_weight is not None:
            kw["clf__sample_weight"] = np.asarray(sample_weight, dtype=float)
        self._pipeline.fit(X, np.asarray(y, dtype=int), **kw)
        return self

    def predict(self, frames: Sequence[ArrivalFrame]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError(f"{self.name} must be fitted before prediction")
        X = build_feature_matrix(frames, self.feature_config)
        return self._pipeline.predict_proba(X)[:, 1]

    def control(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        if targets is None:
            return self.predict([frame])
        # Evaluating a grid requires a synthetic frame per target; this is used
        # for spatial maps only and is intentionally not vectorised, because the
        # primary evaluation never needs it.
        out = []
        for t in np.atleast_2d(targets):
            stub = ArrivalFrame(
                att_xy=frame.att_xy, att_v=frame.att_v, def_xy=frame.def_xy, def_v=frame.def_v,
                target=t, flight_time=float(self._flight_times(frame, t[None, :])[0]),
                att_is_gk=frame.att_is_gk, def_is_gk=frame.def_is_gk,
                att_vmax=frame.att_vmax, def_vmax=frame.def_vmax, meta=frame.meta,
            )
            out.append(float(self.predict([stub])[0]))
        return np.asarray(out, dtype=float)

    @property
    def coefficients(self) -> dict[str, float] | None:
        """Standardised coefficients, for the interpretability column of Table 9.1."""
        if self._pipeline is None:
            return None
        from pcc.models.features import FEATURE_NAMES

        clf = self._pipeline.named_steps["clf"]
        coefs = clf.coef_.ravel()
        names = list(FEATURE_NAMES) + [f"missing_indicator_{i}" for i in range(len(coefs) - len(FEATURE_NAMES))]
        return {"intercept": float(clf.intercept_[0]), **{n: float(c) for n, c in zip(names, coefs)}}


class GBMControl(ControlModel):
    """Model 4: gradient-boosted trees on the same feature set.

    Uses ``sklearn.ensemble.HistGradientBoostingClassifier`` so the study has
    no hard dependency on LightGBM or XGBoost; the loss is the log loss, which
    is a strictly proper scoring rule, so the model is optimising the right
    objective for a probability forecast.

    Expected failure mode worth pre-registering: boosted trees fitted to
    convergence on a strictly proper loss are usually *well* calibrated in
    distribution but degrade sharply under covariate shift, because the
    piecewise-constant fit cannot extrapolate beyond the training support. That
    makes M4 the natural carrier of the RQ4 (optical vs broadcast) and
    cross-competition contrasts.
    """

    name = "M4_gbm"
    requires_fitting = True

    def __init__(
        self,
        *,
        feature_config: FeatureConfig | None = None,
        max_iter: int = 400,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 50,
        l2_regularization: float = 1.0,
        early_stopping: bool = True,
        random_state: int = 0,
    ):
        self.feature_config = feature_config or FeatureConfig()
        self.kwargs = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=early_stopping,
            random_state=random_state,
        )
        self._model = None

    def fit(self, frames: Sequence[ArrivalFrame], y, *, sample_weight=None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        X = build_feature_matrix(frames, self.feature_config)
        self._model = HistGradientBoostingClassifier(**self.kwargs)
        self._model.fit(X, np.asarray(y, dtype=int), sample_weight=sample_weight)
        return self

    def predict(self, frames: Sequence[ArrivalFrame]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name} must be fitted before prediction")
        X = build_feature_matrix(frames, self.feature_config)
        return self._model.predict_proba(X)[:, 1]

    def control(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        if targets is None:
            return self.predict([frame])
        out = []
        for t in np.atleast_2d(targets):
            stub = ArrivalFrame(
                att_xy=frame.att_xy, att_v=frame.att_v, def_xy=frame.def_xy, def_v=frame.def_v,
                target=t, flight_time=float(self._flight_times(frame, t[None, :])[0]),
                att_is_gk=frame.att_is_gk, def_is_gk=frame.def_is_gk,
                att_vmax=frame.att_vmax, def_vmax=frame.def_vmax, meta=frame.meta,
            )
            out.append(float(self.predict([stub])[0]))
        return np.asarray(out, dtype=float)


class MarginalBaseline(ControlModel):
    """The climatological forecast: always predict the training base rate.

    Included because it is the reference against which *resolution* is defined
    in the Brier decomposition. Any model that fails to beat it has no skill,
    and - a point the study should make explicitly - a model that predicts the
    base rate everywhere is *perfectly calibrated* while being useless. This is
    the concrete demonstration that calibration alone is not a sufficient
    criterion and must always be reported alongside sharpness.
    """

    name = "M0_marginal"
    requires_fitting = True

    def __init__(self) -> None:
        self.rate: float | None = None

    def fit(self, frames, y, *, sample_weight=None):
        y = np.asarray(y, dtype=float)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.rate = float(np.average(y, weights=w))
        return self

    def predict(self, frames) -> np.ndarray:
        if self.rate is None:
            raise RuntimeError("MarginalBaseline must be fitted before prediction")
        return np.full(len(list(frames)), self.rate, dtype=float)

    def control(self, frame: ArrivalFrame, targets: np.ndarray | None = None) -> np.ndarray:
        n = 1 if targets is None else np.atleast_2d(targets).shape[0]
        if self.rate is None:
            raise RuntimeError("MarginalBaseline must be fitted before prediction")
        return np.full(n, self.rate, dtype=float)
