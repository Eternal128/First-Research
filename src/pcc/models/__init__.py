"""The candidate control models M0-M5."""

from pcc.models.base import ControlModel, GeometricModel
from pcc.models.geometric import PhysicalControl, ReachabilitySigmoid, VoronoiControl
from pcc.models.statistical import GBMControl, LogisticControl, MarginalBaseline

__all__ = [
    "ControlModel",
    "GeometricModel",
    "VoronoiControl",
    "ReachabilitySigmoid",
    "PhysicalControl",
    "LogisticControl",
    "GBMControl",
    "MarginalBaseline",
    "build_model",
    "default_model_suite",
]

_REGISTRY = {
    "M0_marginal": MarginalBaseline,
    "M1_voronoi": VoronoiControl,
    "M2_physical": PhysicalControl,
    "M2a_reach_sigmoid": ReachabilitySigmoid,
    "M3_logistic": LogisticControl,
    "M4_gbm": GBMControl,
}


def build_model(name: str, **kwargs) -> ControlModel:
    """Instantiate a model by its registry name.

    ``M5_deepset`` is resolved lazily so that the package imports without
    PyTorch installed.
    """
    if name == "M5_deepset":
        from pcc.models.neural import DeepSetControl

        return DeepSetControl(**kwargs)
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY) + ['M5_deepset']}")
    return _REGISTRY[name](**kwargs)


def default_model_suite(include_neural: bool = False) -> list[ControlModel]:
    """The model suite used by the main analysis (Models M0-M4, optionally M5)."""
    models: list[ControlModel] = [
        MarginalBaseline(),
        VoronoiControl(),
        PhysicalControl(),
        ReachabilitySigmoid(),
        LogisticControl(),
        GBMControl(),
    ]
    if include_neural:
        from pcc.models.neural import DeepSetControl

        models.append(DeepSetControl())
    return models
