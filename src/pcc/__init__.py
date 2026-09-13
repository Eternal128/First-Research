"""Pitch-control calibration study (``pcc``).

Research question
-----------------
Pitch-control models emit a field ``C_A(x, y, t) in [0, 1]`` that the
literature and applied practice read as *the probability that team A would
control the ball if it arrived at (x, y) at time t*. This package tests
whether that reading survives contact with realised ball arrivals.

Layout
------
``pcc.geometry``    pitch frame, normalisation, zoning
``pcc.kinematics``  smoothing, velocity/acceleration, time-to-point models
``pcc.models``      the five candidate control models (M1-M5)
``pcc.calibration`` proper scores, reliability, recalibration, cluster bootstrap
``pcc.evaluation``  leakage-safe splits, the evaluation protocol, decision curves
``pcc.selection``   the selection-bias layer (candidate arrivals, IPW, diagnostics)
``pcc.data``        schema contract, loaders, labelling, preprocessing, simulator
``pcc.viz``         reliability diagrams and spatial calibration maps

Nothing in this package produces or contains empirical findings about real
football. ``pcc.data.synthetic`` generates *simulated* data whose sole purpose
is to exercise the pipeline and to unit-test that the calibration machinery
detects miscalibration that was injected on purpose.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
