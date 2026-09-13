"""The evaluation protocol: fit, predict, score, compare - once, consistently.

This module encodes the study's pre-registered analysis order so that the same
sequence is applied to every model, every split and every subgroup. Freezing it
in code rather than describing it in prose is what prevents the analyst degrees
of freedom that an evaluation-focused paper is most vulnerable to: choosing a
bin count after seeing the curve, or a subgroup after seeing which one is
significant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from pcc.calibration.metrics import brier_score, full_report, log_loss
from pcc.calibration.bootstrap import cluster_bootstrap, paired_cluster_bootstrap
from pcc.calibration.recalibration import RECALIBRATORS, Recalibrator
from pcc.data.schema import ArrivalFrame
from pcc.evaluation.splits import Split, leakage_audit
from pcc.models.base import ControlModel


@dataclass
class EvaluationConfig:
    """Every analyst choice, fixed in one place and written to the results."""

    n_bins: int = 15
    n_boot: int = 1000
    bootstrap_level: float = 0.95
    cluster_col: str = "match_id"
    recalibrators: tuple[str, ...] = ("identity", "platt", "isotonic", "beta")
    primary_metric: str = "corp_mcb"
    primary_score: str = "brier"
    random_state: int = 0
    subgroup_cols: tuple[str, ...] = (
        "zone", "pass_length_band", "pressure_band", "game_state", "tracking_source", "arrival_type",
    )
    min_subgroup_n: int = 200
    log_loss_eps: float = 1e-6

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class ModelPredictions:
    """Raw and recalibrated test-set forecasts for one model on one split."""

    model_name: str
    split_name: str
    test_index: np.ndarray
    raw: np.ndarray
    recalibrated: dict[str, np.ndarray] = field(default_factory=dict)


def fit_predict(
    model: ControlModel,
    frames: Sequence[ArrivalFrame],
    y: np.ndarray,
    split: Split,
    *,
    sample_weight: np.ndarray | None = None,
) -> ModelPredictions:
    """Fit on the training fold (if the model fits) and predict on the test fold.

    Unfitted geometric models skip fitting entirely, which is the point of the
    comparison: they are evaluated on the test fold having never seen a
    football outcome.
    """
    frames = list(frames)
    y = np.asarray(y, dtype=int)

    if model.requires_fitting or isinstance(model, type(model)) and hasattr(model, "fit"):
        train_frames = [frames[i] for i in split.train]
        w = None if sample_weight is None else np.asarray(sample_weight)[split.train]
        model.fit(train_frames, y[split.train], sample_weight=w)

    test_frames = [frames[i] for i in split.test]
    raw = np.asarray(model.predict(test_frames), dtype=float)
    if raw.shape != (len(test_frames),):
        raise ValueError(f"{model.name} returned {raw.shape}, expected {(len(test_frames),)}")
    return ModelPredictions(model.name, split.name, split.test, np.clip(raw, 0.0, 1.0))


def apply_recalibration(
    preds: ModelPredictions,
    model: ControlModel,
    frames: Sequence[ArrivalFrame],
    y: np.ndarray,
    split: Split,
    *,
    names: Sequence[str] = ("identity", "platt", "isotonic", "beta"),
) -> ModelPredictions:
    """Fit each recalibration map on the *validation* fold and apply it to test.

    The validation fold is disjoint from both the fitting data and the test
    data. This is the single most important protocol detail in the RQ5
    analysis: a recalibration map fitted on the test fold makes every model
    look calibrated and would turn the entire result into an artefact.
    """
    if split.valid.size == 0:
        raise ValueError(
            f"split {split.name!r} has no validation fold; recalibration maps must not be "
            "fitted on the test fold (see proposal Section 13)"
        )
    frames = list(frames)
    y = np.asarray(y, dtype=int)
    valid_frames = [frames[i] for i in split.valid]
    p_valid = np.asarray(model.predict(valid_frames), dtype=float)
    y_valid = y[split.valid]

    for name in names:
        rc: Recalibrator = RECALIBRATORS[name]()
        rc.fit(p_valid, y_valid)
        preds.recalibrated[name] = np.clip(np.asarray(rc.transform(preds.raw), dtype=float), 0.0, 1.0)
    return preds


def score_predictions(
    preds: ModelPredictions,
    y: np.ndarray,
    meta: pd.DataFrame,
    config: EvaluationConfig,
    *,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """Full metric report for the raw forecast and each recalibrated variant."""
    y = np.asarray(y, dtype=int)
    y_test = y[preds.test_index]
    groups = meta.iloc[preds.test_index][config.cluster_col].to_numpy()
    w = None if sample_weight is None else np.asarray(sample_weight)[preds.test_index]

    rows = []
    variants = {"raw": preds.raw, **preds.recalibrated}
    for variant, p in variants.items():
        rep = full_report(y_test, p, w, groups, n_bins=config.n_bins)
        rows.append({"model": preds.model_name, "split": preds.split_name, "variant": variant, **rep})
    return pd.DataFrame(rows)


def bootstrap_primary(
    preds: ModelPredictions,
    y: np.ndarray,
    meta: pd.DataFrame,
    config: EvaluationConfig,
    *,
    variant: str = "raw",
) -> dict[str, float]:
    """Cluster-bootstrap interval for the primary score and calibration metric."""
    from pcc.calibration.metrics import corp_decomposition

    p = preds.raw if variant == "raw" else preds.recalibrated[variant]
    df = pd.DataFrame(
        {
            "y": np.asarray(y, dtype=int)[preds.test_index],
            "p": p,
            config.cluster_col: meta.iloc[preds.test_index][config.cluster_col].to_numpy(),
        }
    )
    score = cluster_bootstrap(
        lambda d: brier_score(d["y"], d["p"]),
        df,
        cluster_col=config.cluster_col,
        n_boot=config.n_boot,
        level=config.bootstrap_level,
        random_state=config.random_state,
    )
    mcb = cluster_bootstrap(
        lambda d: corp_decomposition(d["y"], d["p"]).miscalibration,
        df,
        cluster_col=config.cluster_col,
        n_boot=config.n_boot,
        level=config.bootstrap_level,
        random_state=config.random_state + 1,
    )
    return {
        "model": preds.model_name,
        "split": preds.split_name,
        "variant": variant,
        **score.as_dict("brier_"),
        **mcb.as_dict("mcb_"),
    }


def pairwise_comparison(
    predictions: Sequence[ModelPredictions],
    y: np.ndarray,
    meta: pd.DataFrame,
    config: EvaluationConfig,
    *,
    reference: str = "M3_logistic",
    variant: str = "raw",
) -> pd.DataFrame:
    """Paired cluster-bootstrap differences against a reference model.

    The reference defaults to the logistic baseline, because the study's
    sharpest hypothesis (H2) is stated as "the physics model is no better
    calibrated than a simple regression". Framing every comparison against that
    baseline keeps the analysis from degenerating into an unstructured
    leaderboard.
    """
    by_name = {p.model_name: p for p in predictions}
    if reference not in by_name:
        raise KeyError(f"reference model {reference!r} not among {sorted(by_name)}")
    ref = by_name[reference]

    rows = []
    for name, preds in by_name.items():
        if name == reference:
            continue
        if not np.array_equal(preds.test_index, ref.test_index):
            raise ValueError("paired comparison requires identical test folds")
        df = pd.DataFrame(
            {
                "y": np.asarray(y, dtype=int)[ref.test_index],
                "p_model": preds.raw if variant == "raw" else preds.recalibrated[variant],
                "p_ref": ref.raw if variant == "raw" else ref.recalibrated[variant],
                config.cluster_col: meta.iloc[ref.test_index][config.cluster_col].to_numpy(),
            }
        )
        res = paired_cluster_bootstrap(
            lambda d: brier_score(d["y"], d["p"]),
            df,
            col_a="p_model",
            col_b="p_ref",
            prob_col="p",
            cluster_col=config.cluster_col,
            n_boot=config.n_boot,
            level=config.bootstrap_level,
            random_state=config.random_state,
        )
        rows.append(
            {
                "model": name,
                "reference": reference,
                "variant": variant,
                "delta_brier": res.point,
                "lo": res.lo,
                "hi": res.hi,
                "se": res.se,
                "favours": "model" if res.hi < 0 else ("reference" if res.lo > 0 else "inconclusive"),
            }
        )
    return pd.DataFrame(rows)


def run_protocol(
    models: Sequence[ControlModel],
    frames: Sequence[ArrivalFrame],
    y: np.ndarray,
    meta: pd.DataFrame,
    split: Split,
    config: EvaluationConfig | None = None,
    *,
    sample_weight: np.ndarray | None = None,
    bootstrap: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the complete protocol for one split and return tidy result tables.

    Returns keys ``metrics``, ``comparisons``, ``bootstrap``, ``leakage``,
    ``predictions``. Everything a results section needs, and nothing computed
    on data the protocol says it should not have seen.
    """
    config = config or EvaluationConfig()
    frames = list(frames)
    y = np.asarray(y, dtype=int)

    all_preds: list[ModelPredictions] = []
    metric_rows: list[pd.DataFrame] = []
    boot_rows: list[dict] = []

    for model in models:
        preds = fit_predict(model, frames, y, split, sample_weight=sample_weight)
        if split.valid.size > 0:
            preds = apply_recalibration(preds, model, frames, y, split, names=config.recalibrators)
        metric_rows.append(score_predictions(preds, y, meta, config, sample_weight=sample_weight))
        if bootstrap:
            boot_rows.append(bootstrap_primary(preds, y, meta, config))
        all_preds.append(preds)

    pred_frame = pd.DataFrame({"arrival_index": split.test})
    for p in all_preds:
        pred_frame[f"{p.model_name}__raw"] = p.raw
        for k, v in p.recalibrated.items():
            if k != "identity":
                pred_frame[f"{p.model_name}__{k}"] = v
    pred_frame["y"] = y[split.test]
    for col in (config.cluster_col, "possession_id", "competition", "tracking_source"):
        if col in meta.columns:
            pred_frame[col] = meta.iloc[split.test][col].to_numpy()

    names = {m.name for m in models}
    reference = "M3_logistic" if "M3_logistic" in names else sorted(names)[0]

    return {
        "metrics": pd.concat(metric_rows, ignore_index=True),
        "comparisons": pairwise_comparison(all_preds, y, meta, config, reference=reference)
        if bootstrap and len(all_preds) > 1
        else pd.DataFrame(),
        "bootstrap": pd.DataFrame(boot_rows),
        "leakage": leakage_audit(meta, split),
        "predictions": pred_frame,
    }
