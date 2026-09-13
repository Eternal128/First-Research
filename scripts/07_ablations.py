#!/usr/bin/env python3
"""Robustness and ablations (Section 17).

Sweeps the assumptions that the main analysis holds fixed, and reports the
spread of the headline calibration statistic across each sweep. The purpose is
adversarial: if the range across plausible preprocessing choices is comparable
to the difference between models, the model comparison is not identified by the
data, and the paper must say so rather than quietly reporting the baseline
configuration.

Axes swept:
  - time-to-point model (constant speed vs bounded acceleration)
  - sprint speed, reaction time, arrival-time uncertainty, control rate
  - velocity removed; acceleration bound removed; nearest-distance only
  - control horizon
  - physics-model integration step
  - assumed pitch dimensions
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import banner, load_config, outdir, save_table, seed_everything, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.calibration import brier_score, calibration_slope_intercept, corp_decomposition  # noqa: E402
from pcc.geometry import Pitch  # noqa: E402
from pcc.kinematics import LocomotionParams  # noqa: E402
from pcc.models.features import FeatureConfig  # noqa: E402
from pcc.models.geometric import PhysicalControl, VoronoiControl  # noqa: E402
from pcc.models.statistical import LogisticControl  # noqa: E402


def evaluate(model, frames, y, *, fit_idx=None, test_idx=None) -> dict:
    if model.requires_fitting:
        model.fit([frames[i] for i in fit_idx], y[fit_idx])
    idx = test_idx if test_idx is not None else np.arange(len(frames))
    p = np.asarray(model.predict([frames[i] for i in idx]), dtype=float)
    yt = y[idx]
    corp = corp_decomposition(yt, p)
    slope = calibration_slope_intercept(yt, p)
    return {
        "brier": brier_score(yt, p),
        "corp_mcb": corp.miscalibration,
        "corp_dsc": corp.discrimination,
        "calibration_slope": slope["slope"],
        "mean_forecast": float(p.mean()),
        "forecast_sd": float(p.std()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="simulated")
    ap.add_argument("--config", default="default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_state", 0))
    banner("Ablations")

    if args.source != "simulated":
        print("ERROR: ablations need frames; implement the provider loader first.")
        return 1

    from pcc.data import SimulationConfig, filter_arrivals
    from pcc.data.preprocess import PreprocessConfig, Provenance
    from pcc.data.synthetic import simulate_dataset
    from pcc.evaluation import add_subgroups, grouped_holdout

    sim = SimulationConfig(
        n_matches=cfg["simulation"]["n_matches"],
        arrivals_per_match=cfg["simulation"]["arrivals_per_match"],
        random_state=cfg.get("random_state", 0),
    )
    frames, arrivals = simulate_dataset(sim)
    filtered = filter_arrivals(arrivals, PreprocessConfig(**cfg.get("preprocess", {})), provenance=Provenance())
    keep = arrivals["arrival_id"].isin(filtered["arrival_id"]).to_numpy()
    frames = [f for f, k in zip(frames, keep) if k]
    df = add_subgroups(filtered).reset_index(drop=True)
    y = df["y_control"].to_numpy(dtype=int)
    split = grouped_holdout(df, random_state=cfg.get("random_state", 0))
    print(f"  {len(frames)} arrivals; test fold {split.test.size}")

    base = LocomotionParams(**{k: v for k, v in cfg["locomotion"].items() if k != "tti_model"})
    tti = cfg["locomotion"]["tti_model"]
    rows: list[dict] = []

    def record(axis, value, model_name, metrics):
        rows.append({"axis": axis, "value": str(value), "model": model_name, **metrics})
        print(f"    {axis:22s} {str(value):16s} {model_name:18s} "
              f"brier={metrics['brier']:.4f} mcb={metrics['corp_mcb']:+.4f} "
              f"slope={metrics['calibration_slope']:.3f}")

    print("\n  [time-to-point model]")
    for t in ("constant_speed", "bounded_accel"):
        record("tti_model", t, "M2_physical",
               evaluate(PhysicalControl(base, tti_model=t), frames, y, test_idx=split.test))
        record("tti_model", t, "M1_voronoi",
               evaluate(VoronoiControl(base, tti_model=t), frames, y, test_idx=split.test))

    print("\n  [locomotion parameters]")
    sweeps = {
        "v_max": [4.0, 5.0, 6.0, 7.0],
        "reaction_time": [0.0, 0.3, 0.7, 1.0],
        "sigma_tti": [0.2, 0.45, 0.8, 1.2],
        "lambda_control": [2.0, 4.3, 8.0],
        "a_max": [4.0, 7.0, 12.0],
    }
    for param, values in sweeps.items():
        for v in values:
            record(param, v, "M2_physical",
                   evaluate(PhysicalControl(base.replace(**{param: v}), tti_model=tti),
                            frames, y, test_idx=split.test))

    print("\n  [integration step for the physics model]")
    for dt in (0.02, 0.04, 0.1, 0.25):
        record("integration_dt", dt, "M2_physical",
               evaluate(PhysicalControl(base, tti_model=tti, dt=dt), frames, y, test_idx=split.test))

    print("\n  [renormalisation of unassigned probability mass]")
    for renorm in (True, False):
        record("renormalise", renorm, "M2_physical",
               evaluate(PhysicalControl(base, tti_model=tti, renormalise=renorm),
                        frames, y, test_idx=split.test))

    print("\n  [information ablations on the logistic baseline]")
    for label, fc in {
        "full": FeatureConfig(params=base, tti_model=tti),
        "no_velocity": FeatureConfig(params=base, tti_model=tti, use_velocity=False),
        "no_reaction_time": FeatureConfig(params=base, tti_model=tti, use_reaction_time=False),
        "nearest_distance_only": FeatureConfig(params=base, tti_model=tti, nearest_distance_only=True),
    }.items():
        record("features", label, "M3_logistic",
               evaluate(LogisticControl(feature_config=fc), frames, y,
                        fit_idx=split.train, test_idx=split.test))

    print("\n  [assumed pitch dimensions]")
    for L, W in ((100.0, 64.0), (105.0, 68.0), (110.0, 72.0)):
        fc = FeatureConfig(params=base, tti_model=tti, pitch=Pitch(L, W))
        record("pitch_dims", f"{L:.0f}x{W:.0f}", "M3_logistic",
               evaluate(LogisticControl(feature_config=fc), frames, y,
                        fit_idx=split.train, test_idx=split.test))

    print("\n  [velocity-free physics model: the StatsBomb-freeze-frame regime]")
    zeroed = [
        type(f)(att_xy=f.att_xy, att_v=np.zeros_like(f.att_v), def_xy=f.def_xy,
                def_v=np.zeros_like(f.def_v), target=f.target, flight_time=f.flight_time,
                att_is_gk=f.att_is_gk, def_is_gk=f.def_is_gk, meta=f.meta)
        for f in frames
    ]
    record("velocity", "zeroed", "M2_physical",
           evaluate(PhysicalControl(base, tti_model=tti), zeroed, y, test_idx=split.test))

    table = pd.DataFrame(rows)
    out = outdir(f"ablations/{args.source}", config=cfg)
    write_manifest(out, script="07_ablations.py", config=cfg)
    save_table(table, out, "ablations")

    print("\n  [spread of the headline statistic within each axis]")
    spread = (
        table.groupby(["axis", "model"])
        .agg(n=("brier", "size"), brier_range=("brier", lambda s: s.max() - s.min()),
             mcb_range=("corp_mcb", lambda s: s.max() - s.min()),
             slope_range=("calibration_slope", lambda s: s.max() - s.min()))
        .reset_index()
        .sort_values("mcb_range", ascending=False)
    )
    print(spread.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    save_table(spread, out, "ablation_spread")
    print("\n  Compare the largest mcb_range above against the between-model MCB differences")
    print("  in results/main/. If they are of the same order, the model ranking is an")
    print("  artefact of the assumed locomotion envelope, not a finding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
