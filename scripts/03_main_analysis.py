#!/usr/bin/env python3
"""The primary analysis: model comparison, calibration, and recalibration.

Answers RQ1, RQ2 and RQ5 on whichever dataset is supplied. Everything it does is
fixed by ``configs/default.yaml`` and the protocol in
``pcc.evaluation.protocol``; there are no decisions taken inside this script
that are not recorded in the manifest it writes.

    python scripts/03_main_analysis.py --source simulated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import RESULTS, banner, load_config, outdir, save_figure, save_table, seed_everything, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.calibration import design_effect  # noqa: E402
from pcc.evaluation import EvaluationConfig, grouped_holdout, run_protocol  # noqa: E402
from pcc.models import build_model  # noqa: E402


def load_dataset(source: str, cfg: dict):
    """Rebuild frames and table. Simulated sources regenerate; real ones load the cache."""
    from pcc.data import SimulationConfig, simulate_dataset
    from pcc.data.preprocess import PreprocessConfig, Provenance
    from pcc.data import filter_arrivals
    from pcc.evaluation import add_subgroups

    if source == "simulated":
        sim = SimulationConfig(
            n_matches=cfg["simulation"]["n_matches"],
            arrivals_per_match=cfg["simulation"]["arrivals_per_match"],
            random_state=cfg.get("random_state", 0),
        )
        frames, arrivals = simulate_dataset(sim)
        pp = PreprocessConfig(**cfg.get("preprocess", {}))
        prov = Provenance()
        filtered = filter_arrivals(arrivals, pp, provenance=prov)
        keep = arrivals["arrival_id"].isin(filtered["arrival_id"]).to_numpy()
        frames = [f for f, k in zip(frames, keep) if k]
        return frames, add_subgroups(filtered).reset_index(drop=True)

    raise SystemExit(
        f"Source {source!r} has no completed loader. Build it with scripts/02_build_dataset.py "
        "once the provider adapter in src/pcc/data/loaders.py is finished."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="simulated")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--include-neural", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_state", 0))
    banner(f"Main analysis  |  source={args.source}")

    frames, df = load_dataset(args.source, cfg)
    y = df["y_control"].to_numpy(dtype=int)
    print(f"  {len(df)} arrivals, {df['match_id'].nunique()} matches, base rate {y.mean():.3f}")

    deff = design_effect(df.assign(_y=y), value_col="_y", cluster_col="match_id")
    print(f"  intra-match ICC {deff['icc']:.4f}, design effect {deff['design_effect']:.2f} "
          f"-> naive standard errors would be ~{np.sqrt(max(deff['design_effect'],1)):.1f}x too small")

    eval_cfg = EvaluationConfig(
        n_bins=cfg["evaluation"]["n_bins"],
        n_boot=args.n_boot if args.n_boot is not None else cfg["evaluation"]["n_boot"],
        recalibrators=tuple(cfg["evaluation"]["recalibrators"]),
        random_state=cfg.get("random_state", 0),
    )

    model_names = list(cfg["models"]["suite"])
    if args.include_neural and "M5_deepset" not in model_names:
        model_names.append("M5_deepset")
    models = []
    for name in model_names:
        try:
            models.append(build_model(name, **cfg["models"].get("params", {}).get(name, {})))
        except ImportError as exc:
            print(f"  skipping {name}: {exc}")
    print(f"  models: {[m.name for m in models]}")

    split = grouped_holdout(
        df,
        group_col=cfg["evaluation"]["split_group"],
        test_frac=cfg["evaluation"]["test_frac"],
        valid_frac=cfg["evaluation"]["valid_frac"],
        random_state=cfg.get("random_state", 0),
        stratify_col=cfg["evaluation"].get("stratify_col"),
    )
    print(f"  split sizes: {split.sizes()}")

    results = run_protocol(models, frames, y, df, split, eval_cfg)

    out = outdir(f"main/{args.source}", config=cfg)
    write_manifest(
        out, script="03_main_analysis.py",
        config={**cfg, "source": args.source, "models": model_names},
        extra={"design_effect": deff, "split_sizes": split.sizes()},
    )
    for key, table in results.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            save_table(table, out, key)

    # The test-fold arrivals are written here, beside the predictions they
    # correspond to, rather than being re-joined later against a separately
    # built dataset. A join across two builds can silently misalign if the two
    # were produced with different parameters, and a misaligned outcome column
    # would corrupt every downstream statistic without raising anything.
    test_arrivals = df.iloc[split.test].reset_index(drop=True)
    test_arrivals.insert(0, "arrival_index", split.test)
    save_table(test_arrivals, out, "test_arrivals")

    metrics = results["metrics"]
    print("\n  headline (raw forecasts, test fold):")
    headline = metrics[metrics["variant"] == "raw"][
        ["model", "n", "base_rate", "brier", "corp_mcb", "corp_dsc", "calibration_slope", "auc", "forecast_sd"]
    ].sort_values("brier")
    print(headline.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\n  recalibration (change in Brier score vs the model's own raw forecast):")
    pivot = metrics.pivot_table(index="model", columns="variant", values="brier")
    if "raw" in pivot:
        delta = pivot.sub(pivot["raw"], axis=0).drop(columns=["raw"], errors="ignore")
        print(delta.to_string(float_format=lambda v: f"{v:+8.4f}"))

    try:
        from pcc.viz import reliability_diagram

        preds = results["predictions"]
        raw_cols = {c.split("__")[0]: preds[c].to_numpy() for c in preds.columns if c.endswith("__raw")}
        fig = reliability_diagram(preds["y"].to_numpy(), raw_cols,
                                  n_bins=cfg["evaluation"]["n_bins"],
                                  title=f"Reliability, raw forecasts ({args.source})")
        save_figure(fig, out, "reliability_raw")
    except Exception as exc:  # pragma: no cover
        print(f"  figure skipped: {exc}")

    print(f"\n  results in {out}")
    if args.source == "simulated":
        print("\n  REMINDER: simulated data. These numbers describe the simulator, not football.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
