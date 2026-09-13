#!/usr/bin/env python3
"""Does recalibration change the football metrics? (Section 16)

Takes the main analysis's predictions and propagates the raw and recalibrated
control forecasts into expected possession value, decision thresholds, and
space metrics, then reports how far each moves.

The reportable claim is the discrete one: what fraction of the passes a model
would have recommended does it no longer recommend once its probabilities are
honest? A mean-EPV shift of a few thousandths is a statistic; a changed
recommendation is a consequence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import banner, load_config, outdir, save_figure, save_table, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.downstream import (  # noqa: E402
    PositionalValueSurface, downstream_sensitivity, pass_value_comparison,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="simulated")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--model", default="M2_physical")
    args = ap.parse_args()

    cfg = load_config(args.config)
    banner(f"Downstream impact  |  {args.model}")

    root = Path(cfg["output"]["root"])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    preds = pd.read_csv(root / "main" / args.source / "tables" / "predictions.csv")
    arr_path = root / "main" / args.source / "tables" / "test_arrivals.csv"
    if not arr_path.exists():
        print(f"ERROR: {arr_path} not found. Re-run scripts/03_main_analysis.py.")
        return 1
    arrivals = pd.read_csv(arr_path)

    # Same alignment guard as scripts/04: a silent misalignment here would
    # attach the wrong destination to every forecast and quietly corrupt the
    # whole downstream comparison.
    merged = preds.merge(arrivals, on="arrival_index", how="left", suffixes=("", "_arr"))
    if len(merged) != len(preds) or merged["dest_x"].isna().any():
        print("ERROR: predictions and test arrivals are misaligned.")
        return 1
    if (merged["y_control"].astype(int) != merged["y"].astype(int)).any():
        print("ERROR: outcome mismatch between predictions and test arrivals.")
        return 1

    raw_col = f"{args.model}__raw"
    cal_cols = {v: f"{args.model}__{v}" for v in ("platt", "isotonic", "beta")
                if f"{args.model}__{v}" in merged.columns}
    if raw_col not in merged.columns:
        print(f"ERROR: {raw_col} not in predictions. Run scripts/03_main_analysis.py first.")
        return 1

    # The value surface is FITTED from the data at hand; nothing is hard-coded.
    # Fitting it on the same rows the downstream comparison uses is acceptable
    # here only because the surface is held identical across variants - it is a
    # common yardstick, not a predictor being evaluated.
    surface = PositionalValueSurface.fit_from_outcomes(
        arrivals[["dest_x", "dest_y"]].to_numpy(),
        arrivals["y_control"].to_numpy(),
        provenance=f"fitted on {len(arrivals)} arrivals from source={args.source}",
    )
    print(f"  value surface: {surface.grid.shape} cells, provenance: {surface.provenance}")

    out = outdir(f"downstream/{args.source}", config=cfg)
    write_manifest(out, script="06_downstream_analysis.py",
                   config={**cfg, "model": args.model},
                   extra={"value_surface_provenance": surface.provenance})

    control_cols = {"raw": raw_col, **cal_cols}
    epv = pass_value_comparison(merged, surface, control_cols=control_cols,
                                loss_discounts=tuple(cfg["downstream"]["loss_discounts"]))
    print("\n  [expected possession value by control variant]")
    print(epv.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    save_table(epv, out, "epv_by_variant")

    if "isotonic" in cal_cols:
        sens = downstream_sensitivity(
            merged, raw_col=raw_col, calibrated_col=cal_cols["isotonic"], surface=surface,
            thresholds=tuple(cfg["downstream"]["decision_thresholds"]),
        )
        print("\n  [decisions changed by recalibrating (isotonic)]")
        print(sens[["threshold", "frac_decisions_changed", "raw_act_rate", "cal_act_rate",
                    "accuracy_raw", "accuracy_cal"]].to_string(index=False,
                                                               float_format=lambda v: f"{v:8.4f}"))
        save_table(sens, out, "decision_displacement")
        worst = sens.loc[sens["frac_decisions_changed"].idxmax()]
        print(f"\n  At threshold {worst['threshold']:.2f}, recalibration changes "
              f"{worst['frac_decisions_changed']:.1%} of pass recommendations.")

    try:
        from pcc.evaluation import net_benefit
        from pcc.viz import decision_curve

        nb = net_benefit(merged["y"].to_numpy(), merged[raw_col].to_numpy())
        save_table(nb, out, "decision_curve_raw")
        fig = decision_curve(nb, title=f"{args.model} raw forecasts: net benefit")
        save_figure(fig, out, "decision_curve_raw")
        if "isotonic" in cal_cols:
            nb_c = net_benefit(merged["y"].to_numpy(), merged[cal_cols["isotonic"]].to_numpy())
            save_table(nb_c, out, "decision_curve_calibrated")
    except Exception as exc:  # pragma: no cover
        print(f"  figure skipped: {exc}")

    print("\n  Interpretation guard: a change in these numbers shows that the metric is")
    print("  SENSITIVE to calibration. Showing the recalibrated version is BETTER for")
    print("  decisions requires the decision-curve comparison, not the EPV shift alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
