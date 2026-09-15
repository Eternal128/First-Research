#!/usr/bin/env python3
"""Where does calibration break down? (RQ3, RQ4)

Consumes ``results/main/<source>/tables/predictions.csv`` and reports calibration
by pitch zone, pass-length band, pressure, game state, flight time, arrival type
and tracking source, with cluster-bootstrap intervals and a false-discovery-rate
adjustment across the pre-registered subgroup family.

The subgroups are those fixed in ``pcc.evaluation.subgroups``. Adding one after
seeing the results would invalidate the multiplicity correction, so the script
takes no subgroup arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import banner, load_config, outdir, save_figure, save_table, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.evaluation import spatial_calibration_map, stratified_metrics  # noqa: E402

SUBGROUPS = ("zone", "third", "pass_length_band", "pressure_band", "game_state",
             "flight_band", "arrival_type", "tracking_source", "quality_flag",
             # Freeze-frame corpora only: whether the camera actually covered the
             # destination, and the ball height. Both are pre-registered because
             # they are the axes on which the fallback design is expected to be
             # weakest, not because they were found to be interesting.
             "dest_visible", "pass_height",
             # Broadcast/freeze-frame corpora: how much of the frame was observed.
             "completeness_band")



def attach_arrivals(preds: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """Join the test-fold covariates onto the predictions, verifying alignment.

    The join key is ``arrival_index``, which both tables carry from the same
    split. The assertions below exist because a silent misalignment here would
    attach the wrong outcome and pitch location to every forecast and would not
    otherwise announce itself.
    """
    if "arrival_index" not in arrivals.columns:
        raise ValueError("test_arrivals.csv has no arrival_index column; re-run scripts/03.")
    merged = preds.merge(arrivals, on="arrival_index", how="left", suffixes=("", "_arr"))
    if len(merged) != len(preds):
        raise AssertionError(f"join changed row count: {len(preds)} -> {len(merged)}")
    if merged["arrival_index"].isna().any():
        raise AssertionError("some predictions have no matching arrival row")
    if "y_control" in merged.columns and "y" in merged.columns:
        mismatch = int((merged["y_control"].astype(int) != merged["y"].astype(int)).sum())
        if mismatch:
            raise AssertionError(
                f"outcome mismatch on {mismatch} rows: predictions and arrivals are misaligned"
            )
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="simulated")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--model", default="M2_physical", help="Model whose forecasts are analysed.")
    ap.add_argument("--variant", default="raw", choices=["raw", "platt", "isotonic", "beta"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    banner(f"Subgroup calibration  |  {args.model} ({args.variant})")

    main_dir = Path(cfg["output"]["root"])
    if not main_dir.is_absolute():
        main_dir = Path(__file__).resolve().parents[1] / main_dir
    pred_path = main_dir / "main" / args.source / "tables" / "predictions.csv"
    arr_path = main_dir / "main" / args.source / "tables" / "test_arrivals.csv"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found. Run scripts/03_main_analysis.py first.")
        return 1

    preds = pd.read_csv(pred_path)
    col = f"{args.model}__{args.variant}"
    if col not in preds.columns:
        print(f"ERROR: column {col!r} absent. Available: "
              f"{[c for c in preds.columns if c.startswith(args.model)]}")
        return 1

    if not arr_path.exists():
        print(f"ERROR: {arr_path} not found. Re-run scripts/03_main_analysis.py; it now "
              "writes the test-fold arrivals beside the predictions.")
        return 1
    preds = attach_arrivals(preds, pd.read_csv(arr_path))

    preds = preds.rename(columns={col: "p"})
    available = [g for g in SUBGROUPS if g in preds.columns]
    print(f"  {len(preds)} test arrivals; subgroups available: {available}")

    out = outdir(f"subgroups/{args.source}", config=cfg)
    write_manifest(out, script="04_subgroup_analysis.py",
                   config={**cfg, "model": args.model, "variant": args.variant})

    all_rows = []
    for group in available:
        tab = stratified_metrics(
            preds, prob_col="p", outcome_col="y", group_col=group,
            cluster_col="match_id", min_n=cfg["evaluation"]["min_subgroup_n"],
            n_boot=max(200, cfg["evaluation"]["n_boot"] // 2),
            random_state=cfg.get("random_state", 0),
        )
        tab.insert(0, "subgroup_variable", group)
        tab = tab.rename(columns={group: "level"})
        all_rows.append(tab)
        shown = tab[~tab["suppressed"]] if "suppressed" in tab else tab
        if not shown.empty:
            print(f"\n  [{group}]")
            print(shown[["level", "n", "base_rate", "mean_forecast", "corp_mcb",
                         "calibration_slope"]].to_string(index=False,
                                                         float_format=lambda v: f"{v:8.4f}"))

    combined = pd.concat(all_rows, ignore_index=True)
    save_table(combined, out, "subgroup_metrics")

    live = combined[~combined["suppressed"].fillna(False)].copy()
    if not live.empty and "mcb_lo" in live:
        # A stratum whose bootstrap interval for MCB excludes zero is flagged;
        # this is an interval-based screen, not a p-value, so the FDR routine is
        # applied to a conservative normal-approximation p-value derived from it.
        se = (live["mcb_hi"] - live["mcb_lo"]) / (2 * 1.96)
        z = live["corp_mcb"] / se.replace(0, np.nan)
        from scipy.stats import norm

        live["p_value"] = 2 * (1 - norm.cdf(np.abs(z)))
        from pcc.evaluation import adjust_multiplicity

        live = adjust_multiplicity(live, pvalue_col="p_value", method="fdr_bh")
        save_table(live, out, "subgroup_metrics_adjusted")
        flagged = live[live["p_value_adj"] < 0.05]
        print(f"\n  strata with FDR-adjusted evidence of miscalibration: {len(flagged)} of {len(live)}")
        if not flagged.empty:
            print(flagged[["subgroup_variable", "level", "n", "corp_mcb", "p_value_adj"]]
                  .to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    if {"dest_x", "dest_y"}.issubset(preds.columns):
        cells = spatial_calibration_map(preds, prob_col="p", outcome_col="y", min_n=30)
        save_table(cells, out, "spatial_calibration")
        try:
            from pcc.viz import spatial_gap_map

            fig = spatial_gap_map(cells, title=f"{args.model} ({args.variant}): observed - forecast")
            save_figure(fig, out, "spatial_gap_map")
        except Exception as exc:  # pragma: no cover
            print(f"  figure skipped: {exc}")

    if {"zone", "corp_mcb"}.issubset(combined.columns) or "level" in combined:
        try:
            from pcc.viz import calibration_gap_by_group

            zone_tab = combined[combined["subgroup_variable"] == "zone"]
            if not zone_tab.empty:
                fig = calibration_gap_by_group(zone_tab, group_col="level",
                                               title=f"{args.model}: CORP miscalibration by zone")
                save_figure(fig, out, "mcb_by_zone")
        except Exception as exc:  # pragma: no cover
            print(f"  figure skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
