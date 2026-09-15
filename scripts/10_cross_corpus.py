#!/usr/bin/env python3
"""Run the same protocol across every available corpus and compare (RQ4, H4).

This is the artefact that three adapters exist for: one table in which the same
models, the same outcome definition and the same evaluation protocol are applied
to optical tracking, broadcast tracking and freeze frames, so that differences
are attributable to the data rather than to three different codebases.

    python scripts/10_cross_corpus.py
    python scripts/10_cross_corpus.py --sources metrica_sample skillcorner_open

**What this comparison can and cannot show.** The three corpora differ in
competition, season and provider as well as in tracking modality, so a
difference between them confounds measurement quality with the football being
played. The confound is not removable by any amount of care here; it is what the
degradation-simulation ablation and the *within*-corpus completeness contrast
exist to address. This script therefore reports the contrast and prints the
caveat alongside it rather than presenting it as an RQ4 answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    DATA_RAW, banner, load_config, load_corpus, outdir, save_figure, save_table,
    seed_everything, write_manifest,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.calibration import design_effect  # noqa: E402
from pcc.evaluation import EvaluationConfig, grouped_holdout, run_protocol  # noqa: E402
from pcc.models import build_model  # noqa: E402

DEFAULT_SOURCES = ["metrica_sample", "skillcorner_open", "statsbomb_open"]

#: Modality label per source, for the comparison table.
MODALITY = {
    "metrica_sample": "optical",
    "skillcorner_open": "broadcast",
    "statsbomb_open": "freeze_frame",
    "simulated": "simulated",
}

#: Below this many matches a corpus cannot support a confidence interval, and
#: its rows are reported as a demonstration rather than an estimate.
MIN_MATCHES_FOR_INFERENCE = 10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_state", 0))
    banner("Cross-corpus comparison")

    eval_cfg = EvaluationConfig(
        n_bins=cfg["evaluation"]["n_bins"], n_boot=args.n_boot,
        recalibrators=tuple(cfg["evaluation"]["recalibrators"]),
        random_state=cfg.get("random_state", 0),
    )

    metric_rows, corpus_rows, predictions = [], [], {}
    for source in args.sources:
        root = DATA_RAW / source
        if source != "simulated" and not root.exists():
            print(f"  {source:18s} SKIPPED - not present at {root}")
            continue
        try:
            frames, df, _prov = load_corpus(source, cfg, root=str(root))
        except Exception as exc:
            print(f"  {source:18s} SKIPPED - {type(exc).__name__}: {exc}")
            continue

        y = df["y_control"].to_numpy(dtype=int)
        n_matches = int(df["match_id"].nunique())
        deff = design_effect(df.assign(_y=y), value_col="_y", cluster_col="match_id")
        underpowered = n_matches < MIN_MATCHES_FOR_INFERENCE

        print(f"\n  {source} ({MODALITY.get(source, '?')}): {len(df)} arrivals, "
              f"{n_matches} match(es), base rate {y.mean():.3f}"
              + ("   [UNDERPOWERED]" if underpowered else ""))

        split = grouped_holdout(
            df, group_col="match_id",
            test_frac=cfg["evaluation"]["test_frac"],
            valid_frac=cfg["evaluation"]["valid_frac"] if n_matches >= 3 else 0.0,
            random_state=cfg.get("random_state", 0),
        )
        models = [build_model(n, **cfg["models"].get("params", {}).get(n, {}))
                  for n in cfg["models"]["suite"]]
        results = run_protocol(models, frames, y, df, split, eval_cfg, bootstrap=False)

        metrics = results["metrics"].assign(
            source=source, modality=MODALITY.get(source, "unknown"),
            n_matches=n_matches, underpowered=underpowered,
        )
        metric_rows.append(metrics)
        predictions[source] = results["predictions"]

        corpus_rows.append({
            "source": source,
            "modality": MODALITY.get(source, "unknown"),
            "n_arrivals": len(df),
            "n_matches": n_matches,
            "n_competitions": int(df["competition"].nunique()),
            "base_rate": float(y.mean()),
            "censored_frac": float(df["outcome_censored"].mean()),
            "median_frame_completeness": float(df["frame_completeness"].median()),
            "mean_progression_m": float((df["dest_x"] - df["origin_x"]).mean()),
            "median_pass_length_m": float(df["pass_length"].median()),
            "median_flight_s": float(df["flight_time"].median()),
            "frac_exogenous": float(1.0 - df["is_endogenous"].mean()),
            "flight_imputed_frac": float(df["flight_time_imputed"].mean()),
            "icc": deff["icc"],
            "design_effect": deff["design_effect"],
            "underpowered": underpowered,
        })

    if not metric_rows:
        print("\nNo corpora available. Fetch data with scripts/fetch_data.py.")
        return 1

    out = outdir("cross_corpus", config=cfg)
    write_manifest(out, script="10_cross_corpus.py", config={**cfg, "sources": args.sources})

    corpora = pd.DataFrame(corpus_rows)
    save_table(corpora, out, "corpus_summary")
    print("\n  [corpus characteristics]")
    print(corpora[["source", "modality", "n_arrivals", "n_matches", "base_rate",
                   "median_frame_completeness", "mean_progression_m", "frac_exogenous"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    metrics = pd.concat(metric_rows, ignore_index=True)
    save_table(metrics, out, "metrics_by_source")

    raw = metrics[metrics["variant"] == "raw"]
    for column, label in (("corp_mcb", "CORP miscalibration (lower is better)"),
                          ("calibration_slope", "calibration slope (1.0 is calibrated)"),
                          ("auc", "AUC (blind to calibration)")):
        pivot = raw.pivot_table(index="model", columns="source", values=column)
        print(f"\n  [{label}]")
        print(pivot.to_string(float_format=lambda v: f"{v:8.4f}"))
        save_table(pivot.reset_index(), out, f"by_source_{column}")

    # The study's central figure in table form: discrimination says one thing,
    # calibration another, and they disagree in the same direction everywhere.
    physics = raw[raw["model"] == "M2_physical"]
    if not physics.empty:
        print("\n  [the headline contrast, physics-based model]")
        print(physics[["source", "modality", "n_matches", "auc", "brier",
                       "corp_mcb", "calibration_slope"]]
              .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    try:
        from pcc.viz import reliability_diagram

        for source, preds in predictions.items():
            col = "M2_physical__raw"
            if col in preds.columns:
                fig = reliability_diagram(
                    preds["y"].to_numpy(), {f"M2_physical ({MODALITY.get(source)})": preds[col].to_numpy()},
                    n_bins=10, title=f"Physics-based control: {source}",
                )
                save_figure(fig, out, f"reliability_{source}")
    except Exception as exc:  # pragma: no cover
        print(f"  figures skipped: {exc}")

    print("\n  CAVEAT: these corpora differ in competition, season and provider as well")
    print("  as in tracking modality. Any difference between them confounds measurement")
    print("  quality with the football being played, and none of them has enough matches")
    print("  for a confidence interval. This is a demonstration that one protocol runs")
    print("  across three data shapes - not an answer to RQ4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
