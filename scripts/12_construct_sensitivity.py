#!/usr/bin/env python3
"""Does the conclusion survive the definition of control? (Section 15.10)

The study imposes an operational definition of an undefined construct: team A in
possession ``h`` seconds after the ball arrives. That choice is defended in
Section 7.3, but a defence is not evidence. This script re-labels the corpus
under every pre-registered horizon and every candidate outcome definition and
re-runs the headline comparison on each.

The decision rule was fixed in advance and is applied here without amendment:

> **A conclusion that flips across this range is reported as a conclusion that
> flips.** If the physics model is miscalibrated at ``h = 1`` and calibrated at
> ``h = 2``, then "pitch control is miscalibrated" is under-specified and the
> paper must say so rather than choosing the horizon that supports it.

    python scripts/12_construct_sensitivity.py --source statsbomb_open --max-matches 64

Each (definition, horizon) pair re-labels from the raw data, so the first run is
slow; results are cached per setting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    DATA_RAW, banner, load_config, load_corpus, outdir, save_table, seed_everything,
    write_manifest,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.calibration import (  # noqa: E402
    brier_score, calibration_slope_intercept, corp_decomposition, roc_auc, spherical_score,
)
from pcc.evaluation import grouped_holdout  # noqa: E402
from pcc.models import build_model  # noqa: E402

#: Pre-registered horizons (proposal Section 7.3) and outcome definitions
#: (Section 8.3). Fixed before any result was seen.
HORIZONS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
DEFINITIONS = ["controlled_at_horizon", "first_touch", "retained_sequence", "possession_at_5s"]



def _censored_fraction(provenance) -> float:
    """Share of extracted arrivals dropped for overlapping a stoppage.

    Taken from the preprocessing provenance, not from the surviving table: the
    filter removes these rows, so the retained frame always reports zero.
    """
    steps = {row["step"]: row for row in provenance.steps}
    dropped = steps.get("drop_censored", {}).get("dropped")
    summary = steps.get("summary", {})
    n_in = summary.get("n_in")
    if dropped is None or not n_in:
        return float("nan")
    return float(dropped) / float(n_in)


def evaluate(models, frames, y, df, cfg, random_state: int) -> list[dict]:
    split = grouped_holdout(df, group_col="match_id",
                            test_frac=cfg["evaluation"]["test_frac"],
                            valid_frac=cfg["evaluation"]["valid_frac"],
                            random_state=random_state)
    rows = []
    for name in models:
        model = build_model(name, **cfg["models"].get("params", {}).get(name, {}))
        if model.requires_fitting:
            model.fit([frames[i] for i in split.train], y[split.train])
        p = np.asarray(model.predict([frames[i] for i in split.test]), dtype=float)
        yt = y[split.test]
        corp = corp_decomposition(yt, p)
        slope = calibration_slope_intercept(yt, p)
        rows.append({
            "model": name,
            "n_test": int(yt.size),
            "base_rate": float(yt.mean()),
            "brier": brier_score(yt, p),
            "spherical": spherical_score(yt, p),
            "corp_mcb": corp.miscalibration,
            "corp_dsc": corp.discrimination,
            "corp_unc": corp.uncertainty,
            "calibration_slope": slope["slope"],
            "auc": roc_auc(yt, p),
            "mean_forecast": float(p.mean()),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="statsbomb_open")
    ap.add_argument("--root", default=None)
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--max-matches", type=int, default=64,
                    help="Subset the corpus; the sweep re-parses once per setting.")
    ap.add_argument("--models", nargs="+", default=["M2_physical", "M3_logistic"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("random_state", 0)
    seed_everything(seed)
    banner("Construct sensitivity: horizon and outcome definition")
    print("  Pre-registered rule: a conclusion that flips across this sweep is")
    print("  reported as a conclusion that flips.\n")

    root = args.root or str(DATA_RAW / args.source)
    loader_kwargs = {"max_matches": args.max_matches} if args.max_matches else {}

    rows = []
    for definition in DEFINITIONS:
        horizons = HORIZONS if definition == "controlled_at_horizon" else [
            cfg.get("labels", {}).get("horizon", 1.0)
        ]
        for horizon in horizons:
            overrides = {"definition": definition, "horizon": horizon}
            try:
                frames, df, prov = load_corpus(
                    args.source, cfg, root=root, label_overrides=overrides,
                    loader_kwargs=loader_kwargs,
                )
            except Exception as exc:
                print(f"  {definition} h={horizon}: FAILED - {type(exc).__name__}: {exc}")
                continue

            y = df["y_control"].to_numpy(dtype=int)
            if len(np.unique(y)) < 2:
                print(f"  {definition} h={horizon}: degenerate outcome, skipped")
                continue

            # Censored arrivals are *removed* by the inclusion criteria, so the
            # rate must be read from the preprocessing provenance. Reading it
            # from the surviving table would report zero by construction - which
            # an earlier version of this script did, while its epilogue claimed
            # censoring rises with the horizon.
            censored = _censored_fraction(prov)
            for row in evaluate(args.models, frames, y, df, cfg, seed):
                rows.append({
                    "definition": definition, "horizon": horizon,
                    "n_arrivals": len(df), "n_matches": int(df["match_id"].nunique()),
                    "censored_frac": censored, **row,
                })
            head = [r for r in rows if r["definition"] == definition and r["horizon"] == horizon]
            phys = next((r for r in head if r["model"] == "M2_physical"), head[0])
            print(f"  {definition:22s} h={horizon:<4} n={len(df):6d} base={phys['base_rate']:.3f} "
                  f"cens={censored:.3f} | M2 mcb={phys['corp_mcb']:+.4f} "
                  f"slope={phys['calibration_slope']:.3f} auc={phys['auc']:.3f}")

    if not rows:
        print("\nNo settings produced results.")
        return 1

    table = pd.DataFrame(rows)
    out = outdir(f"construct_sensitivity/{args.source}", config=cfg)
    write_manifest(out, script="12_construct_sensitivity.py",
                   config={**cfg, "source": args.source, "max_matches": args.max_matches},
                   extra={"horizons": HORIZONS, "definitions": DEFINITIONS})
    save_table(table, out, "construct_sensitivity")

    # --- the pre-registered verdict ---------------------------------------
    print("\n  [does the conclusion flip?]")
    for model, sub in table.groupby("model"):
        mcb = sub["corp_mcb"]
        slope = sub["calibration_slope"]
        verdict = "STABLE" if (mcb > 0.01).all() or (mcb < 0.01).all() else "FLIPS"
        print(f"    {model:18s} MCB range [{mcb.min():+.4f}, {mcb.max():+.4f}]  "
              f"slope range [{slope.min():.3f}, {slope.max():.3f}]  -> {verdict}")

    horizon_only = table[table["definition"] == "controlled_at_horizon"]
    if not horizon_only.empty:
        print("\n  [horizon sweep, physics model]")
        h = horizon_only[horizon_only["model"] == "M2_physical"]
        print(h[["horizon", "base_rate", "censored_frac", "corp_mcb",
                 "calibration_slope", "auc"]]
              .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
        lo, hi = h["censored_frac"].min(), h["censored_frac"].max()
        print(f"\n  Censored fraction across the sweep: {lo:.4f} to {hi:.4f}.")
        print("  Censoring is not random - fouls cluster in contested areas - so if it")
        print("  grows with the horizon, a longer horizon buys construct breadth at the")
        print("  cost of a more selected sample. Check the column above rather than")
        print("  assuming the direction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
