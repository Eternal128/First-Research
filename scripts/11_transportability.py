#!/usr/bin/env python3
"""Does calibration transport across competitions? (RQ3, H4a, H5c)

Implements levels 3-6 of the split hierarchy in proposal Section 12.2, which the
match-level hold-out in ``scripts/03_main_analysis.py`` cannot reach:

* **Leave-one-competition-out** - fit on every competition but one, evaluate on
  the one held out. The strictest generalisation test the corpus supports.
* **Recalibration transfer** - fit a recalibration map on the training
  competitions and apply it to the held-out one. This is the practically
  decisive question: is a single global fix deployable, or must every club
  refit for its own league?
* **Within- versus between-competition** - the same models under a match-level
  split inside each competition, so the cost of crossing a competition boundary
  can be separated from the cost of holding out matches at all.

    python scripts/11_transportability.py --source statsbomb_open

Requires a corpus spanning at least two competitions.
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
    RECALIBRATORS, brier_score, calibration_slope_intercept, cluster_bootstrap,
    corp_decomposition, roc_auc,
)
from pcc.evaluation import grouped_holdout, leave_one_competition_out  # noqa: E402
from pcc.models import build_model  # noqa: E402


def score(y, p, groups=None, *, n_boot: int = 0, random_state: int = 0) -> dict:
    corp = corp_decomposition(y, p)
    slope = calibration_slope_intercept(y, p)
    out = {
        "n": int(y.size),
        "base_rate": float(np.mean(y)),
        "mean_forecast": float(np.mean(p)),
        "brier": brier_score(y, p),
        "corp_mcb": corp.miscalibration,
        "corp_dsc": corp.discrimination,
        "corp_unc": corp.uncertainty,
        "calibration_slope": slope["slope"],
        "calibration_in_the_large": slope["intercept_in_the_large"],
        "auc": roc_auc(y, p),
    }
    if n_boot and groups is not None:
        work = pd.DataFrame({"y": y, "p": p, "match_id": groups})
        if work["match_id"].nunique() >= 5:
            res = cluster_bootstrap(
                lambda d: corp_decomposition(d["y"], d["p"]).miscalibration,
                work, cluster_col="match_id", n_boot=n_boot, random_state=random_state,
            )
            out["mcb_lo"], out["mcb_hi"] = res.lo, res.hi
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="statsbomb_open")
    ap.add_argument("--root", default=None)
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--models", nargs="+", default=["M2_physical", "M3_logistic", "M4_gbm"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_state", 0))
    banner(f"Transportability across competitions  |  source={args.source}")

    root = args.root or str(DATA_RAW / args.source)
    frames, df, _prov = load_corpus(args.source, cfg, root=root)
    y = df["y_control"].to_numpy(dtype=int)
    competitions = sorted(df["competition"].unique())
    print(f"  {len(df)} arrivals, {df['match_id'].nunique()} matches, "
          f"{len(competitions)} competitions")
    for comp in competitions:
        sub = df[df["competition"] == comp]
        print(f"    {comp:32s} {len(sub):7d} arrivals  {sub['match_id'].nunique():3d} matches  "
              f"base rate {sub['y_control'].mean():.3f}")

    if len(competitions) < 2:
        print("\nERROR: transportability needs at least two competitions.")
        return 1

    out = outdir(f"transportability/{args.source}", config=cfg)
    write_manifest(out, script="11_transportability.py",
                   config={**cfg, "source": args.source, "models": args.models},
                   extra={"competitions": competitions})

    rows: list[dict] = []
    recal_rows: list[dict] = []

    for split in leave_one_competition_out(df, random_state=cfg.get("random_state", 0)):
        held = split.name.replace("loco_", "")
        test_matches = df.iloc[split.test]["match_id"].to_numpy()
        print(f"\n  [hold out: {held}]  train {split.train.size}  valid {split.valid.size}  test {split.test.size}")

        for name in args.models:
            model = build_model(name, **cfg["models"].get("params", {}).get(name, {}))
            if model.requires_fitting:
                model.fit([frames[i] for i in split.train], y[split.train])

            p_test = np.asarray(model.predict([frames[i] for i in split.test]), dtype=float)
            stats = score(y[split.test], p_test, test_matches,
                          n_boot=args.n_boot, random_state=cfg.get("random_state", 0))
            rows.append({"held_out": held, "model": name, "variant": "raw", **stats})
            print(f"    {name:18s} raw        brier={stats['brier']:.4f} "
                  f"mcb={stats['corp_mcb']:+.4f} slope={stats['calibration_slope']:.3f} "
                  f"auc={stats['auc']:.3f}")

            # Recalibration transfer: the map is fitted on the *training*
            # competitions only. If it still works on the held-out one, a single
            # global fix is deployable; if not, every league needs its own.
            if split.valid.size:
                p_valid = np.asarray(model.predict([frames[i] for i in split.valid]), dtype=float)
                for rname in ("platt", "isotonic", "beta"):
                    rc = RECALIBRATORS[rname]()
                    rc.fit(p_valid, y[split.valid])
                    p_cal = np.clip(rc.transform(p_test), 0.0, 1.0)
                    s = score(y[split.test], p_cal, test_matches)
                    rows.append({"held_out": held, "model": name, "variant": rname, **s})
                    recal_rows.append({
                        "held_out": held, "model": name, "map": rname,
                        "mcb_raw": stats["corp_mcb"], "mcb_after": s["corp_mcb"],
                        "mcb_removed_frac": 1.0 - s["corp_mcb"] / max(stats["corp_mcb"], 1e-9),
                        "brier_raw": stats["brier"], "brier_after": s["brier"],
                    })

    table = pd.DataFrame(rows)
    save_table(table, out, "leave_one_competition_out")

    # --- within-competition reference -------------------------------------
    print("\n  [within-competition reference: match-level split inside each competition]")
    within_rows = []
    for comp in competitions:
        sub = df[df["competition"] == comp].reset_index()
        if sub["match_id"].nunique() < 6:
            print(f"    {comp:32s} skipped ({sub['match_id'].nunique()} matches)")
            continue
        idx = sub["index"].to_numpy()
        local = grouped_holdout(sub, group_col="match_id", random_state=cfg.get("random_state", 0))
        for name in args.models:
            model = build_model(name, **cfg["models"].get("params", {}).get(name, {}))
            tr, te = idx[local.train], idx[local.test]
            if model.requires_fitting:
                model.fit([frames[i] for i in tr], y[tr])
            p = np.asarray(model.predict([frames[i] for i in te]), dtype=float)
            s = score(y[te], p, df.iloc[te]["match_id"].to_numpy())
            within_rows.append({"competition": comp, "model": name, **s})
            print(f"    {comp:32s} {name:18s} mcb={s['corp_mcb']:+.4f} "
                  f"slope={s['calibration_slope']:.3f} auc={s['auc']:.3f}")
    within = pd.DataFrame(within_rows)
    save_table(within, out, "within_competition")

    # --- the contrast that answers H4a ------------------------------------
    if not within.empty:
        across = table[table["variant"] == "raw"].groupby("model")["corp_mcb"].mean()
        inside = within.groupby("model")["corp_mcb"].mean()
        contrast = pd.DataFrame({"mcb_within_competition": inside, "mcb_across_competition": across})
        contrast["penalty_for_crossing"] = contrast["mcb_across_competition"] - contrast["mcb_within_competition"]
        print("\n  [H4a: cost of crossing a competition boundary]")
        print(contrast.to_string(float_format=lambda v: f"{v:9.4f}"))
        save_table(contrast.reset_index(), out, "crossing_penalty")

    if recal_rows:
        recal = pd.DataFrame(recal_rows)
        save_table(recal, out, "recalibration_transfer")
        print("\n  [H5c: does a recalibration map fitted elsewhere still work?]")
        summary = recal.groupby(["model", "map"])["mcb_removed_frac"].agg(["mean", "min"])
        print(summary.to_string(float_format=lambda v: f"{v:8.3f}"))
        print("  (1.000 = the map removed all measured miscalibration on the held-out competition)")

    print("\n  NOTE ON SCOPE: with a freeze-frame corpus these results describe the")
    print("  VELOCITY-FREE, camera-limited variant of each model. They bound what the")
    print("  full-tracking versions would do, they do not measure it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
