#!/usr/bin/env python3
"""How many matches are needed? A design question, answered before data access.

Simulates the study at several corpus sizes and reports, for each, the width of
the cluster-bootstrap interval on the primary calibration statistic and the
power to detect a calibration slope departing from one by a stated amount.

Why this belongs in the repository rather than in a footnote: the honest answer
to "is the Metrica sample enough?" is a number, and the number can be produced
before any provider is contacted. If nine matches cannot separate a slope of
0.85 from a slope of 1.0, that fact should change the study design, not be
discovered in the results section.

The simulated football is crude, so these are *order-of-magnitude* design
guides, not guarantees. Real arrivals are more strongly clustered within
matches than the simulator's are, so the required corpus is if anything larger.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import banner, load_config, outdir, save_table, seed_everything, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.calibration import calibration_slope_intercept, cluster_bootstrap, corp_decomposition  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--match-counts", type=int, nargs="+", default=[3, 9, 20, 40, 64])
    ap.add_argument("--arrivals-per-match", type=int, default=400)
    ap.add_argument("--n-replicates", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--target-slope", type=float, default=0.85,
                    help="Mean calibration slope the study should distinguish from 1.0.")
    ap.add_argument("--slope-heterogeneity", type=float, default=0.15,
                    help="Between-match SD of the calibration slope. This, not the number "
                         "of arrivals, is what limits power; vary it to stress the design.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_state", 0))
    banner("Design power analysis")
    print(f"  Target: distinguish a mean calibration slope of {args.target_slope} from 1.0")
    print(f"  Between-match slope SD: {args.slope_heterogeneity}")
    print(f"  Match counts: {args.match_counts}  |  {args.arrivals_per_match} arrivals/match")
    print(f"  {args.n_replicates} replicate corpora per size\n")

    rng = np.random.default_rng(cfg.get("random_state", 0))
    rows = []
    for n_matches in args.match_counts:
        slope_hits = 0
        mcb_widths, slope_ests = [], []
        for rep in range(args.n_replicates):
            # Clustered forecast/outcome process. Two levels of match-level
            # dependence are modelled, and the second is the one that actually
            # governs the sampling variance of a calibration statistic:
            #
            #   (i) a match random intercept on the latent signal - shifts the
            #       base rate between matches;
            #  (ii) match-level heterogeneity in the DISTORTION itself - one
            #       team's shape makes the model overconfident, another's does
            #       not.
            #
            # Omitting (ii) is the common mistake: it makes matches exchangeable
            # in exactly the dimension being estimated and produces power curves
            # that are far too optimistic. ``--slope-heterogeneity`` controls it
            # and is the parameter to vary when the design guide is challenged.
            m = np.repeat(np.arange(n_matches), args.arrivals_per_match)
            n = m.size
            match_eff = rng.normal(0, 0.35, n_matches)[m]
            z = rng.normal(0, 1.4, n) + match_eff
            p_true = 1 / (1 + np.exp(-z))
            y = rng.binomial(1, p_true)

            match_slope = np.clip(
                rng.normal(args.target_slope, args.slope_heterogeneity, n_matches), 0.2, 3.0
            )[m]
            p_hat = 1 / (1 + np.exp(-z / match_slope))

            df = pd.DataFrame({"y": y, "p": p_hat, "match_id": m})
            res = cluster_bootstrap(
                lambda d: calibration_slope_intercept(d["y"], d["p"])["slope"],
                df, cluster_col="match_id", n_boot=args.n_boot, random_state=rep,
            )
            if np.isfinite(res.lo) and np.isfinite(res.hi):
                slope_hits += int(res.hi < 1.0)
                slope_ests.append(res.point)
            mcb = cluster_bootstrap(
                lambda d: corp_decomposition(d["y"], d["p"]).miscalibration,
                df, cluster_col="match_id", n_boot=args.n_boot, random_state=rep + 1000,
            )
            if np.isfinite(mcb.lo) and np.isfinite(mcb.hi):
                mcb_widths.append(mcb.hi - mcb.lo)

        power = slope_hits / max(args.n_replicates, 1)
        row = {
            "n_matches": n_matches,
            "n_arrivals": n_matches * args.arrivals_per_match,
            "power_slope_ci_excludes_1": power,
            "mean_slope_estimate": float(np.mean(slope_ests)) if slope_ests else np.nan,
            "mean_mcb_ci_width": float(np.mean(mcb_widths)) if mcb_widths else np.nan,
            "n_replicates": args.n_replicates,
            "target_slope": args.target_slope,
            "slope_heterogeneity": args.slope_heterogeneity,
        }
        rows.append(row)
        print(f"  {n_matches:3d} matches ({row['n_arrivals']:6d} arrivals): "
              f"power {power:.2f}  |  mean MCB CI width {row['mean_mcb_ci_width']:.4f}")

    table = pd.DataFrame(rows)
    out = outdir("power", config=cfg)
    write_manifest(out, script="08_power_analysis.py", config={**cfg, "args": vars(args)})
    save_table(table, out, "power_analysis")

    print()
    # The smallest size at which this and EVERY larger tested size clear the
    # threshold. Taking the smallest size that clears it in isolation would
    # report a Monte Carlo fluctuation as a design recommendation.
    ordered = table.sort_values("n_matches")
    power = ordered["power_slope_ci_excludes_1"].to_numpy()
    sizes = ordered["n_matches"].to_numpy()
    stable = [int(sizes[i]) for i in range(len(sizes)) if (power[i:] >= 0.8).all()]

    se = np.sqrt(0.8 * 0.2 / max(args.n_replicates, 1))
    if not (np.diff(power) >= -2 * se).all():
        print(f"  WARNING: the power curve is not monotone beyond Monte Carlo noise "
              f"(SE ~ {se:.2f} at {args.n_replicates} replicates).")
        print("  Increase --n-replicates before treating any single row as a recommendation.")

    if not stable:
        print(f"  No tested corpus size reached 80% power for a mean slope of {args.target_slope}")
        print(f"  with between-match slope SD {args.slope_heterogeneity}. Either a larger corpus")
        print("  is required, or the study should pre-register a larger minimum detectable")
        print("  effect and say so.")
    else:
        print(f"  ~{min(stable)} matches reach 80% power for a mean slope of {args.target_slope}")
        print(f"  (between-match slope SD {args.slope_heterogeneity}, {args.n_replicates} replicates).")
        print("  The simulated clustering is milder than real football's, so treat this as a")
        print("  LOWER bound on the corpus the real study needs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
