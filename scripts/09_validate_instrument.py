#!/usr/bin/env python3
"""Validate the measuring instrument before measuring anything with it.

A calibration study's central tool is its calibration estimator. This script
checks that the estimator does what it claims, on data whose truth is known:

1. The **oracle forecast** (the simulator's own ``p_true``) should show
   miscalibration indistinguishable from zero and a calibration slope of one.
2. A **deliberately distorted** oracle (logit rescaled by a known factor)
   should be detected, and the recovered slope should match the injected one.
3. **Recalibration should recover** most of the score lost to the distortion.
4. **AUC should be unchanged** by the distortion - the demonstration that
   discrimination metrics are blind to miscalibration.
5. The **cluster bootstrap** should produce wider intervals than a naive
   independent bootstrap on the same clustered data.

If any of these fails, no result from this repository means anything. This is
also the fastest end-to-end smoke test of the whole package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import banner, load_config, outdir, save_figure, save_table, seed_everything, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.calibration import (  # noqa: E402
    IsotonicRecalibrator, PlattScaling, brier_score, calibration_slope_intercept,
    cluster_bootstrap, corp_decomposition, roc_auc,
)


def check(name: str, passed: bool, detail: str) -> dict:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return {"check": name, "passed": bool(passed), "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--distortion", type=float, default=1.8,
                    help="Logit-scale factor; the recovered slope should be ~1/distortion.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_state", 0))
    banner("Instrument validation")

    from pcc.data.synthetic import SimulationConfig, simulate_dataset

    sim = SimulationConfig(
        n_matches=max(cfg["simulation"]["n_matches"], 12),
        arrivals_per_match=cfg["simulation"]["arrivals_per_match"],
        random_state=cfg.get("random_state", 0),
    )
    _frames, df = simulate_dataset(sim)
    y = df["y_control"].to_numpy(dtype=int)
    p_true = df["p_true"].to_numpy(dtype=float)
    m = df["match_id"].to_numpy()
    print(f"  {len(y)} simulated arrivals across {df['match_id'].nunique()} matches, "
          f"base rate {y.mean():.3f}\n")

    logit = np.log(np.clip(p_true, 1e-9, 1 - 1e-9) / np.clip(1 - p_true, 1e-9, 1))
    p_bad = 1 / (1 + np.exp(-args.distortion * logit))

    results = []
    corp_true = corp_decomposition(y, p_true)
    slope_true = calibration_slope_intercept(y, p_true)["slope"]
    results.append(check(
        "oracle miscalibration near zero",
        abs(corp_true.miscalibration) < 0.005,
        f"MCB = {corp_true.miscalibration:+.5f} (|.| < 0.005)",
    ))
    results.append(check(
        "oracle calibration slope near one",
        abs(slope_true - 1.0) < 0.10,
        f"slope = {slope_true:.4f}",
    ))

    corp_bad = corp_decomposition(y, p_bad)
    slope_bad = calibration_slope_intercept(y, p_bad)["slope"]
    expected = 1.0 / args.distortion
    results.append(check(
        "injected distortion detected",
        corp_bad.miscalibration > corp_true.miscalibration + 0.002,
        f"MCB rose {corp_true.miscalibration:+.5f} -> {corp_bad.miscalibration:+.5f}",
    ))
    results.append(check(
        "recovered slope matches the injected one",
        abs(slope_bad - expected) < 0.12,
        f"recovered {slope_bad:.4f}, injected 1/{args.distortion} = {expected:.4f}",
    ))

    auc_true, auc_bad = roc_auc(y, p_true), roc_auc(y, p_bad)
    results.append(check(
        "AUC is blind to miscalibration",
        abs(auc_true - auc_bad) < 1e-6,
        f"AUC {auc_true:.6f} vs {auc_bad:.6f} - identical, as a rank metric must be",
    ))

    half = len(y) // 2
    for name, rc in (("platt", PlattScaling()), ("isotonic", IsotonicRecalibrator())):
        rc.fit(p_bad[:half], y[:half])
        fixed = rc.transform(p_bad[half:])
        before = brier_score(y[half:], p_bad[half:])
        after = brier_score(y[half:], fixed)
        oracle = brier_score(y[half:], p_true[half:])
        recovered = (before - after) / max(before - oracle, 1e-9)
        results.append(check(
            f"{name} recalibration recovers the lost score",
            after < before and recovered > 0.5,
            f"Brier {before:.4f} -> {after:.4f} (oracle {oracle:.4f}); "
            f"{recovered:.0%} of the gap recovered",
        ))

    work = pd.DataFrame({"y": y, "p": p_bad, "match_id": m, "fake_id": np.arange(len(y))})
    clustered = cluster_bootstrap(lambda d: brier_score(d["y"], d["p"]), work,
                                  cluster_col="match_id", n_boot=300)
    naive = cluster_bootstrap(lambda d: brier_score(d["y"], d["p"]), work,
                              cluster_col="fake_id", n_boot=300)
    results.append(check(
        "cluster bootstrap is wider than the naive one",
        (clustered.hi - clustered.lo) > (naive.hi - naive.lo),
        f"clustered width {clustered.hi - clustered.lo:.5f} vs naive "
        f"{naive.hi - naive.lo:.5f} ({(clustered.hi - clustered.lo)/max(naive.hi - naive.lo,1e-9):.1f}x)",
    ))

    table = pd.DataFrame(results)
    out = outdir("validation", config=cfg)
    write_manifest(out, script="09_validate_instrument.py", config={**cfg, "distortion": args.distortion})
    save_table(table, out, "instrument_validation")

    try:
        from pcc.viz import reliability_diagram

        fig = reliability_diagram(
            y, {"oracle (truth)": p_true, f"distorted (logit x{args.distortion})": p_bad},
            n_bins=12, title="Instrument validation on simulated data",
        )
        save_figure(fig, out, "instrument_validation")
    except Exception as exc:  # pragma: no cover
        print(f"  figure skipped: {exc}")

    n_fail = int((~table["passed"]).sum())
    print(f"\n  {len(table) - n_fail}/{len(table)} checks passed")
    if n_fail:
        print("  The calibration machinery is not behaving as specified. Do not run the")
        print("  main analysis until this is resolved.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
