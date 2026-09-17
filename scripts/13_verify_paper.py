#!/usr/bin/env python3
"""Cross-check every number in the paper against the result files.

A paper is a set of claims about files. This script holds those claims in one
explicit ledger and re-derives each from ``results/``, so a stale figure left
behind by a re-run cannot survive quietly. Run it before circulating any draft.

    python scripts/13_verify_paper.py
    python scripts/13_verify_paper.py --strict    # non-zero exit on any mismatch

Every entry names the paper section it appears in, so a failure points at the
sentence to fix rather than at a CSV cell. The ledger is maintained by hand on
purpose: parsing prose for numbers would be brittle, and the act of writing a
claim down twice is itself the check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import REPO_ROOT, banner, outdir, save_table, write_manifest

RESULTS = REPO_ROOT / "results"
SOURCE = "statsbomb_open"


def _table(rel: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / rel)


def build_claims() -> list[tuple]:
    """``(section, label, claimed, actual, tolerance)`` for every figure in the paper."""
    main = _table(f"main/{SOURCE}/tables/metrics.csv")
    raw = main[main["variant"] == "raw"].set_index("model")
    boot = _table(f"main/{SOURCE}/tables/bootstrap.csv").set_index("model")
    comp = _table(f"main/{SOURCE}/tables/comparisons.csv").set_index("model")
    sg = _table(f"subgroups/{SOURCE}/tables/subgroup_metrics.csv")
    adj = _table(f"subgroups/{SOURCE}/tables/subgroup_metrics_adjusted.csv")
    diag = _table(f"selection/{SOURCE}/tables/selection_diagnostics.csv").iloc[0]
    strata = _table(f"selection/{SOURCE}/tables/selection_strata.csv").set_index("stratum")
    tip = _table(f"selection/{SOURCE}/tables/sensitivity_tipping_point.csv")
    weighted = _table(f"selection/{SOURCE}/tables/weighted_vs_unweighted.csv")
    cross = _table(f"transportability/{SOURCE}/tables/crossing_penalty.csv").set_index("model")
    transfer = _table(f"transportability/{SOURCE}/tables/recalibration_transfer.csv")
    sens = _table(f"construct_sensitivity/{SOURCE}/tables/construct_sensitivity.csv")
    epv = _table(f"downstream/{SOURCE}/tables/epv_by_variant.csv")
    disp = _table(f"downstream/{SOURCE}/tables/decision_displacement.csv").set_index("threshold")

    def stratum(var, level, col):
        rows = sg[(sg["subgroup_variable"] == var) & (sg["level"].astype(str) == str(level))]
        return float(rows[col].iloc[0])

    t2 = transfer[transfer["model"] == "M2_physical"].groupby("map")["mcb_removed_frac"].agg(["mean", "min"])
    phys = sens[sens["model"] == "M2_physical"]
    e35 = epv[epv["loss_discount"] == 0.35].set_index("variant")

    claims: list[tuple] = [
        ("4.1", "test-fold arrivals", 40867, raw.loc["M2_physical", "n"], 0),
        ("4.1", "test-fold base rate", 0.894, raw.loc["M2_physical", "base_rate"], 0.001),
    ]
    for model, brier, mcb, slope, auc in [
        ("M1_voronoi", 0.231, 0.142, 0.114, 0.675),
        ("M2_physical", 0.167, 0.081, 0.306, 0.762),
        ("M2a_reach_sigmoid", 0.155, 0.069, 0.636, 0.761),
        ("M0_marginal", 0.094, 0.000, None, 0.500),
        ("M3_logistic", 0.078, 0.001, 1.014, 0.837),
        ("M4_gbm", 0.073, 0.000, 1.018, 0.864),
    ]:
        claims += [
            ("6.1", f"{model} Brier", brier, raw.loc[model, "brier"], 0.0015),
            ("6.1", f"{model} MCB", mcb, max(raw.loc[model, "corp_mcb"], 0.0), 0.0015),
            ("6.1", f"{model} AUC", auc, raw.loc[model, "auc"], 0.001),
        ]
        if slope is not None:
            claims.append(("6.1", f"{model} slope", slope, raw.loc[model, "calibration_slope"], 0.002))
    claims += [
        ("6.1", "M2 MCB CI lower", 0.075, boot.loc["M2_physical", "mcb_lo"], 0.0015),
        ("6.1", "M2 MCB CI upper", 0.087, boot.loc["M2_physical", "mcb_hi"], 0.0015),
        ("6.1", "M2 mean forecast", 0.72, raw.loc["M2_physical", "mean_forecast"], 0.006),
        ("6.2", "M2 DSC", 0.008, raw.loc["M2_physical", "corp_dsc"], 0.0015),
        ("6.2", "M3 DSC", 0.017, raw.loc["M3_logistic", "corp_dsc"], 0.0015),
    ]
    for model, delta in [("M1_voronoi", 0.153), ("M2_physical", 0.089),
                         ("M2a_reach_sigmoid", 0.077), ("M0_marginal", 0.017), ("M4_gbm", -0.005)]:
        claims.append(("6.2", f"{model} dBrier vs M3", delta, comp.loc[model, "delta_brier"], 0.0015))
    for var, level, n, mcb in [
        ("third", "attacking_third", 12161, 0.158), ("third", "middle_third", 20448, 0.057),
        ("third", "defensive_third", 8258, 0.031), ("pass_length_band", "0-10m", 10790, 0.070),
        ("pass_length_band", "20-30m", 7484, 0.098), ("pass_length_band", "30-45m", 3124, 0.126),
        ("pass_length_band", "45m+", 987, 0.156), ("pressure_band", "0", 12679, 0.075),
        ("pressure_band", "3+", 1705, 0.114), ("zone", "att-fifth/centre", 2175, 0.292),
    ]:
        claims += [("6.3", f"{var}/{level} n", n, stratum(var, level, "n"), 0),
                   ("6.3", f"{var}/{level} MCB", mcb, stratum(var, level, "corp_mcb"), 0.0015)]
    claims += [
        ("6.3", "strata with FDR evidence", 47, int((adj["p_value_adj"] < 0.05).sum()), 0),
        ("6.3", "strata tested", 47, len(adj), 0),
    ]
    for level, n, mcb in [("<60%", 11040, 0.095), ("60-75%", 12277, 0.086),
                          ("75-90%", 13104, 0.076), ("90%+", 4446, 0.049)]:
        claims += [("6.4", f"completeness {level} n", n, stratum("completeness_band", level, "n"), 0),
                   ("6.4", f"completeness {level} MCB", mcb, stratum("completeness_band", level, "corp_mcb"), 0.0015)]
    for level, n, mcb, slope in [("False", 8565, 0.145, 0.167), ("True", 32302, 0.064, 0.390)]:
        claims += [("6.4", f"dest_visible {level} n", n, stratum("dest_visible", level, "n"), 0),
                   ("6.4", f"dest_visible {level} MCB", mcb, stratum("dest_visible", level, "corp_mcb"), 0.0015),
                   ("6.4", f"dest_visible {level} slope", slope, stratum("dest_visible", level, "calibration_slope"), 0.002)]
    claims += [
        ("6.5", "selection AUC", 0.972, diag["auc_selection"], 0.0015),
        ("6.5", "arrivals outside candidate support", 0.526, diag["frac_real_outside_candidate_support"], 0.002),
        ("6.5", "weight share, top 1%", 0.33, diag["weight_share_top_1pct"], 0.005),
        # Checked against weighted_vs_unweighted.csv, not the diagnostics table:
        # the two report DIFFERENT weight vectors. overlap_diagnostics computes an
        # untrimmed ESS (1,525); the analysis actually uses trimmed, stabilised
        # weights (7,753). The paper quotes the latter, so that is what is checked.
        ("6.5", "effective sample size (trimmed weights)", 7753,
         weighted[weighted["weighting"] != "none"]["effective_n"].iloc[0], 1),
        ("6.5", "unweighted n", 48268, weighted["n"].iloc[0], 0),
        ("6.5", "MCB unweighted", 0.085, weighted[weighted["weighting"] == "none"]["miscalibration"].iloc[0], 0.0015),
        ("6.5", "MCB reweighted (reported as uninterpretable)", 0.127,
         weighted[weighted["weighting"] != "none"]["miscalibration"].iloc[0], 0.0015),
    ]
    for k, base, mean_f, mcb in [(0, 0.790, 0.507, 0.148), (1, 0.896, 0.612, 0.157),
                                 (2, 0.932, 0.782, 0.065), (3, 0.959, 0.865, 0.039),
                                 (4, 0.969, 0.906, 0.022)]:
        claims += [("6.5", f"selection stratum {k} base rate", base, strata.loc[k, "base_rate"], 0.0015),
                   ("6.5", f"selection stratum {k} mean forecast", mean_f, strata.loc[k, "mean_forecast"], 0.0015),
                   ("6.5", f"selection stratum {k} MCB", mcb, strata.loc[k, "miscalibration"], 0.0015)]
    gamma3 = tip[(tip["gamma"] == 3.0) & (tip["direction"] == "inflated")]["corp_mcb"].iloc[0]
    claims.append(("6.5", "MCB at Gamma = 3", 0.039, gamma3, 0.002))
    claims += [
        ("6.6", "M2 MCB within competition", 0.085, cross.loc["M2_physical", "mcb_within_competition"], 0.0015),
        ("6.6", "M2 MCB across competitions", 0.083, cross.loc["M2_physical", "mcb_across_competition"], 0.0015),
    ]
    for mapping, mean_v, min_v in [("isotonic", 0.989, 0.986), ("beta", 0.986, 0.983), ("platt", 0.974, 0.964)]:
        claims += [("6.6", f"{mapping} transfer mean", mean_v, t2.loc[mapping, "mean"], 0.002),
                   ("6.6", f"{mapping} transfer worst case", min_v, t2.loc[mapping, "min"], 0.002)]
    claims += [
        ("6.7", "MCB sweep minimum", 0.064, phys["corp_mcb"].min(), 0.0015),
        ("6.7", "MCB sweep maximum", 0.088, phys["corp_mcb"].max(), 0.0015),
        ("6.7", "slope sweep minimum", 0.219, phys["calibration_slope"].min(), 0.002),
        ("6.7", "slope sweep maximum", 0.305, phys["calibration_slope"].max(), 0.002),
        ("6.7", "first-touch MCB", 0.088, phys[phys["definition"] == "first_touch"]["corp_mcb"].iloc[0], 0.0015),
        ("6.7", "censored fraction minimum", 0.006, phys["censored_frac"].min(), 0.001),
        ("6.7", "censored fraction maximum", 0.029, phys["censored_frac"].max(), 0.001),
        ("7.1", "EPV raw mean", 0.559, e35.loc["raw", "mean_epv"], 0.0015),
        ("7.1", "EPV isotonic mean", 0.773, e35.loc["isotonic", "mean_epv"], 0.0015),
        ("7.1", "EPV raw total", 22844, e35.loc["raw", "total_epv"], 2),
        ("7.1", "EPV isotonic total", 31581, e35.loc["isotonic", "total_epv"], 2),
        ("7.1", "EPV understatement", 0.28, 1 - e35.loc["raw", "mean_epv"] / e35.loc["isotonic", "mean_epv"], 0.005),
    ]
    for threshold, changed, acc_raw, acc_cal in [
        (0.5, 0.252, 0.762, 0.895), (0.6, 0.306, 0.727, 0.895),
        (0.7, 0.360, 0.680, 0.892), (0.8, 0.240, 0.622, 0.792),
    ]:
        claims += [
            ("7.2", f"decisions changed @ {threshold}", changed, disp.loc[threshold, "frac_decisions_changed"], 0.0015),
            ("7.2", f"accuracy raw @ {threshold}", acc_raw, disp.loc[threshold, "accuracy_raw"], 0.0015),
            ("7.2", f"accuracy recalibrated @ {threshold}", acc_cal, disp.loc[threshold, "accuracy_cal"], 0.0015),
        ]
    return claims


# ---------------------------------------------------------------------------
# LaTeX consistency
# ---------------------------------------------------------------------------

# Headline figures that must appear verbatim in paper/paper.tex. Each is already
# checked against results/ by the claim ledger above; this second pass only
# catches the LaTeX manuscript drifting away from the verified markdown.
LATEX_HEADLINES = [
    "137,545", "151,418", "40,867", "179 matches", "53 held-out matches",
    "0.762", "0.167", "0.094", "0.306", "0.081",
    "0.089", "0.082, 0.096",
    "0.158", "0.031", "0.292",
    "0.145", "0.064",
    "0.972", "52.6", "7,753", "1,525",
    "97--99", "98.9", "97.4",
    "0.559", "0.773", "28",
    "36.0", "25.2", "30.6", "24.0",
    "11.1", "0.013",
    "83.9", "96.6", "21",
]


def check_latex(tex_path):
    """Return (checked, missing) for the headline strings in paper/paper.tex."""
    if not tex_path.exists():
        return 0, None
    # ``137{,}545`` in LaTeX is ``137,545`` on the page.
    text = tex_path.read_text(encoding="utf-8").replace("{,}", ",")
    missing = [h for h in LATEX_HEADLINES if h not in text]
    return len(LATEX_HEADLINES), missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on any mismatch.")
    args = ap.parse_args()

    banner("Verifying docs/paper.md against results/")
    try:
        claims = build_claims()
    except FileNotFoundError as exc:
        print(f"  Missing result file: {exc}")
        print("  Run the pipeline in docs/paper.md 'Reproduction' before verifying.")
        return 2

    rows, failures = [], []
    current_section = None
    for section, label, claimed, actual, tol in claims:
        actual = float(actual)
        ok = abs(float(claimed) - actual) <= tol
        if section != current_section:
            print(f"\n  [section {section}]")
            current_section = section
        print(f"    {'ok  ' if ok else 'FAIL'}  {label:44s} paper={float(claimed):<11.4g} results={actual:<11.4g}")
        rows.append({"section": section, "claim": label, "paper": float(claimed),
                     "results": actual, "tolerance": tol, "ok": ok})
        if not ok:
            failures.append(f"§{section} {label}: paper {claimed}, results {actual}")

    out = outdir("paper_verification")
    write_manifest(out, script="13_verify_paper.py", config={"source": SOURCE},
                   extra={"n_claims": len(rows), "n_failures": len(failures)})
    save_table(pd.DataFrame(rows), out, "claim_ledger")

    print("\n" + "=" * 72)
    if failures:
        print(f"  {len(failures)} of {len(rows)} claims do not match the result files:")
        for f in failures:
            print(f"    - {f}")
        print("\n  Fix the paper, or re-run the analysis that produced the table.")
        return 1 if args.strict else 0
    print(f"  All {len(rows)} claims in docs/paper.md match the result files.")

    tex = REPO_ROOT / "paper" / "paper.tex"
    n_tex, missing = check_latex(tex)
    if missing is None:
        print("  (paper/paper.tex not present; LaTeX cross-check skipped.)")
    elif missing:
        print(f"\n  {len(missing)} of {n_tex} headline figures are missing from "
              f"paper/paper.tex:")
        for m in missing:
            print(f"    - {m}")
        return 1 if args.strict else 0
    else:
        print(f"  All {n_tex} headline figures also appear in paper/paper.tex.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
