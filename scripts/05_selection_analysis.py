#!/usr/bin/env python3
"""Selection bias: how non-random are the destinations we evaluate on? (Section 14)

Produces four things, in increasing order of how much they can be trusted:

1. **Density-ratio weights** and their diagnostics: how strongly do passers
   select destinations, and what is the effective sample size after reweighting?
2. **Stratified results** by selection score - the transparent alternative to a
   single weighted number.
3. **The chosen-versus-unchosen contrast**: calibration on arrivals nobody aimed
   (deflections, clearances, second balls) against arrivals that were chosen.
   This is the study's strongest identification argument.
4. **A sensitivity analysis** bounding how strong unmeasured selection would
   have to be to explain away the measured miscalibration.

The script prints the caveat it needs to print: none of this identifies control
at destinations the ball never reached.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import banner, load_config, outdir, save_table, seed_everything, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.calibration import brier_score, corp_decomposition  # noqa: E402
from pcc.models.features import build_feature_matrix  # noqa: E402
from pcc.selection import (  # noqa: E402
    CandidateConfig, SelectionModel, effective_sample_size, generate_candidates,
    negative_control_comparison, overlap_diagnostics, propensity_strata,
    selection_weights, tipping_point_analysis,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="simulated")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--model", default="M2_physical")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_state", 0))
    banner("Selection analysis")
    print("  Reweighting corrects covariate shift in WHERE the ball arrives.")
    print("  It does NOT correct selection on what the passer knew and we did not,")
    print("  and it does NOT identify control where the ball never arrived.\n")

    if args.source != "simulated":
        print("ERROR: this script currently rebuilds frames only for the simulated source; "
              "for provider data, complete the loader and cache frames in scripts/02.")
        return 1

    from pcc.data import SimulationConfig, filter_arrivals
    from pcc.data.preprocess import PreprocessConfig, Provenance
    from pcc.data.synthetic import simulate_dataset
    from pcc.evaluation import add_subgroups
    from pcc.models import build_model

    sim = SimulationConfig(
        n_matches=cfg["simulation"]["n_matches"],
        arrivals_per_match=cfg["simulation"]["arrivals_per_match"],
        random_state=cfg.get("random_state", 0),
    )
    frames, arrivals = simulate_dataset(sim)
    prov = Provenance()
    filtered = filter_arrivals(arrivals, PreprocessConfig(**cfg.get("preprocess", {})), provenance=prov)
    keep = arrivals["arrival_id"].isin(filtered["arrival_id"]).to_numpy()
    frames = [f for f, k in zip(frames, keep) if k]
    df = add_subgroups(filtered).reset_index(drop=True)
    y = df["y_control"].to_numpy(dtype=int)

    model = build_model(args.model, **cfg["models"].get("params", {}).get(args.model, {}))
    if model.requires_fitting:
        model.fit(frames, y)
    p = np.asarray(model.predict(frames), dtype=float)
    df["p"] = p

    out = outdir(f"selection/{args.source}", config=cfg)
    write_manifest(out, script="05_selection_analysis.py", config={**cfg, "model": args.model})

    # --- 1. density ratio ----------------------------------------------------
    sel_cfg = CandidateConfig(
        n_per_arrival=cfg["selection"]["candidates_per_arrival"],
        proposal=cfg["selection"]["proposal"],
        max_pass_length=cfg["selection"]["max_pass_length"],
        random_state=cfg.get("random_state", 0),
    )
    print(f"  generating {sel_cfg.n_per_arrival} candidate destinations per arrival "
          f"({sel_cfg.proposal} proposal)...")
    cands, parent = generate_candidates(frames, sel_cfg)
    real_X = build_feature_matrix(frames)
    cand_X = build_feature_matrix(cands)

    sel = SelectionModel().fit(real_X, cand_X, random_state=cfg.get("random_state", 0))
    diag = overlap_diagnostics(sel, real_X, cand_X)
    w = selection_weights(sel, real_X, trim_quantile=cfg["selection"]["trim_quantile"])

    print("\n  [selection diagnostics]")
    for k, v in diag.items():
        print(f"    {k:38s} {v: .4f}")
    print(f"    {'effective n after weighting':38s} {effective_sample_size(w):.0f} of {len(w)}")
    save_table(pd.DataFrame([diag]), out, "selection_diagnostics")

    unweighted = corp_decomposition(y, p)
    weighted = corp_decomposition(y, p, w)
    comp = pd.DataFrame(
        [
            {"weighting": "none", "n": len(y), "effective_n": float(len(y)),
             "brier": brier_score(y, p), **unweighted.as_dict()},
            {"weighting": f"ipw_{sel_cfg.proposal}", "n": len(y),
             "effective_n": effective_sample_size(w),
             "brier": brier_score(y, p, w), **weighted.as_dict()},
        ]
    )
    print("\n  [unweighted vs reweighted]")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    save_table(comp, out, "weighted_vs_unweighted")

    # --- 2. stratified -------------------------------------------------------
    strata = propensity_strata(sel.odds(real_X), n_strata=cfg["selection"]["n_strata"])
    rows = []
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < 100 or len(np.unique(y[m])) < 2:
            rows.append({"stratum": int(s), "n": int(m.sum()), "suppressed": True})
            continue
        c = corp_decomposition(y[m], p[m])
        rows.append({"stratum": int(s), "n": int(m.sum()), "suppressed": False,
                     "base_rate": float(y[m].mean()), "mean_forecast": float(p[m].mean()),
                     **c.as_dict()})
    strat = pd.DataFrame(rows)
    print("\n  [by selection-score stratum: stratum 0 = least typical destinations]")
    print(strat.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    save_table(strat, out, "selection_strata")

    # --- 3. negative control -------------------------------------------------
    nc = negative_control_comparison(df, prob_col="p", outcome_col="y_control",
                                     n_boot=max(200, cfg["evaluation"]["n_boot"] // 2))
    print("\n  [chosen vs unchosen destinations]  <- the strongest identification argument")
    print(nc.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    save_table(nc, out, "negative_control")
    if len(nc) == 2 and not nc["suppressed"].any():
        gap = float(nc.set_index("subsample")["corp_mcb"].get("chosen", np.nan)
                    - nc.set_index("subsample")["corp_mcb"].get("unchosen", np.nan))
        print(f"    MCB(chosen) - MCB(unchosen) = {gap:+.4f}")
        print("    A positive gap is consistent with selection on information the model "
              "does not see;\n    it is also consistent with chosen and unchosen arrivals "
              "differing in arrival physics.\n    Matched comparison is required before "
              "attributing it to selection.")

    # --- 4. sensitivity ------------------------------------------------------
    tip = tipping_point_analysis(y, p, gamma_grid=cfg["selection"]["gamma_grid"])
    print("\n  [sensitivity: unmeasured selection as an odds multiplier Gamma]")
    print(tip.to_string(index=False, float_format=lambda v: f"{v:10.4f}"))
    save_table(tip, out, "sensitivity_tipping_point")
    return 0


if __name__ == "__main__":
    sys.exit(main())
