#!/usr/bin/env python3
"""Assemble the arrivals table and frames from a configured source.

Usage
-----
    python scripts/02_build_dataset.py --source simulated
    python scripts/02_build_dataset.py --source metrica_sample --root data/raw/metrica_sample

Writes ``results/dataset/<source>/arrivals.csv`` plus a preprocessing provenance
table and a validation report. Frames are cached as a compressed ``.npz`` so the
analysis scripts do not have to rebuild them, but the CSV is the durable
artefact: it is readable in ten years without this code.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from _common import banner, load_config, outdir, save_table, seed_everything, write_manifest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from pcc.data import (  # noqa: E402
    DataNotAvailable, PreprocessConfig, Provenance, SimulationConfig,
    filter_arrivals, load_source, validate_arrivals,
)
from pcc.evaluation import add_subgroups  # noqa: E402


def frames_to_npz(frames, path) -> None:
    """Cache frames as ragged arrays. Player counts vary, so store offsets."""
    payload = {}
    for i, f in enumerate(frames):
        payload[f"att_xy_{i}"] = f.att_xy.astype(np.float32)
        payload[f"att_v_{i}"] = f.att_v.astype(np.float32)
        payload[f"def_xy_{i}"] = f.def_xy.astype(np.float32)
        payload[f"def_v_{i}"] = f.def_v.astype(np.float32)
        payload[f"meta_{i}"] = np.array(
            [
                f.target[0], f.target[1], f.flight_time,
                f.meta.get("origin", f.target)[0], f.meta.get("origin", f.target)[1],
            ],
            dtype=np.float32,
        )
        payload[f"gk_{i}"] = np.concatenate([f.att_is_gk, f.def_is_gk]).astype(np.int8)
        payload[f"obs_{i}"] = np.concatenate([f.att_observed, f.def_observed]).astype(np.int8)
    payload["n_frames"] = np.array([len(frames)])
    np.savez_compressed(path, **payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="simulated")
    ap.add_argument("--root", default=None, help="Path to raw data for provider sources.")
    ap.add_argument("--config", default="default.yaml")
    ap.add_argument("--n-matches", type=int, default=None, help="Simulated source only.")
    ap.add_argument("--arrivals-per-match", type=int, default=None, help="Simulated source only.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("random_state", 0))
    banner(f"Building dataset from source: {args.source}")

    kwargs: dict = {}
    if args.source == "simulated":
        sim = SimulationConfig(
            n_matches=args.n_matches or cfg["simulation"]["n_matches"],
            arrivals_per_match=args.arrivals_per_match or cfg["simulation"]["arrivals_per_match"],
            random_state=cfg.get("random_state", 0),
        )
        kwargs = {k: v for k, v in sim.__dict__.items()}
        print("  NOTE: simulated data. Nothing produced from it describes real football.")
    else:
        if args.root is None:
            print(f"ERROR: --root is required for source {args.source!r}")
            return 2
        kwargs = {"root": args.root}

    try:
        frames, arrivals = load_source(args.source, **kwargs)
    except (DataNotAvailable, NotImplementedError) as exc:
        print(f"\nCannot build from {args.source!r}:\n\n{exc}\n")
        return 1

    print(f"  loaded {len(frames)} frames, {len(arrivals)} arrival rows")

    prov = Provenance()
    pp = PreprocessConfig(**cfg.get("preprocess", {}))
    filtered = filter_arrivals(arrivals, pp, provenance=prov)
    keep = arrivals["arrival_id"].isin(filtered["arrival_id"]).to_numpy()
    frames = [f for f, k in zip(frames, keep) if k]
    print(f"  after inclusion criteria: {len(filtered)} arrivals ({len(filtered)/max(len(arrivals),1):.1%} retained)")

    filtered = add_subgroups(filtered)
    validate_arrivals(filtered, strict=False)

    out = outdir(f"dataset/{args.source}", config=cfg)
    write_manifest(out, script="02_build_dataset.py", config={**cfg, "source": args.source, "kwargs": kwargs})
    save_table(filtered, out, "arrivals")
    save_table(prov.to_frame(), out, "preprocessing_provenance")
    save_table(
        pd.DataFrame(
            [
                {"metric": "n_arrivals", "value": len(filtered)},
                {"metric": "n_matches", "value": filtered["match_id"].nunique()},
                {"metric": "n_competitions", "value": filtered["competition"].nunique()},
                {"metric": "base_rate_control", "value": float(filtered["y_control"].mean())},
                {"metric": "frac_endogenous", "value": float(filtered["is_endogenous"].mean())},
                {"metric": "median_frame_completeness", "value": float(filtered["frame_completeness"].median())},
                {"metric": "arrivals_per_match", "value": len(filtered) / max(filtered["match_id"].nunique(), 1)},
            ]
        ),
        out, "summary",
    )
    frames_to_npz(frames, out / "frames.npz")
    print(f"  cached {len(frames)} frames -> {out / 'frames.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
