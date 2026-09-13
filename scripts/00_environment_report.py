#!/usr/bin/env python3
"""Report what is installed and which parts of the study can run here.

Run this first. It distinguishes three tiers:

* **core** - everything needed for the full pipeline on simulated data;
* **loaders** - what is needed to read real provider data;
* **optional** - Model 5, alternative GBMs, Bayesian recalibration.

The point is to fail informatively at the start rather than three hours into an
experiment.
"""

from __future__ import annotations

import importlib
import json
import sys

from _common import RESULTS, banner, write_manifest

TIERS = {
    "core": ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "yaml"],
    "loaders": ["kloppy", "statsbombpy"],
    "optional": ["torch", "lightgbm", "statsmodels", "socceraction", "mplsoccer", "pytest"],
}


def main() -> int:
    banner("Environment report")
    report: dict[str, dict[str, str]] = {}
    for tier, mods in TIERS.items():
        report[tier] = {}
        print(f"\n[{tier}]")
        for mod in mods:
            try:
                m = importlib.import_module(mod)
                v = getattr(m, "__version__", "unknown")
                report[tier][mod] = v
                print(f"  {mod:16s} {v}")
            except ImportError:
                report[tier][mod] = "MISSING"
                print(f"  {mod:16s} MISSING")

    missing_core = [m for m, v in report["core"].items() if v == "MISSING"]
    print()
    if missing_core:
        print(f"BLOCKED: core packages missing: {missing_core}")
        print("  pip install -r requirements.txt")
    else:
        print("Core pipeline: OK. `python scripts/09_validate_instrument.py` will exercise it.")

    if any(v == "MISSING" for v in report["loaders"].values()):
        print("Provider loaders unavailable: real data cannot be read. Simulated runs still work.")
    if report["optional"].get("torch") == "MISSING":
        print("PyTorch absent: Model 5 (DeepSets) is skipped. It is optional by design.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "environment.json").write_text(json.dumps(report, indent=2))
    write_manifest(RESULTS, script="00_environment_report.py", config={}, extra={"report": report})
    print(f"\nWrote {RESULTS / 'environment.json'}")
    return 1 if missing_core else 0


if __name__ == "__main__":
    sys.exit(main())
