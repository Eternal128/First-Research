#!/usr/bin/env python3
"""Check which data sources are actually present and reachable. Assert nothing.

This script exists because the proposal must not claim a dataset's contents
without checking. It does three things:

1. Reports what is on disk under ``data/raw/<key>/``.
2. Optionally probes each source's documented URL for reachability
   (``--probe-network``). Reachability says nothing about licence or contents.
3. Writes ``results/data_availability.json``, which the data section of any
   write-up should cite instead of restating provider marketing copy.

It deliberately does **not** download anything. Provider terms differ and
several require the user to accept them personally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import DATA_RAW, RESULTS, banner, write_manifest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcc.data.sources import FALLBACK_REQUIREMENTS, PRIMARY_REQUIREMENTS, SOURCES  # noqa: E402


def probe(url: str, timeout: float = 10.0) -> dict:
    """HEAD/GET the URL and report the status. Network failures are results, not errors."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "pcc-availability-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"reachable": True, "http_status": resp.status, "interpretation": "endpoint responded"}
    except urllib.error.HTTPError as e:
        # A 403/407 from a corporate or sandbox egress proxy says nothing about
        # the dataset. Conflating "my network blocked this" with "this dataset
        # does not exist" is precisely the kind of unverified claim this script
        # exists to prevent.
        blocked = e.code in (403, 407)
        return {
            "reachable": False,
            "http_status": e.code,
            "error": str(e),
            "interpretation": (
                "BLOCKED BY LOCAL NETWORK POLICY - tells you nothing about the dataset; "
                "re-run from an unrestricted network before recording a conclusion"
                if blocked
                else "endpoint responded with an error status"
            ),
        }
    except Exception as e:
        return {
            "reachable": False,
            "http_status": None,
            "error": f"{type(e).__name__}: {e}",
            "interpretation": "no response; could be the network, not the source",
        }


def inspect_local(key: str) -> dict:
    d = DATA_RAW / key
    if not d.exists():
        return {"present": False, "path": str(d), "n_files": 0, "bytes": 0}
    files = [p for p in d.rglob("*") if p.is_file()]
    return {
        "present": bool(files),
        "path": str(d),
        "n_files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "sample_files": [str(p.relative_to(d)) for p in files[:10]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-network", action="store_true",
                    help="Attempt an HTTP request to each documented URL.")
    args = ap.parse_args()

    banner("Data availability check")
    print("This check verifies presence and reachability only.")
    print("It does NOT verify dataset contents, field inventories, or licence terms.\n")

    report = {}
    for key, src in SOURCES.items():
        entry = {
            "name": src.name,
            "modality": src.modality,
            "registry_status": src.status,
            "access": src.access,
            "url": src.url,
            "licence_note": src.licence_note,
            "missing_for_primary_design": list(src.missing(PRIMARY_REQUIREMENTS)),
            "missing_for_fallback_design": list(src.missing(FALLBACK_REQUIREMENTS)),
            "local": inspect_local(key),
        }
        if args.probe_network and src.url:
            entry["network"] = probe(src.url)
        report[key] = entry

        local = entry["local"]
        mark = "present" if local["present"] else "absent"
        net = ""
        if "network" in entry:
            n = entry["network"]
            if n["reachable"]:
                net = " | url reachable"
            elif n.get("http_status") in (403, 407):
                net = " | url BLOCKED by local network (inconclusive)"
            else:
                net = f" | url no response ({n.get('http_status')})"
        print(f"  {key:20s} local: {mark:8s} ({local['n_files']} files){net}")

    if args.probe_network and all(
        v.get("network", {}).get("http_status") in (403, 407)
        for v in report.values() if "network" in v
    ) and any("network" in v for v in report.values()):
        print("\n  NOTE: every probe was blocked by this network's egress policy.")
        print("        The network column above is inconclusive and must not be cited.")

    usable = [k for k, v in report.items() if v["local"]["present"] and not v["missing_for_primary_design"]]
    print("\nSources on disk that meet the primary design's field requirements (per the")
    print(f"registry's *unverified* claims): {usable or 'none'}")
    if not usable:
        print("\nNo primary-design source is present. Options:")
        print("  - run the pipeline on simulated data (scripts/09_validate_instrument.py);")
        print("  - obtain a source under its own terms into data/raw/<key>/;")
        print("  - adopt the freeze-frame fallback design (see docs/proposal.md, Section 24).")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "data_availability.json").write_text(json.dumps(report, indent=2))
    write_manifest(RESULTS, script="01_check_data_availability.py",
                   config={"probe_network": args.probe_network}, extra={"summary_usable": usable})
    print(f"\nWrote {RESULTS / 'data_availability.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
