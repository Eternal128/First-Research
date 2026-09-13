#!/usr/bin/env python3
"""Acquire the open datasets into ``data/raw/``.

Not part of the analysis pipeline - a one-off acquisition utility, kept
separate from the numbered scripts for that reason.

    python scripts/fetch_data.py --source metrica_sample --accept-terms
    python scripts/fetch_data.py --source skillcorner_open --accept-terms --max-matches 4
    python scripts/fetch_data.py --source statsbomb_open --accept-terms \
        --competition 43 --season 106

**Licence.** Each source has its own terms. The script prints them and refuses
to download without ``--accept-terms``, because accepting a provider's terms is
the user's act, not the script's. Nothing downloaded here may be redistributed;
``data/raw/`` is git-ignored for that reason.

Two acquisition routes are used, and the difference matters:

* **git clone** for repositories whose files are stored in git directly.
* **``media.githubusercontent.com/media/...``** for files stored in Git LFS.
  SkillCorner's tracking files are LFS pointers: a plain clone yields ~130 byte
  stub files containing an oid, not tracking data. Loading those stubs would
  fail in a confusing way well downstream, so this script fetches the real
  bytes and verifies that what arrived is not a pointer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from _common import DATA_RAW, banner

GITHUB_RAW = "https://raw.githubusercontent.com"
GITHUB_LFS = "https://media.githubusercontent.com/media"

TERMS = {
    "metrica_sample": (
        "Metrica Sports sample data (github.com/metrica-sports/sample-data).\n"
        "  The repository asks that use be responsible and that the source be\n"
        "  acknowledged if anything is made public. Read the repository's own\n"
        "  README and any terms it links before publishing."
    ),
    "skillcorner_open": (
        "SkillCorner Open Data (github.com/SkillCorner/opendata).\n"
        "  Released jointly by SkillCorner and PySport. The repository asks that\n"
        "  SkillCorner be credited if the data is used. Read its README before\n"
        "  publishing."
    ),
    "statsbomb_open": (
        "StatsBomb Open Data (github.com/statsbomb/open-data).\n"
        "  Governed by StatsBomb's own user agreement, which is in the repository\n"
        "  and which you must read. It restricts what may be done with the data,\n"
        "  including redistribution and commercial use."
    ),
}


def _get(url: str, dest: Path, *, timeout: int = 300) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "pcc-fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as fh:
        n = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            n += len(chunk)
    return n


def _get_json(url: str, *, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": "pcc-fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _is_lfs_pointer(path: Path) -> bool:
    """A Git LFS stub begins with a version line and is a few hundred bytes."""
    try:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes()[:40].startswith(b"version https://git-lfs")
    except OSError:
        return False


def _clone(url: str, dest: Path, *, timeout: int = 1800) -> None:
    if dest.exists():
        print(f"  {dest} already exists; leaving it alone")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {url} -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", url, str(dest)],
        check=True, timeout=timeout,
    )
    git_dir = dest / ".git"
    if git_dir.exists():
        subprocess.run(["rm", "-rf", str(git_dir)], check=False)


def _default_branch(url: str) -> str:
    out = subprocess.run(
        ["git", "ls-remote", "--symref", url, "HEAD"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    for line in out.splitlines():
        if line.startswith("ref:"):
            return line.split("refs/heads/")[-1].split()[0]
    return "main"


# ---------------------------------------------------------------------------
def fetch_metrica(dest: Path) -> None:
    _clone("https://github.com/metrica-sports/sample-data", dest)
    games = sorted((dest / "data").glob("Sample_Game_*")) if (dest / "data").is_dir() else []
    print(f"  {len(games)} game directories")
    csv_games = [g for g in games if (g / f"{g.name}_RawTrackingData_Home_Team.csv").exists()]
    print(f"  {len(csv_games)} in the CSV layout the adapter reads: {[g.name for g in csv_games]}")
    if len(csv_games) < len(games):
        print("  NOTE: the remainder are in the EPTS/FIFA format and need a different")
        print("        deserialiser (kloppy). They are not read by pcc.data.metrica.")


def fetch_skillcorner(dest: Path, *, max_matches: int | None) -> None:
    repo = "https://github.com/SkillCorner/opendata"
    _clone(repo, dest)
    branch = _default_branch(repo)
    matches_dir = dest / "data" / "matches"
    if not matches_dir.is_dir():
        print("  no data/matches directory found")
        return

    ids = sorted(p.name for p in matches_dir.iterdir() if p.is_dir())
    if max_matches:
        ids = ids[:max_matches]
    print(f"  resolving Git LFS tracking files for {len(ids)} match(es) on branch {branch!r}")

    for mid in ids:
        target = matches_dir / mid / f"{mid}_tracking_extrapolated.jsonl"
        if target.exists() and not _is_lfs_pointer(target):
            print(f"    {mid}: already resolved")
            continue
        rel = f"data/matches/{mid}/{target.name}"
        url = f"{GITHUB_LFS}/SkillCorner/opendata/{branch}/{rel}"
        try:
            n = _get(url, target, timeout=900)
            status = "POINTER (failed)" if _is_lfs_pointer(target) else f"{n/1e6:.1f} MB"
            print(f"    {mid}: {status}")
        except Exception as exc:
            print(f"    {mid}: FAILED - {type(exc).__name__}: {exc}")


def fetch_statsbomb(dest: Path, *, competition: int, season: int,
                    max_matches: int | None, with_360: bool) -> None:
    """Fetch one competition-season selectively.

    The full repository is large and mostly irrelevant to this study, so only
    the chosen competition's matches, events and (optionally) 360 frames are
    downloaded. 360 frames are what make the fallback design possible at all,
    and only some competitions have them.
    """
    branch = "master"
    base = f"{GITHUB_RAW}/statsbomb/open-data/{branch}/data"

    comps = _get_json(f"{base}/competitions.json")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "competitions.json").write_text(json.dumps(comps, indent=1))

    match = [c for c in comps if c["competition_id"] == competition and c["season_id"] == season]
    if not match:
        print(f"  competition {competition}/season {season} not in competitions.json")
        print("  available with 360 frames:")
        for c in comps:
            if c.get("match_available_360"):
                print(f"    {c['competition_id']:5d}/{c['season_id']:4d}  "
                      f"{c['competition_name']} {c['season_name']}")
        return
    meta = match[0]
    print(f"  {meta['competition_name']} {meta['season_name']}")
    has_360 = bool(meta.get("match_available_360"))
    print(f"  360 frames available: {has_360}")
    if with_360 and not has_360:
        print("  requested --with-360 but this competition has none; continuing without")
        with_360 = False

    matches = _get_json(f"{base}/matches/{competition}/{season}.json")
    out_matches = dest / "matches" / str(competition)
    out_matches.mkdir(parents=True, exist_ok=True)
    (out_matches / f"{season}.json").write_text(json.dumps(matches, indent=1))
    print(f"  {len(matches)} matches")

    ids = [m["match_id"] for m in matches]
    if max_matches:
        ids = ids[:max_matches]

    for kind, flag in (("events", True), ("three-sixty", with_360)):
        if not flag:
            continue
        total = 0
        failed = 0
        for mid in ids:
            target = dest / kind / f"{mid}.json"
            if target.exists():
                total += target.stat().st_size
                continue
            try:
                total += _get(f"{base}/{kind}/{mid}.json", target)
            except Exception as exc:
                failed += 1
                print(f"    {kind}/{mid}: FAILED - {type(exc).__name__}: {exc}")
        print(f"  {kind}: {len(ids) - failed}/{len(ids)} files, {total/1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(TERMS))
    ap.add_argument("--accept-terms", action="store_true",
                    help="Confirm you have read and accepted the source's terms.")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--competition", type=int, default=43, help="StatsBomb competition_id")
    ap.add_argument("--season", type=int, default=106, help="StatsBomb season_id")
    ap.add_argument("--with-360", action="store_true",
                    help="StatsBomb: also fetch 360 freeze frames (needed for the fallback design).")
    args = ap.parse_args()

    banner(f"Fetching {args.source}")
    print("Licence and terms:\n  " + TERMS[args.source] + "\n")
    if not args.accept_terms:
        print("Refusing to download. Re-run with --accept-terms once you have read the")
        print("terms above and the source's own documentation. Accepting a provider's")
        print("terms is your act, not this script's.")
        return 2

    dest = DATA_RAW / args.source
    if args.source == "metrica_sample":
        fetch_metrica(dest)
    elif args.source == "skillcorner_open":
        fetch_skillcorner(dest, max_matches=args.max_matches)
    else:
        fetch_statsbomb(dest, competition=args.competition, season=args.season,
                        max_matches=args.max_matches, with_360=args.with_360)

    print(f"\nDone. Now run:  python scripts/01_check_data_availability.py")
    print("Then work through the verification checklist in docs/data_sources.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
