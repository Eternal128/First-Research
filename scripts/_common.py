"""Shared plumbing for the analysis scripts: config, paths, seeding, manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_RAW = REPO_ROOT / "data" / "raw"
RESULTS = REPO_ROOT / "results"
CONFIGS = REPO_ROOT / "configs"


def load_config(name: str = "default.yaml") -> dict:
    """Load a YAML config from ``configs/``, falling back to JSON if PyYAML is absent."""
    path = CONFIGS / name if not Path(name).is_absolute() else Path(name)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    text = path.read_text()
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        raise ImportError(
            "PyYAML is required to read configs. `pip install pyyaml`, or pass a JSON config."
        ) from None


def outdir(name: str, *, config: dict | None = None) -> Path:
    """Create and return a results directory for one experiment.

    A relative ``output.root`` in the config is resolved against the repository
    root, not the current working directory, so a script run from anywhere
    writes to the same place.
    """
    root = Path((config or {}).get("output", {}).get("root", RESULTS))
    if not root.is_absolute():
        root = REPO_ROOT / root
    d = root / name
    (d / "figures").mkdir(parents=True, exist_ok=True)
    (d / "tables").mkdir(parents=True, exist_ok=True)
    return d




def _corpus_cache_key(source: str, root: str, label_cfg, config: dict,
                      loader_kwargs: dict | None = None) -> str:
    """Hash of everything that changes what the parsed corpus contains.

    Deliberately includes the raw-data fingerprint (file count, total size and
    latest mtime) as well as the label and preprocessing settings, so adding
    matches to ``data/raw`` invalidates the cache rather than silently serving a
    stale, smaller corpus - which would be a very quiet way to report results
    from the wrong sample.
    """
    import hashlib
    import json
    from pathlib import Path

    files = sorted(p for p in Path(root).rglob("*") if p.is_file())
    fingerprint = {
        "n_files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "latest_mtime": max((p.stat().st_mtime_ns for p in files), default=0),
    }
    payload = {
        "source": source,
        "fingerprint": fingerprint,
        "label": {"definition": label_cfg.definition, "horizon": label_cfg.horizon,
                  "censor": label_cfg.censor_on_stoppage},
        "preprocess": config.get("preprocess", {}),
        "loader_kwargs": loader_kwargs or {},
        "schema_version": 3,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _load_with_cache(source: str, root: str, label_cfg, config: dict, *, use_cache: bool = True,
                     loader_kwargs: dict | None = None):
    """Parse a corpus, caching the result under the scratch cache directory."""
    import pandas as pd

    from pcc.data import load_source
    from pcc.data.tracking import load_frames, save_frames

    cache_dir = REPO_ROOT / "results" / "cache"
    key = _corpus_cache_key(source, root, label_cfg, config, loader_kwargs)
    table_path = cache_dir / f"{source}_{key}.csv"
    frames_path = cache_dir / f"{source}_{key}.npz"

    if use_cache and table_path.exists() and frames_path.exists():
        try:
            arrivals = pd.read_csv(table_path)
            frames = load_frames(frames_path)
            if len(frames) == len(arrivals):
                print(f"  (cache hit: {len(arrivals)} arrivals from {table_path.name})")
                return frames, arrivals
            print("  (cache discarded: frame/table length mismatch)")
        except Exception as exc:
            print(f"  (cache unreadable, re-parsing: {type(exc).__name__})")

    frames, arrivals = load_source(source, root=root, label_config=label_cfg, **(loader_kwargs or {}))
    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            arrivals.to_csv(table_path, index=False)
            save_frames(frames, frames_path)
        except Exception as exc:  # caching is a convenience, never a hard failure
            print(f"  (cache write failed: {type(exc).__name__}: {exc})")
    return frames, arrivals


def load_corpus(source: str, config: dict, *, root: str | None = None, use_cache: bool = True,
                label_overrides: dict | None = None, loader_kwargs: dict | None = None):
    """Load a corpus and apply the study's inclusion criteria and subgroups.

    One code path for every source, so an analysis script never needs to know
    which provider produced the data. Parsing is cached on disk (see
    :func:`_load_with_cache`); pass ``use_cache=False`` to force a re-parse. Returns ``(frames, arrivals_table)`` with
    the table reset to a contiguous index that matches the frame list
    positionally - every downstream split indexes both by position, so the two
    must not drift apart.
    """
    from pcc.data import SimulationConfig, filter_arrivals, load_source
    from pcc.data.labels import LabelConfig
    from pcc.data.preprocess import PreprocessConfig, Provenance
    from pcc.evaluation import add_subgroups

    labels = {**config.get("labels", {}), **(label_overrides or {})}
    label_cfg = LabelConfig(
        definition=labels.get("definition", "controlled_at_horizon"),
        horizon=float(labels.get("horizon", 1.0)),
        censor_on_stoppage=bool(labels.get("censor_on_stoppage", True)),
    )

    if source == "simulated":
        sim = SimulationConfig(
            n_matches=config["simulation"]["n_matches"],
            arrivals_per_match=config["simulation"]["arrivals_per_match"],
            control_horizon=label_cfg.horizon,
            random_state=config.get("random_state", 0),
        )
        frames, arrivals = load_source("simulated", **sim.__dict__)
    else:
        if root is None:
            root = str(DATA_RAW / source)
        frames, arrivals = _load_with_cache(
            source, root, label_cfg, config, use_cache=use_cache,
            loader_kwargs=loader_kwargs or {},
        )

    prov = Provenance()
    pp = PreprocessConfig(**config.get("preprocess", {}))
    filtered = filter_arrivals(arrivals, pp, provenance=prov)

    keep = arrivals["arrival_id"].isin(filtered["arrival_id"]).to_numpy()
    if keep.sum() != len(filtered):
        raise AssertionError(
            f"filter kept {len(filtered)} rows but the mask selects {int(keep.sum())} frames; "
            "arrival_id is not unique or the filter reordered rows"
        )
    frames = [f for f, k in zip(frames, keep) if k]
    table = add_subgroups(filtered).reset_index(drop=True)
    if len(frames) != len(table):
        raise AssertionError(f"frame/table length mismatch: {len(frames)} vs {len(table)}")
    return frames, table, prov


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:
        return True


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def write_manifest(out: Path, *, script: str, config: dict, extra: dict | None = None) -> Path:
    """Record everything needed to explain how a result was produced.

    A results directory without a manifest is not reproducible, so every script
    writes one before it writes anything else. ``git_dirty`` is recorded rather
    than hidden: a result produced from an uncommitted tree is a fact the reader
    should know.
    """
    packages = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "kloppy", "torch"):
        try:
            m = __import__(mod)
            packages[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            packages[mod] = "not installed"

    manifest = {
        "script": script,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "git_dirty": git_dirty(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "config": _jsonable(config),
        "config_sha256": hashlib.sha256(
            json.dumps(_jsonable(config), sort_keys=True).encode()
        ).hexdigest(),
        **(_jsonable(extra) if extra else {}),
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path



def _rel(path: Path) -> str:
    """Repo-relative path for display, falling back to the absolute path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save_table(df, out: Path, name: str) -> Path:
    """Write a tidy table as CSV. CSV, not pickle: results must outlive this code."""
    path = out / "tables" / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {_rel(path)}  ({len(df)} rows)")
    return path


def save_figure(fig, out: Path, name: str, *, dpi: int = 150) -> Path:
    path = out / "figures" / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        fig.savefig(out / "figures" / f"{name}.pdf", bbox_inches="tight")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  wrote {_rel(path)}")
    return path


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
