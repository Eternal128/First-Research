"""Leakage-safe data splitting.

Why random row-level splitting is invalid here
----------------------------------------------
Three distinct leakage channels make an ordinary random split of arrivals
optimistic, and all three operate at once:

1. **Within-possession duplication.** Consecutive passes in one possession
   share almost the same defensive shape. A random split puts pass *k* in
   training and pass *k+1* in test; the test observation is then a near copy of
   a training observation.
2. **Within-match structure.** Teams keep a shape for 90 minutes. A model can
   learn "this match's low block" and score well on held-out arrivals from the
   same match without having learned anything transferable.
3. **Player and team identity.** With enough features, a model can partially
   identify the teams. Cross-team splitting is the only way to show the result
   is about football rather than about these twenty-two players.

The splitters here therefore operate on *groups*: possession, match,
team-pairing, competition. The study's headline numbers come from the
strictest split the sample size will support - competition-level where more
than one competition is available, match-level otherwise.

A second, separate concern is **temporal** validity. If the study ever fits on
one season and predicts another, the split must respect time, because tactical
distributions drift. ``temporal_split`` implements this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    """Index arrays for one train/validation/test partition."""

    name: str
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    description: str = ""

    def sizes(self) -> dict[str, int]:
        return {"train": self.train.size, "valid": self.valid.size, "test": self.test.size}

    def assert_disjoint(self, df: pd.DataFrame, group_col: str) -> None:
        """Fail loudly if any group appears on both sides of the split."""
        g = df[group_col].to_numpy()
        tr, va, te = set(g[self.train]), set(g[self.valid]), set(g[self.test])
        overlaps = {
            "train/test": tr & te,
            "train/valid": tr & va,
            "valid/test": va & te,
        }
        bad = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
        if bad:
            raise AssertionError(f"split {self.name!r} leaks {group_col}: {bad}")


def grouped_holdout(
    df: pd.DataFrame,
    *,
    group_col: str = "match_id",
    test_frac: float = 0.3,
    valid_frac: float = 0.15,
    random_state: int = 0,
    stratify_col: str | None = None,
) -> Split:
    """Single train/valid/test split with whole groups kept together.

    ``stratify_col`` (typically ``competition`` or ``tracking_source``) balances
    the group-level composition across folds, so that a fold does not
    accidentally contain all of the broadcast-tracked matches.
    """
    rng = np.random.default_rng(random_state)
    groups = df[group_col].to_numpy()
    unique = pd.unique(groups)

    if stratify_col is not None:
        # One group's stratum is taken from its first row; a group spanning two
        # strata would be a data error and is surfaced rather than averaged.
        strata = df.groupby(group_col, observed=True)[stratify_col].first()
        n_strata = df.groupby(group_col, observed=True)[stratify_col].nunique()
        if (n_strata > 1).any():
            raise ValueError(
                f"{int((n_strata > 1).sum())} {group_col} groups span multiple {stratify_col} values"
            )
        buckets = {
            level: rng.permutation(np.asarray(members, dtype=object))
            for level, members in strata.groupby(strata).groups.items()
        }
    else:
        buckets = {"_all": rng.permutation(unique)}

    test_groups: list = []
    valid_groups: list = []
    train_groups: list = []
    for _s, arr in buckets.items():
        n = arr.size
        n_test = max(1, int(round(test_frac * n))) if n > 2 else max(0, n - 1)
        n_valid = max(1, int(round(valid_frac * n))) if n - n_test > 2 else 0
        test_groups.extend(arr[:n_test])
        valid_groups.extend(arr[n_test : n_test + n_valid])
        train_groups.extend(arr[n_test + n_valid :])

    def idx_of(gs: Sequence) -> np.ndarray:
        return np.flatnonzero(np.isin(groups, np.array(gs, dtype=object)))

    split = Split(
        name=f"holdout_by_{group_col}",
        train=idx_of(train_groups),
        valid=idx_of(valid_groups),
        test=idx_of(test_groups),
        description=(
            f"Whole {group_col} units assigned to exactly one fold"
            + (f", stratified by {stratify_col}" if stratify_col else "")
        ),
    )
    split.assert_disjoint(df, group_col)
    return split


def grouped_kfold(
    df: pd.DataFrame,
    *,
    group_col: str = "match_id",
    n_splits: int = 5,
    valid_frac_of_train: float = 0.2,
    random_state: int = 0,
) -> Iterator[Split]:
    """K folds of whole groups, each yielding a nested validation set.

    The nested validation split is carved out of the training groups - never
    out of the test fold - and is what hyper-parameters and recalibration maps
    are selected on. Selecting anything on the test fold would make the
    reported calibration a fitted quantity.
    """
    from sklearn.model_selection import GroupKFold

    groups = df[group_col].to_numpy()
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))

    for k, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(idx, groups=groups)):
        train_groups = pd.unique(groups[train_idx])
        shuffled = rng.permutation(train_groups)
        n_valid = max(1, int(round(valid_frac_of_train * shuffled.size)))
        valid_groups = shuffled[:n_valid]
        valid_mask = np.isin(groups[train_idx], valid_groups)
        split = Split(
            name=f"kfold{k}_by_{group_col}",
            train=train_idx[~valid_mask],
            valid=train_idx[valid_mask],
            test=test_idx,
            description=f"GroupKFold fold {k + 1}/{n_splits} on {group_col}",
        )
        split.assert_disjoint(df, group_col)
        yield split


def leave_one_competition_out(df: pd.DataFrame, *, competition_col: str = "competition",
                              valid_frac: float = 0.15, random_state: int = 0) -> Iterator[Split]:
    """Hold out an entire competition; the strictest available generalisation test.

    This is the split that answers RQ3's transportability component and
    hypothesis H4: a model fitted on one competition's tactical distribution
    and evaluated on another's. Only usable when the assembled corpus spans
    more than one competition, which is itself a dataset-availability question
    (see ``docs/data_sources.md``).
    """
    comps = df[competition_col].to_numpy()
    rng = np.random.default_rng(random_state)
    for comp in pd.unique(comps):
        test_idx = np.flatnonzero(comps == comp)
        rest = np.flatnonzero(comps != comp)
        if rest.size == 0:
            continue
        rest_matches = pd.unique(df.iloc[rest]["match_id"].to_numpy())
        n_valid = max(1, int(round(valid_frac * rest_matches.size)))
        valid_matches = rng.permutation(rest_matches)[:n_valid]
        valid_mask = np.isin(df.iloc[rest]["match_id"].to_numpy(), valid_matches)
        yield Split(
            name=f"loco_{comp}",
            train=rest[~valid_mask],
            valid=rest[valid_mask],
            test=test_idx,
            description=f"Leave-one-competition-out: test on {comp}",
        )


def leave_one_source_out(df: pd.DataFrame, *, source_col: str = "tracking_source") -> Iterator[Split]:
    """Train on one tracking modality, test on the other (RQ4).

    The intended use is: fit on optical tracking, evaluate on broadcast-derived
    tracking. Any degradation then confounds two things - genuinely different
    football, and degraded measurement. The study separates them by also
    running the *degradation simulation* ablation
    (``scripts/08_ablations.py``), which applies broadcast-like occlusion and
    noise to optical data and re-evaluates. Agreement between the two routes is
    the evidence that the effect is measurement rather than sampling.
    """
    sources = df[source_col].to_numpy()
    for src in pd.unique(sources):
        test_idx = np.flatnonzero(sources == src)
        train_idx = np.flatnonzero(sources != src)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        yield Split(
            name=f"loso_{src}",
            train=train_idx,
            valid=np.array([], dtype=int),
            test=test_idx,
            description=f"Leave-one-source-out: test on {src} tracking",
        )


def temporal_split(df: pd.DataFrame, *, time_col: str = "match_date",
                   test_frac: float = 0.3, valid_frac: float = 0.15) -> Split:
    """Forward-chaining split: earliest matches train, latest matches test.

    Prevents the model from being fitted on the future of the period it is
    evaluated on. Relevant whenever the corpus spans seasons; within a single
    tournament the ordering is close to arbitrary and the split degenerates
    toward a random group split, which the returned description records.
    """
    if time_col not in df.columns:
        raise KeyError(f"{time_col!r} not present; temporal validation requires match dates")
    order = df.groupby("match_id", observed=True)[time_col].min().sort_values()
    matches = order.index.to_numpy()
    n = matches.size
    n_test = max(1, int(round(test_frac * n)))
    n_valid = max(1, int(round(valid_frac * n)))
    test_m, valid_m = matches[n - n_test :], matches[n - n_test - n_valid : n - n_test]
    m = df["match_id"].to_numpy()
    split = Split(
        name="temporal",
        train=np.flatnonzero(~np.isin(m, np.concatenate([test_m, valid_m]))),
        valid=np.flatnonzero(np.isin(m, valid_m)),
        test=np.flatnonzero(np.isin(m, test_m)),
        description=f"Forward chaining on {time_col}; {n} matches ordered by first timestamp",
    )
    split.assert_disjoint(df, "match_id")
    return split


def leakage_audit(df: pd.DataFrame, split: Split, *, group_cols: Sequence[str] = ("match_id", "possession_id")) -> pd.DataFrame:
    """Report group overlap across folds for every candidate grouping column.

    Run and *saved with the results* for every experiment. A leakage audit that
    lives only in a code review is not reproducible evidence.
    """
    rows = []
    for col in group_cols:
        if col not in df.columns:
            continue
        g = df[col].to_numpy()
        tr, va, te = set(g[split.train]), set(g[split.valid]), set(g[split.test])
        rows.append(
            {
                "split": split.name,
                "group_col": col,
                "n_train_groups": len(tr),
                "n_valid_groups": len(va),
                "n_test_groups": len(te),
                "overlap_train_test": len(tr & te),
                "overlap_train_valid": len(tr & va),
                "overlap_valid_test": len(va & te),
            }
        )
    return pd.DataFrame(rows)
