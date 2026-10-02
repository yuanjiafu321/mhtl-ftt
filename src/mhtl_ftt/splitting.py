from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import TASKS
from .data import RawData


@dataclass
class SplitPlan:
    registry: pd.DataFrame
    assignments: pd.DataFrame
    id_column: str
    n_splits: int

    def subset(self, raw, fold, partition):
        if fold not in range(1, self.n_splits + 1) or partition not in {"train", "validation", "test"}:
            raise ValueError("Unknown fold or partition")
        mask = self.assignments["fold"].eq(fold) & self.assignments["partition"].eq(partition)
        identifiers = set(self.assignments.loc[mask, self.id_column])
        return RawData(
            {t: frame.loc[frame[self.id_column].isin(identifiers)].reset_index(drop=True)
             for t, frame in raw.positives.items()},
            raw.background.loc[raw.background[self.id_column].isin(identifiers)].reset_index(drop=True),
        )


def create_split_plan(raw, schema, config):
    id_col = schema.id_column
    identifiers = sorted(set(raw.background[id_col]).union(*(set(raw.positives[t][id_col]) for t in TASKS)))
    registry = pd.DataFrame({id_col: identifiers})
    for task in TASKS:
        registry[task] = registry[id_col].isin(set(raw.positives[task][id_col])).astype(int)
    registry["background"] = registry[id_col].isin(set(raw.background[id_col])).astype(int)
    registry["signature"] = registry[[*TASKS, "background"]].astype(str).agg("".join, axis=1)
    registry["outer_fold"] = 0
    rng = np.random.default_rng(config.seed)
    cursor = 0
    for _, group in registry.groupby("signature", sort=True):
        order = rng.permutation(group.index.to_numpy())
        assigned = (np.arange(len(order)) + cursor) % config.n_splits + 1
        registry.loc[order, "outer_fold"] = assigned
        cursor = (cursor + len(order)) % config.n_splits
    assignments = []
    for fold in range(1, config.n_splits + 1):
        part = registry[[id_col, "outer_fold"]].copy()
        part.insert(0, "fold", fold)
        part["partition"] = np.where(part.outer_fold.eq(fold), "test", "train")
        remaining = registry.loc[~registry.outer_fold.eq(fold)]
        for _, group in remaining.groupby("signature", sort=True):
            indices = group.index.to_numpy()
            if len(indices) > 1:
                _, val_idx = train_test_split(indices, test_size=config.validation_size,
                                              random_state=config.seed + fold - 1)
                part.loc[val_idx, "partition"] = "validation"
        assignments.append(part)
    registry = registry.drop(columns="signature")
    plan = SplitPlan(registry, pd.concat(assignments, ignore_index=True), id_col, config.n_splits)
    for fold in range(1, config.n_splits + 1):
        for partition in ("train", "validation", "test"):
            part = plan.subset(raw, fold, partition)
            if part.background.empty:
                raise ValueError(f"Fold {fold}: no background candidates in {partition}")
            for task in TASKS:
                if part.positives[task].empty:
                    raise ValueError(f"{task}: no positives in fold {fold} {partition}; increase the sample count")
                if len(part.background) < len(part.positives[task]):
                    raise ValueError(f"{task}: insufficient background candidates in fold {fold} {partition}")
    return plan

