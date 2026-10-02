from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .artifacts import file_digest
from .config import TASKS


@dataclass
class RawData:
    positives: Dict[str, pd.DataFrame]
    background: pd.DataFrame


def validate_frame(frame, schema, name):
    required = [schema.id_column, *schema.features]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    frame = frame[required].copy()
    if frame.isna().any().any():
        raise ValueError(f"{name}: missing values")
    frame[schema.id_column] = frame[schema.id_column].astype(str).str.strip()
    if frame[schema.id_column].eq("").any() or frame[schema.id_column].duplicated().any():
        raise ValueError(f"{name}: identifiers must be nonempty and unique")
    for col in schema.numerical_columns:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    if schema.numerical_columns and not np.isfinite(frame[list(schema.numerical_columns)].to_numpy()).all():
        raise ValueError(f"{name}: numerical values must be finite")
    for col in schema.categorical_columns:
        frame[col] = frame[col].astype(str)
    if frame.empty:
        raise ValueError(f"{name}: input is empty")
    return frame.reset_index(drop=True)


def load_raw_data(directory, schema):
    directory = Path(directory)
    sources = {task: directory / schema.positive_files[task] for task in TASKS}
    sources["background"] = directory / schema.background_file
    frames = {}
    for name, path in sources.items():
        frames[name] = validate_frame(pd.read_csv(path, dtype={schema.id_column: str}), schema, name)
    positive_ids = set().union(*(set(frames[t][schema.id_column]) for t in TASKS))
    if positive_ids & set(frames["background"][schema.id_column]):
        raise ValueError("The background pool includes known positive identifiers")
    shared = pd.concat([frames[t] for t in TASKS], ignore_index=True)
    for _, group in shared[shared[schema.id_column].duplicated(keep=False)].groupby(schema.id_column):
        if any(group[col].nunique() > 1 for col in schema.features):
            raise ValueError("A shared identifier has conflicting conditioning factors")
    fingerprints = {name: {"filename": path.name, "sha256": file_digest(path), "rows": len(frames[name])}
                    for name, path in sources.items()}
    return RawData({task: frames[task] for task in TASKS}, frames["background"]), fingerprints

