from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import save_frame, save_json
from .config import DataConfig, TASKS


def _features(rng, latent, schema, first_id):
    n = len(latent)
    z = latent
    values = {
        "elevation": 550 + 180 * z[:, 0],
        "aspect": np.mod(180 + 65 * z[:, 1], 360),
        "plain_curvature": 0.15 * z[:, 2],
        "prof_curvature": 0.12 * z[:, 3],
        "relief": np.exp(3.0 + 0.4 * z[:, 0]),
        "roughness": np.exp(0.2 + 0.2 * z[:, 2]),
        "slope": np.clip(20 + 7 * z[:, 0] + 2 * z[:, 3], 0, 75),
        "fault_dist": np.exp(6.4 + 0.6 * z[:, 1]),
        "river_dist": np.exp(5.2 + 0.6 * z[:, 2]),
        "road_dist": np.exp(5.5 + 0.5 * z[:, 3]),
        "rainfall": 1500 + 160 * z[:, 4],
        "spi": np.exp(1.0 + 0.5 * z[:, 2]),
        "twi": 7 + z[:, 4],
        "steep_rati": 1 / (1 + np.exp(-z[:, 0])),
        "dist_tlog": 5 + 0.6 * z[:, 1],
        "ndvi": np.clip(0.55 + 0.08 * z[:, 5], 0, 1),
        "landuse": np.mod(np.floor(z[:, 4] + 4).astype(int), 5),
        "lil": np.mod(np.floor(z[:, 5] * 2 + 5).astype(int), 8),
    }
    frame = pd.DataFrame({schema.id_column: [f"SU{first_id + i:07d}" for i in range(n)]})
    for col in schema.numerical_columns:
        frame[col] = values[col] if col in values else rng.normal(size=n)
    for col in schema.categorical_columns:
        frame[col] = values[col] if col in values else rng.integers(0, 5, n)
    return frame


def generate_synthetic(directory, seed=42, counts=(120, 80, 50), pool_size=1200, schema=None):
    schema = schema or DataConfig()
    if len(counts) != 3 or min(counts) < 20 or pool_size < max(counts) * 3:
        raise ValueError("Use at least 20 positives per task and a sufficiently large background pool")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = [*schema.positive_files.values(), schema.background_file, "synthetic_manifest.json"]
    if any((directory / name).exists() for name in names):
        raise FileExistsError("The synthetic output directory already contains input files")
    rng = np.random.default_rng(seed)
    next_id = 1
    positives = {}
    shifts = (np.array([1.1, -0.4, 0.3, 0.1, 0.2, -0.2]),
              np.array([0.9, 0.4, -0.2, 0.5, 0.0, 0.3]),
              np.array([0.8, 0.0, -0.8, -0.1, 0.8, 0.1]))
    for index, (task, count) in enumerate(zip(TASKS, counts)):
        overlap = 0 if index == 0 else max(2, count // 10)
        frame = _features(rng, rng.normal(size=(count - overlap, 6)) + shifts[index], schema, next_id)
        next_id += count - overlap
        if overlap:
            frame = pd.concat([frame, positives["landslide"].iloc[:overlap]], ignore_index=True)
        positives[task] = frame
        save_frame(directory / schema.positive_files[task], frame)
    background = _features(rng, rng.normal(size=(pool_size, 6)), schema, 100000)
    save_frame(directory / schema.background_file, background)
    save_json(directory / "synthetic_manifest.json", {
        "kind": "synthetic", "seed": seed,
        "positive_counts": dict(zip(TASKS, counts)), "background_count": pool_size,
    })
    return directory

