from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import yaml

TASKS = ("landslide", "rockfall", "debris")
DISPLAY_NAMES = {"landslide": "Landslide", "rockfall": "Collapse", "debris": "Debris flow"}
NUMERICAL_COLUMNS = (
    "elevation", "aspect", "plain_curvature", "prof_curvature", "relief",
    "roughness", "slope", "fault_dist", "river_dist", "road_dist", "rainfall",
    "spi", "twi", "steep_rati", "dist_tlog", "ndvi",
)


@dataclass(frozen=True)
class DataConfig:
    id_column: str = "Id"
    label_column: str = "label"
    positive_files: Dict[str, str] = field(default_factory=lambda: {
        "landslide": "landslide_positive.csv", "rockfall": "collapse_positive.csv",
        "debris": "debrisflow_positive.csv",
    })
    background_file: str = "background.csv"
    numerical_columns: Tuple[str, ...] = NUMERICAL_COLUMNS
    categorical_columns: Tuple[str, ...] = ("landuse", "lil")

    @property
    def features(self):
        return list(self.numerical_columns) + list(self.categorical_columns)


@dataclass(frozen=True)
class SamplingConfig:
    mode: str = "mixed"
    score_method: str = "centroid"
    beta: float = 0.3
    hard_fraction: float = 0.5
    random_state: int = 42
    shuffle: bool = False


@dataclass(frozen=True)
class SplitConfig:
    n_splits: int = 5
    validation_size: float = 0.3
    seed: int = 10


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    dim_feedforward: int = 2048


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 0.0005
    weight_decay: float = 1.32626
    batch_size: int = 64
    debris_batch_size: int = 16
    max_epochs: int = 60
    patience: int = 20
    threshold: float = 0.5
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "landslide": 1.0, "rockfall": 1.5, "debris": 1.8,
    })


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    num_threads: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self):
        if set(self.data.positive_files) != set(TASKS):
            raise ValueError("positive_files must specify all three tasks")
        features = self.data.features
        if not features or len(set(features)) != len(features):
            raise ValueError("Feature names must be nonempty and unique")
        if self.data.id_column in features or self.data.label_column in features:
            raise ValueError("Identifiers and labels cannot be input features")
        if not 0 < self.split.validation_size < 1 or self.split.n_splits < 2:
            raise ValueError("Invalid validation fraction or fold count")
        if self.sampling.mode not in {"mixed", "hard", "easy"}:
            raise ValueError("Sampling mode must be mixed, hard, or easy")
        if self.sampling.score_method not in {"centroid", "nearest_neighbour"}:
            raise ValueError("Unknown distance method")
        if not 0 <= self.sampling.beta <= 1 or not 0 <= self.sampling.hard_fraction <= 1:
            raise ValueError("Sampling fractions must be between zero and one")
        if self.model.n_heads < 1 or self.model.d_model % self.model.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.model.d_model < 2 or self.model.n_layers < 1 or self.model.dim_feedforward < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")
        if min(self.training.batch_size, self.training.debris_batch_size,
               self.training.max_epochs, self.training.patience) < 1:
            raise ValueError("Training counts must be positive")
        if self.training.lr <= 0 or self.training.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if not 0 < self.training.threshold < 1:
            raise ValueError("Classification threshold must be in (0, 1)")
        if set(self.training.loss_weights) != set(TASKS) or any(
            value <= 0 for value in self.training.loss_weights.values()
        ):
            raise ValueError("Positive loss weights are required for all tasks")
        if self.runtime.device not in {"auto", "cpu", "cuda"} or self.runtime.num_threads < 1:
            raise ValueError("Invalid runtime settings")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        known = {"data", "sampling", "split", "model", "training", "runtime"}
        if set(values) - known:
            raise ValueError(f"Unknown configuration sections: {sorted(set(values) - known)}")
        data = dict(values.get("data", {}))
        for key in ("numerical_columns", "categorical_columns"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(
            data=DataConfig(**data), sampling=SamplingConfig(**values.get("sampling", {})),
            split=SplitConfig(**values.get("split", {})), model=ModelConfig(**values.get("model", {})),
            training=TrainingConfig(**values.get("training", {})),
            runtime=RuntimeConfig(**values.get("runtime", {})),
        ).validate()


def _merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _read_yaml(path, seen):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Circular configuration inheritance")
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a mapping")
    parent = values.pop("extends", None)
    if parent:
        return _merge(_read_yaml(path.parent / parent, seen | {path}), values)
    return values


def load_config(path=None):
    return ExperimentConfig.from_dict(_read_yaml(path, set())) if path else ExperimentConfig().validate()

