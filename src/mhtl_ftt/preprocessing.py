from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer
from torch.utils.data import DataLoader, Dataset

from .config import TASKS


class ModelPreprocessor:
    def __init__(self, schema, seed=10):
        self.schema = schema
        self.seed = seed

    def fit(self, train_frames):
        combined = pd.concat([train_frames[task] for task in TASKS], ignore_index=True)
        self.quantile = None
        self.ordinal = None
        if self.schema.numerical_columns:
            self.quantile = QuantileTransformer(n_quantiles=min(len(combined), 1000),
                                                output_distribution="normal", random_state=self.seed)
            self.quantile.fit(combined[list(self.schema.numerical_columns)].to_numpy())
        if self.schema.categorical_columns:
            self.ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self.ordinal.fit(combined[list(self.schema.categorical_columns)].to_numpy())
        self.fit_ids = sorted(set(combined[self.schema.id_column]))
        return self

    @property
    def cardinalities(self):
        return [len(categories) + 1 for categories in self.ordinal.categories_] if self.ordinal else []

    def transform(self, frame):
        x_num = (self.quantile.transform(frame[list(self.schema.numerical_columns)].to_numpy()).astype(np.float32)
                 if self.quantile else np.empty((len(frame), 0), dtype=np.float32))
        x_cat = (self.ordinal.transform(frame[list(self.schema.categorical_columns)].to_numpy()).astype(np.int64) + 1
                 if self.ordinal else np.empty((len(frame), 0), dtype=np.int64))
        return x_num, x_cat

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class TabDataset(Dataset):
    def __init__(self, x_num, x_cat, y):
        self.x_num = torch.from_numpy(x_num).float()
        self.x_cat = torch.from_numpy(x_cat).long()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x_num[index], self.x_cat[index], self.y[index]


def make_loaders(frames, preprocessor, config, shuffle):
    loaders = {}
    for task in TASKS:
        frame = frames[task]
        x_num, x_cat = preprocessor.transform(frame)
        y = frame[config.data.label_column].to_numpy(dtype=np.float32)
        batch_size = config.training.debris_batch_size if task == "debris" else config.training.batch_size
        loaders[task] = DataLoader(TabDataset(x_num, x_cat, y), batch_size=batch_size,
                                   shuffle=shuffle, drop_last=False, num_workers=0)
    return loaders

