from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Selection:
    negatives: pd.DataFrame
    candidates: pd.DataFrame


class DistanceScorer:
    def __init__(self, schema, method="centroid"):
        self.schema = schema
        self.method = method

    def fit(self, positives, background):
        if positives.empty or background.empty:
            raise ValueError("Distance fitting requires positives and a background pool")
        transformers = []
        if self.schema.numerical_columns:
            transformers.append(("num", StandardScaler(), list(self.schema.numerical_columns)))
        if self.schema.categorical_columns:
            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
            transformers.append(("cat", encoder, list(self.schema.categorical_columns)))
        self.preprocessor = ColumnTransformer(transformers, remainder="drop")
        self.preprocessor.fit(pd.concat([positives[self.schema.features], background[self.schema.features]], ignore_index=True))
        self.positive_features = self.preprocessor.transform(positives[self.schema.features])
        self.center = np.asarray(self.positive_features.mean(axis=0)).reshape(1, -1)
        if self.method == "nearest_neighbour":
            self.neighbours = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(self.positive_features)
        elif self.method != "centroid":
            raise ValueError("Unknown distance method")
        self.fit_ids = pd.concat([positives[self.schema.id_column], background[self.schema.id_column]]).unique().tolist()
        self.positive_ids = positives[self.schema.id_column].tolist()
        return self

    def score(self, background):
        x = self.preprocessor.transform(background[self.schema.features])
        if self.method == "nearest_neighbour":
            return self.neighbours.kneighbors(x, return_distance=True)[0].ravel()
        return euclidean_distances(x, self.center).ravel()


def sampling_counts(n_neg, config):
    if int(n_neg) != n_neg or n_neg < 1:
        raise ValueError("The negative count must be a positive integer")
    n_random = int(np.floor(n_neg * config.beta + 0.5))
    n_main = int(n_neg) - n_random
    n_hard = n_main if config.mode == "hard" else 0 if config.mode == "easy" else int(np.floor(n_main * config.hard_fraction + 0.5))
    return n_hard, n_main - n_hard, n_random


def select_negatives(background, scores, n_neg, schema, config):
    if n_neg > len(background):
        raise ValueError("The partition's background pool is smaller than the requested negative count")
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(background) or not np.isfinite(scores).all():
        raise ValueError("Invalid candidate distance scores")
    rng = np.random.default_rng(config.random_state)
    shuffled = rng.permutation(len(scores))
    order = shuffled[np.argsort(scores[shuffled], kind="stable")]
    n_hard, n_easy, n_random = sampling_counts(n_neg, config)
    hard = order[:n_hard]
    easy = order[len(order) - n_easy:] if n_easy else np.array([], dtype=int)
    main = np.concatenate([hard, easy])
    remaining = np.setdiff1d(np.arange(len(scores)), main)
    random = rng.choice(remaining, size=n_random, replace=False)
    selected = np.concatenate([main, random])
    if len(selected) != n_neg or len(np.unique(selected)) != n_neg:
        raise RuntimeError("Negative selection failed its count or uniqueness check")
    sources = np.full(len(scores), "unselected", dtype=object)
    sources[hard], sources[easy], sources[random] = "hard", "easy", "random"
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    candidates = pd.DataFrame({schema.id_column: background[schema.id_column].to_numpy(),
                               "distance": scores, "rank": ranks, "source": sources,
                               "selected": sources != "unselected"})
    return Selection(background.iloc[selected].copy().reset_index(drop=True), candidates)


def build_binary_dataset(positives, negatives, schema, sampling):
    if len(positives) != len(negatives):
        raise ValueError("The positive-to-negative ratio must be 1:1")
    if set(positives[schema.id_column]) & set(negatives[schema.id_column]):
        raise ValueError("Positive and negative identifiers overlap")
    positive, negative = positives.copy(), negatives.copy()
    positive[schema.label_column] = 1
    negative[schema.label_column] = 0
    frame = pd.concat([positive, negative], ignore_index=True)
    if sampling.shuffle:
        frame = frame.sample(frac=1, random_state=sampling.random_state).reset_index(drop=True)
    return frame

