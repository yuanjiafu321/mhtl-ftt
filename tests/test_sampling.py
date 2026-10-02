import numpy as np
import pandas as pd
import pytest

from mhtl_ftt.config import DataConfig, SamplingConfig
from mhtl_ftt.sampling import DistanceScorer, select_negatives


def test_mixed_sampling_extremes_and_random_pool():
    schema = DataConfig(numerical_columns=("x",), categorical_columns=())
    pool = pd.DataFrame({"Id": [str(i) for i in range(500)], "x": np.arange(500)})
    selected = select_negatives(pool, np.arange(500), 100, schema, SamplingConfig())
    groups = selected.candidates
    assert set(groups.loc[groups.source.eq("hard"), "Id"]) == set(str(i) for i in range(35))
    assert set(groups.loc[groups.source.eq("easy"), "Id"]) == set(str(i) for i in range(465, 500))
    assert groups.source.eq("random").sum() == 30
    random_ids = set(groups.loc[groups.source.eq("random"), "Id"])
    assert random_ids <= set(str(i) for i in range(35, 465))
    assert len(selected.negatives) == selected.negatives.Id.nunique() == 100
    repeat = select_negatives(pool, np.arange(500), 100, schema, SamplingConfig())
    assert selected.negatives.equals(repeat.negatives)


@pytest.mark.parametrize("mode,expected", [("hard", {"hard": 10}), ("easy", {"easy": 10})])
def test_pure_sampling(mode, expected):
    schema = DataConfig(numerical_columns=("x",), categorical_columns=())
    pool = pd.DataFrame({"Id": [str(i) for i in range(40)], "x": np.arange(40)})
    selected = select_negatives(pool, np.arange(40), 10, schema, SamplingConfig(mode=mode, beta=0))
    assert selected.candidates.loc[selected.candidates.selected, "source"].value_counts().to_dict() == expected


def test_distances_use_frozen_training_representation():
    schema = DataConfig(numerical_columns=("x",), categorical_columns=("category",))
    positives = pd.DataFrame({"Id": ["p1", "p2"], "x": [0.0, 1.0], "category": ["a", "a"]})
    background = pd.DataFrame({"Id": ["n1", "n2"], "x": [2.0, 3.0], "category": ["b", "b"]})
    heldout = pd.DataFrame({"Id": ["e1", "e2"], "x": [10.0, 12.0], "category": ["unseen", "a"]})
    scorer = DistanceScorer(schema).fit(positives, background)
    mean_before = scorer.preprocessor.named_transformers_["num"].mean_.copy()
    center_before = scorer.center.copy()
    probabilities = scorer.score(heldout)
    assert np.isfinite(probabilities).all()
    np.testing.assert_array_equal(mean_before, scorer.preprocessor.named_transformers_["num"].mean_)
    np.testing.assert_array_equal(center_before, scorer.center)
    assert not set(heldout.Id) & set(scorer.fit_ids)
    nearest = DistanceScorer(schema, "nearest_neighbour").fit(positives, background)
    assert np.all(nearest.score(positives) < 1e-7)

