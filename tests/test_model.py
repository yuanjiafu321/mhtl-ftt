import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from mhtl_ftt.config import ExperimentConfig, TASKS, load_config
from mhtl_ftt.model import MultiTaskFTTransformer, build_model
from mhtl_ftt.preprocessing import ModelPreprocessor
from mhtl_ftt.training import load_checkpoint, save_checkpoint


def test_default_architecture():
    config = ExperimentConfig()
    model = build_model(config, [6, 9], torch.device("cpu"))
    assert model.num_weight.shape == (1, 16, 128)
    assert [embedding.num_embeddings for embedding in model.cat_embeddings] == [6, 9]
    assert len(model.transformer.layers) == 2
    layer = model.transformer.layers[0]
    assert layer.self_attn.num_heads == 4
    assert layer.norm_first
    assert layer.linear1.out_features == 2048
    assert layer.dropout.p == 0.2
    for head in (model.head_landslide, model.head_rockfall, model.head_debris):
        assert head[1].in_features == 128 and head[1].out_features == 64
        assert isinstance(head[2], torch.nn.ReLU)
        assert head[3].out_features == 1
    model.eval()
    for task in TASKS:
        assert model(torch.randn(3, 16), torch.zeros(3, 2, dtype=torch.long), task).shape == (3,)
    with pytest.raises(ValueError, match="Unknown task"):
        model(torch.randn(3, 16), torch.zeros(3, 2, dtype=torch.long), "invalid")


def test_default_configuration():
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = load_config(path)
    assert config.to_dict() == ExperimentConfig().to_dict()
    assert config.split.n_splits == 5 and config.split.validation_size == 0.3
    assert (config.model.d_model, config.model.n_heads, config.model.n_layers) == (128, 4, 2)
    assert config.model.dropout == 0.2
    assert config.training.lr == 0.0005
    assert config.training.max_epochs == 60
    assert config.training.batch_size == 64
    assert config.training.debris_batch_size == 16
    assert config.training.loss_weights == {"landslide": 1.0, "rockfall": 1.5, "debris": 1.8}


def test_unseen_categories_and_checkpoint_roundtrip(tmp_path):
    config = ExperimentConfig()
    frame = pd.DataFrame({"Id": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
    for col in config.data.numerical_columns:
        frame[col] = [0.0, 1.0, 2.0, 3.0]
    for col in config.data.categorical_columns:
        frame[col] = ["a", "a", "b", "b"]
    pre = ModelPreprocessor(config.data).fit({task: frame for task in TASKS})
    unseen = frame.iloc[[0]].copy()
    unseen["landuse"], unseen["lil"] = "new", "new"
    xn, xc = pre.transform(unseen)
    assert xc.tolist() == [[0, 0]]
    pre.save(tmp_path / "preprocessor.joblib")
    loaded = ModelPreprocessor.load(tmp_path / "preprocessor.joblib")
    np.testing.assert_array_equal(loaded.transform(unseen)[1], xc)
    model = build_model(config, pre.cardinalities, torch.device("cpu")).eval()
    save_checkpoint(tmp_path / "model.pth", model, pre, config, 1)
    restored, saved_config = load_checkpoint(tmp_path / "model.pth", torch.device("cpu"))
    assert saved_config.to_dict() == config.to_dict()
    with torch.no_grad():
        for task in TASKS:
            expected = model(torch.from_numpy(xn), torch.from_numpy(xc), task)
            actual = restored(torch.from_numpy(xn), torch.from_numpy(xc), task)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

