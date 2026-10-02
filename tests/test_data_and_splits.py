from dataclasses import replace

import pandas as pd
import pytest

from mhtl_ftt.config import DataConfig, SplitConfig, TASKS, load_config
from mhtl_ftt.data import load_raw_data
from mhtl_ftt.splitting import create_split_plan
from mhtl_ftt.synthetic import generate_synthetic


def test_shared_ids_have_one_outer_test_fold(tmp_path):
    schema = DataConfig()
    generate_synthetic(tmp_path / "inputs", counts=(60, 40, 30), pool_size=400)
    raw, _ = load_raw_data(tmp_path / "inputs", schema)
    plan = create_split_plan(raw, schema, SplitConfig())
    assert plan.registry.Id.is_unique
    assert not plan.assignments.duplicated(["fold", "Id"]).any()
    tests = plan.assignments.loc[plan.assignments.partition.eq("test")]
    assert tests.Id.is_unique and set(tests.Id) == set(plan.registry.Id)
    repeat = create_split_plan(raw, schema, SplitConfig())
    assert repeat.registry.equals(plan.registry) and repeat.assignments.equals(plan.assignments)
    for fold in range(1, 6):
        assignments = plan.assignments.loc[plan.assignments.fold.eq(fold)]
        groups = {name: set(assignments.loc[assignments.partition.eq(name), "Id"])
                  for name in ("train", "validation", "test")}
        assert not groups["train"] & groups["validation"]
        assert not groups["train"] & groups["test"]
        assert not groups["validation"] & groups["test"]
        assert set.union(*groups.values()) == set(plan.registry.Id)
        part = {name: plan.subset(raw, fold, name) for name in groups}
        for name, target in (("train", 0.56), ("validation", 0.24), ("test", 0.2)):
            assert abs(len(part[name].background) / len(raw.background) - target) < 0.02
            for task in TASKS:
                assert set(part[name].positives[task].Id) <= groups[name]
    for task in TASKS:
        ids = []
        for fold in range(1, 6):
            part = plan.subset(raw, fold, "test")
            actual = len(part.positives[task]) / len(raw.positives[task])
            assert abs(actual - 0.2) <= 0.08
            ids.extend(part.positives[task].Id)
        assert len(ids) == len(set(ids)) == len(raw.positives[task])


def test_known_positive_cannot_enter_background(tmp_path):
    schema = DataConfig()
    generate_synthetic(tmp_path, counts=(40, 30, 20), pool_size=300)
    path = tmp_path / schema.background_file
    background = pd.read_csv(path)
    positive = pd.read_csv(tmp_path / schema.positive_files["landslide"]).iloc[[0]]
    pd.concat([background, positive], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="known positive"):
        load_raw_data(tmp_path, schema)


def test_inheritance_and_configuration_validation(tmp_path):
    (tmp_path / "base.yaml").write_text("training:\n  lr: 0.0005\n  weight_decay: 1.32626\n", encoding="utf-8")
    (tmp_path / "small.yaml").write_text("extends: base.yaml\ntraining:\n  max_epochs: 2\n", encoding="utf-8")
    config = load_config(tmp_path / "small.yaml")
    assert config.training.max_epochs == 2
    assert config.training.lr == 0.0005
    assert config.training.weight_decay == 1.32626
    with pytest.raises(ValueError, match="divisible"):
        replace(config, model=replace(config.model, d_model=127)).validate()
    with pytest.raises(ValueError, match="validation"):
        replace(config, split=replace(config.split, validation_size=1)).validate()

