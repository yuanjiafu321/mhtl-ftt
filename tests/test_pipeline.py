import json
from dataclasses import replace

import numpy as np
import pandas as pd

from mhtl_ftt.config import ExperimentConfig, TASKS
from mhtl_ftt import pipeline
from mhtl_ftt.pipeline import cross_validate, evaluate, predict, prepare
from mhtl_ftt.preprocessing import ModelPreprocessor
from mhtl_ftt.synthetic import generate_synthetic


def test_end_to_end_five_fold_pipeline(tmp_path, monkeypatch):
    config = ExperimentConfig()
    config = replace(config,
                     model=replace(config.model, d_model=16, n_layers=1, dim_feedforward=64),
                     training=replace(config.training, max_epochs=2, patience=1),
                     runtime=replace(config.runtime, device="cpu", num_threads=1))
    inputs = generate_synthetic(tmp_path / "inputs", counts=(40, 30, 20), pool_size=300)
    output = prepare(inputs, tmp_path / "run", config)
    registry = pd.read_csv(output / "splits" / "unit_assignments.csv")
    assignments = pd.read_csv(output / "splits" / "fold_assignments.csv")
    for fold in range(1, 6):
        part = assignments.loc[assignments.fold.eq(fold)]
        forbidden = set(part.loc[~part.partition.eq("train"), "Id"])
        for task in TASKS:
            references = pd.read_csv(output / "folds" / f"fold_{fold}" / "sampling" / f"{task}_reference_ids.csv")
            assert not set(references.Id) & forbidden
    reads = []
    original_read = pipeline._read_partition

    def tracked_read(directory, name, cfg):
        reads.append(name)
        return original_read(directory, name, cfg)

    monkeypatch.setattr(pipeline, "_read_partition", tracked_read)
    first = cross_validate(output)
    assert reads == [name for _ in range(5) for name in ("train", "validation")] + ["test"] * 5
    metrics = pd.read_csv(output / "metrics" / "test_fold_metrics.csv")
    assert len(metrics) == 15 and set(metrics.fold) == set(range(1, 6))
    validation = pd.read_csv(output / "metrics" / "validation_fold_metrics.csv")
    assert len(validation) == 15 and set(validation.best_epoch) <= {1, 2}
    summary = pd.read_csv(output / "metrics" / "test_summary.csv").set_index("task")
    for fold in range(1, 6):
        pre = ModelPreprocessor.load(output / "folds" / f"fold_{fold}" / "model_preprocessor.joblib")
        part = assignments.loc[assignments.fold.eq(fold)]
        forbidden = set(part.loc[~part.partition.eq("train"), "Id"])
        assert not set(pre.fit_ids) & forbidden
    second = evaluate(output)
    assert first == second
    for task in TASKS:
        part = metrics.loc[metrics.task.eq(task)]
        np.testing.assert_allclose(summary.loc[task, "auc_mean"], part.auc.mean())
        np.testing.assert_allclose(summary.loc[task, "auc_sd"], part.auc.std(ddof=1))
        expected = pd.read_csv(output / "predictions" / f"{task}_test.csv")
        assert expected.Id.is_unique and set(expected.fold) == set(range(1, 6))
        positive_ids = set(registry.loc[registry[task].eq(1), "Id"])
        assert set(expected.loc[expected.y_true.eq(1), "Id"]) == positive_ids
        assert set(expected.columns) == {"Id", "fold", "y_true", "y_prob", "y_pred"}
        test_path = output / "folds" / "fold_1" / "datasets" / f"{task}_test.csv"
        expected = pd.read_csv(output / "folds" / "fold_1" / "predictions" / f"{task}_test.csv")
        actual = predict(output, test_path, output / f"{task}_all_tasks.csv", fold=1)
        assert actual.Id.tolist() == expected.Id.tolist()
        np.testing.assert_allclose(actual[f"p_{task}"], expected.y_prob, rtol=0, atol=1e-7)
    assert not (output / "final").exists()
    assert not (output / "cv" / "selection.json").exists()
    merged = pd.read_csv(output / "predictions" / "test_predictions.csv")
    assert not merged.duplicated(["task", "Id"]).any()
    assert json.loads((output / "run_state.json").read_text())["stage"] == "evaluated"

