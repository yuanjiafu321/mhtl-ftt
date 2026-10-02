from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .artifacts import configure_logging, configure_runtime, environment_info, save_frame, save_json
from .config import ExperimentConfig, TASKS
from .data import load_raw_data, validate_frame
from .evaluation import METRICS, classification_metrics, evaluate_model, prediction_frame, save_test_evaluation
from .preprocessing import ModelPreprocessor, TabDataset, make_loaders
from .sampling import DistanceScorer, build_binary_dataset, select_negatives
from .splitting import create_split_plan
from .training import load_checkpoint, save_checkpoint, train_fold


def _state(output, stage, **extra):
    path = Path(output) / "run_state.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update({"stage": stage, **extra})
    save_json(path, payload)


def _build_partitions(reference, heldout, config, directory):
    counts = []
    directory = Path(directory)
    partitions = {"train": reference, **heldout}
    selected_ids = {partition: set() for partition in partitions}
    for task in TASKS:
        scorer = DistanceScorer(config.data, config.sampling.score_method).fit(reference.positives[task], reference.background)
        for partition, raw in partitions.items():
            selection = select_negatives(raw.background, scorer.score(raw.background),
                                          len(raw.positives[task]), config.data, config.sampling)
            frame = build_binary_dataset(raw.positives[task], selection.negatives, config.data, config.sampling)
            selected_ids[partition].update(frame[config.data.id_column])
            save_frame(directory / "datasets" / f"{task}_{partition}.csv", frame)
            save_frame(directory / "sampling" / f"{task}_{partition}_candidates.csv", selection.candidates)
            counts.append({"task": task, "partition": partition, "positive_count": int(frame[config.data.label_column].sum()),
                           "negative_count": len(selection.negatives),
                           **{f"{source}_count": int(selection.candidates.source.eq(source).sum()) for source in ("hard", "easy", "random")}})
        positive_ids = set(scorer.positive_ids)
        save_frame(directory / "sampling" / f"{task}_reference_ids.csv", pd.DataFrame({
            config.data.id_column: scorer.fit_ids,
            "is_positive": [identifier in positive_ids for identifier in scorer.fit_ids],
        }))
        (directory / "sampling").mkdir(parents=True, exist_ok=True)
        joblib.dump(scorer, directory / "sampling" / f"{task}_distance_scorer.joblib")
    names = list(selected_ids)
    if any(selected_ids[a] & selected_ids[b] for i, a in enumerate(names) for b in names[i + 1:]):
        raise RuntimeError("Training, validation, and test identifiers overlap across tasks")
    save_frame(directory / "sampling" / "counts.csv", pd.DataFrame(counts))


def prepare(data_directory, output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise FileExistsError("This run directory is already prepared; choose a new output directory")
    config.validate()
    raw, fingerprints = load_raw_data(data_directory, config.data)
    plan = create_split_plan(raw, config.data, config.split)
    save_frame(output / "splits" / "unit_assignments.csv", plan.registry)
    save_frame(output / "splits" / "fold_assignments.csv", plan.assignments)
    save_json(output / "config.json", config.to_dict())
    save_json(output / "input_manifest.json", fingerprints)
    logger = configure_logging(output)
    logger.info("Saved %d outer folds with training, validation, and test partitions", config.split.n_splits)
    for fold in range(1, config.split.n_splits + 1):
        train = plan.subset(raw, fold, "train")
        heldout = {name: plan.subset(raw, fold, name) for name in ("validation", "test")}
        _build_partitions(train, heldout, config, output / "folds" / f"fold_{fold}")
    _state(output, "prepared")
    return output


def load_run_config(output):
    return ExperimentConfig.from_dict(json.loads((Path(output) / "config.json").read_text(encoding="utf-8")))


def _read_partition(directory, name, config):
    result = {}
    for task in TASKS:
        path = Path(directory) / "datasets" / f"{task}_{name}.csv"
        frame = pd.read_csv(path, dtype={config.data.id_column: str})
        labels = frame[config.data.label_column].to_numpy()
        if set(labels) != {0, 1}:
            raise ValueError(f"{task}: the constructed dataset must contain both classes")
        normalized = validate_frame(frame, config.data, task)
        normalized[config.data.label_column] = labels.astype(int)
        result[task] = normalized
    return result


def cross_validate(output):
    fit(output)
    return evaluate(output)


def fit(output):
    output = Path(output)
    config = load_run_config(output)
    if any((output / "folds" / f"fold_{fold}" / "best_model.pth").exists()
           for fold in range(1, config.split.n_splits + 1)):
        raise FileExistsError("Fold models already exist; choose a new run directory")
    device = configure_runtime(config)
    logger = configure_logging(output)
    save_json(output / "environment.json", environment_info())
    rows = []
    for fold in range(1, config.split.n_splits + 1):
        directory = output / "folds" / f"fold_{fold}"
        train = _read_partition(directory, "train", config)
        validation = _read_partition(directory, "validation", config)
        preprocessor = ModelPreprocessor(config.data, config.split.seed).fit(train)
        preprocessor.save(directory / "model_preprocessor.joblib")
        result = train_fold(train, validation, preprocessor, config, fold, device, logger)
        save_checkpoint(directory / "best_model.pth", result.model, preprocessor, config, result.best_epoch)
        save_frame(directory / "history.csv", result.history)
        for task in TASKS:
            rows.append({"fold": fold, "task": task, "best_epoch": result.best_epoch, **result.evaluation[task]["metrics"]})
            frame = prediction_frame(validation[task], result.evaluation[task], config)
            frame.insert(0, "task", task)
            frame.insert(0, "fold", fold)
            save_frame(directory / f"{task}_validation_predictions.csv", frame)
    save_frame(output / "metrics" / "validation_fold_metrics.csv", pd.DataFrame(rows))
    _state(output, "fit_completed")
    return rows


def evaluate(output):
    output = Path(output)
    config = load_run_config(output)
    device = configure_runtime(config)
    logger = configure_logging(output)
    rows, predictions = [], {task: [] for task in TASKS}
    registry = pd.read_csv(output / "splits" / "unit_assignments.csv", dtype={config.data.id_column: str})
    for fold in range(1, config.split.n_splits + 1):
        directory = output / "folds" / f"fold_{fold}"
        model, saved_config = load_checkpoint(directory / "best_model.pth", device)
        if saved_config.to_dict() != config.to_dict():
            raise ValueError("The checkpoint configuration differs from the run configuration")
        preprocessor = ModelPreprocessor.load(directory / "model_preprocessor.joblib")
        frames = _read_partition(directory, "test", config)
        allowed = set(registry.loc[registry.outer_fold.eq(fold), config.data.id_column])
        if any(not set(frame[config.data.id_column]) <= allowed for frame in frames.values()):
            raise ValueError("A test sample is assigned to a different outer fold")
        results = evaluate_model(model, make_loaders(frames, preprocessor, config, shuffle=False), config, device)
        fold_rows = save_test_evaluation(directory, frames, results, config, fold=fold)
        for row in fold_rows:
            logger.info("Fold %d %s test: AUC=%.4f F1=%.4f", fold, row["task"], row["auc"], row["f1"])
        rows.extend(fold_rows)
        for task in TASKS:
            frame = prediction_frame(frames[task], results[task], config)
            frame.insert(1, "fold", fold)
            predictions[task].append(frame)
    metrics = pd.DataFrame(rows)
    save_frame(output / "metrics" / "test_fold_metrics.csv", metrics)
    summary = []
    combined, pooled = {}, {}
    for task in TASKS:
        part = metrics.loc[metrics.task.eq(task)]
        summary.append({"task": task, **{f"{metric}_{stat}": float(part[metric].mean() if stat == "mean" else part[metric].std(ddof=1))
                                        for metric in METRICS for stat in ("mean", "sd")}})
        frame = pd.concat(predictions[task], ignore_index=True)
        if frame[config.data.id_column].duplicated().any():
            raise RuntimeError("An identifier has multiple outer test predictions within a task")
        expected = set(registry.loc[registry[task].eq(1), config.data.id_column])
        if set(frame.loc[frame.y_true.eq(1), config.data.id_column]) != expected:
            raise RuntimeError("Outer test predictions do not cover all positive identifiers")
        combined[task] = frame
        pooled[task] = {"y_true": frame.y_true.to_numpy(), "probabilities": frame.y_prob.to_numpy(),
                        "metrics": classification_metrics(frame.y_true, frame.y_prob, config.training.threshold)}
    save_frame(output / "metrics" / "test_summary.csv", pd.DataFrame(summary))
    save_test_evaluation(output, combined, pooled, config, metrics_filename="pooled_test_metrics.csv")
    merged = pd.concat([frame.assign(task=task) for task, frame in combined.items()], ignore_index=True)
    save_frame(output / "predictions" / "test_predictions.csv", merged)
    _state(output, "evaluated")
    return rows


def run(data_directory, output, config):
    prepare(data_directory, output, config)
    return cross_validate(output)


def predict(output, input_csv, destination, device_name="cpu", fold=None):
    output = Path(output)
    device = torch.device(device_name)
    config = load_run_config(output)
    folds = range(1, config.split.n_splits + 1) if fold is None else [fold]
    if any(index not in range(1, config.split.n_splits + 1) for index in folds):
        raise ValueError("Unknown prediction fold")
    torch.set_num_threads(config.runtime.num_threads)
    raw = pd.read_csv(input_csv, dtype={config.data.id_column: str})
    frame = validate_frame(raw, config.data, "prediction")
    result = pd.DataFrame({config.data.id_column: frame[config.data.id_column]})
    predictions = {task: [] for task in TASKS}
    from torch.utils.data import DataLoader
    with torch.no_grad():
        for index in folds:
            directory = output / "folds" / f"fold_{index}"
            model, saved_config = load_checkpoint(directory / "best_model.pth", device)
            if saved_config.to_dict() != config.to_dict():
                raise ValueError("The checkpoint configuration differs from the run configuration")
            preprocessor = ModelPreprocessor.load(directory / "model_preprocessor.joblib")
            x_num, x_cat = preprocessor.transform(frame)
            dataset = TabDataset(x_num, x_cat, np.zeros(len(frame), dtype=np.float32))
            for task in TASKS:
                batch_size = config.training.debris_batch_size if task == "debris" else config.training.batch_size
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                probs = [torch.sigmoid(model(xn.to(device), xc.to(device), task)).cpu().numpy()
                         for xn, xc, _ in loader]
                predictions[task].append(np.concatenate(probs))
    for task in TASKS:
        result[f"p_{task}"] = np.mean(predictions[task], axis=0)
    save_frame(destination, result)
    return result

