from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from .artifacts import save_frame
from .config import DISPLAY_NAMES, TASKS

METRICS = ("auc", "acc", "f1", "pre", "rec")


def classification_metrics(y_true, probabilities, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("AUC requires both classes in the evaluation set")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and between zero and one")
    predictions = (probabilities >= threshold).astype(int)
    return {"auc": float(roc_auc_score(y_true, probabilities)),
            "acc": float(accuracy_score(y_true, predictions)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "pre": float(precision_score(y_true, predictions, zero_division=0)),
            "rec": float(recall_score(y_true, predictions, zero_division=0))}


def evaluate_model(model, loaders, config, device):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    result = {}
    with torch.no_grad():
        for task in TASKS:
            probabilities, labels, total_loss = [], [], 0.0
            for x_num, x_cat, y in loaders[task]:
                logits = model(x_num.to(device), x_cat.to(device), task)
                total_loss += criterion(logits, y.to(device)).item() * len(y)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y.numpy())
            p, y = np.concatenate(probabilities), np.concatenate(labels).astype(int)
            result[task] = {"y_true": y, "probabilities": p,
                            "loss": total_loss / len(y),
                            "metrics": classification_metrics(y, p, config.training.threshold)}
    return result


def prediction_frame(frame, result, config):
    return pd.DataFrame({config.data.id_column: frame[config.data.id_column].to_numpy(),
                         "y_true": result["y_true"], "y_prob": result["probabilities"],
                         "y_pred": (result["probabilities"] >= config.training.threshold).astype(int)})


def save_test_evaluation(output, frames, results, config, fold=None, metrics_filename="test_metrics.csv"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 6))
    for task in TASKS:
        result = results[task]
        row = {"task": task, "n_samples": len(result["y_true"]), **result["metrics"]}
        frame = prediction_frame(frames[task], result, config)
        if fold is not None:
            row["fold"] = fold
            frame.insert(1, "fold", fold)
        elif "fold" in frames[task]:
            frame.insert(1, "fold", frames[task]["fold"].to_numpy())
        rows.append(row)
        save_frame(output / "predictions" / f"{task}_test.csv", frame)
        fpr, tpr, thresholds = roc_curve(result["y_true"], result["probabilities"])
        save_frame(output / "roc" / f"{task}_points.csv", pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}))
        ax.plot(fpr, tpr, linewidth=2, label=f"{DISPLAY_NAMES[task]} (AUC={result['metrics']['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Multi-task FT-Transformer: test ROC")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    (output / "roc").mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "roc" / "test_roc.png", dpi=200)
    plt.close(fig)
    save_frame(output / "metrics" / metrics_filename, pd.DataFrame(rows))
    return rows

