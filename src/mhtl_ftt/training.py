from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .artifacts import set_seed
from .config import TASKS
from .evaluation import METRICS, evaluate_model
from .model import build_model
from .preprocessing import make_loaders


@dataclass
class FoldResult:
    model: torch.nn.Module
    best_epoch: int
    evaluation: dict
    history: pd.DataFrame


def train_epoch(model, loaders, optimizer, weights, device):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_sum, steps = 0.0, 0
    for batches in itertools.zip_longest(*(loaders[task] for task in TASKS), fillvalue=None):
        optimizer.zero_grad()
        losses = []
        for task, batch in zip(TASKS, batches):
            if batch is not None:
                x_num, x_cat, y = batch
                losses.append(weights[task] * criterion(model(x_num.to(device), x_cat.to(device), task), y.to(device)))
        if not losses:
            continue
        loss = torch.stack(losses).sum()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        steps += 1
    return loss_sum / max(steps, 1)


def train_fold(train_frames, validation_frames, preprocessor, config, fold, device, logger):
    set_seed(config.split.seed + fold - 1)
    model = build_model(config, preprocessor.cardinalities, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr,
                                  weight_decay=config.training.weight_decay)
    train_loaders = make_loaders(train_frames, preprocessor, config, shuffle=True)
    validation_loaders = make_loaders(validation_frames, preprocessor, config, shuffle=False)
    best_auc, best_epoch, best_state, best_evaluation = -np.inf, 1, None, None
    stale, rows = 0, []
    for epoch in range(1, config.training.max_epochs + 1):
        loss = train_epoch(model, train_loaders, optimizer, config.training.loss_weights, device)
        evaluated = evaluate_model(model, validation_loaders, config, device)
        mean_auc = float(np.mean([evaluated[task]["metrics"]["auc"] for task in TASKS]))
        row = {"epoch": epoch, "train_loss": loss, "val_mean_auc": mean_auc,
               "val_loss_total": sum(config.training.loss_weights[t] * evaluated[t]["loss"] for t in TASKS)}
        for task in TASKS:
            row[f"{task}_val_loss"] = evaluated[task]["loss"]
            row.update({f"{task}_{metric}": evaluated[task]["metrics"][metric] for metric in METRICS})
        rows.append(row)
        logger.info("Fold %d epoch %d: train_loss=%.4f val_auc=%.4f", fold, epoch, loss, mean_auc)
        if mean_auc > best_auc:
            best_auc, best_epoch = mean_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            best_evaluation = copy.deepcopy(evaluated)
            stale = 0
        else:
            stale += 1
            if stale >= config.training.patience:
                logger.info("Fold %d stopped at epoch %d", fold, epoch)
                break
    model.load_state_dict(best_state)
    return FoldResult(model, best_epoch, best_evaluation, pd.DataFrame(rows))


def save_checkpoint(path, model, preprocessor, config, epochs):
    torch.save({"state_dict": model.state_dict(), "config": config.to_dict(),
                "cat_cardinalities": preprocessor.cardinalities, "epochs": int(epochs)}, path)


def load_checkpoint(path, device):
    from .config import ExperimentConfig
    payload = torch.load(path, map_location=device, weights_only=True)
    config = ExperimentConfig.from_dict(payload["config"])
    model = build_model(config, payload["cat_cardinalities"], device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config

