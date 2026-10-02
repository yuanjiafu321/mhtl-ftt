from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def save_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(config):
    torch.set_num_threads(config.runtime.num_threads)
    if config.runtime.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device("cuda" if config.runtime.device == "auto" and torch.cuda.is_available()
                        else "cpu" if config.runtime.device == "auto" else config.runtime.device)


def environment_info():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}


def configure_logging(output):
    logger = logging.getLogger("mhtl_ftt")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(Path(output) / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

