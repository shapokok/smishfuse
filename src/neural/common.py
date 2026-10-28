"""Shared pieces for step-3 neural baselines: config, data, seeding, persistence.

Every result is written as soon as it exists (one file per Optuna trial, per
seed), so an interrupted run can be restarted and skips finished work.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SMISHFUSE_DATA", ROOT / "data" / "processed"))
RESULTS = Path(os.environ.get("SMISHFUSE_RESULTS", ROOT / "results"))


def load_yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def load_configs() -> tuple[dict, dict]:
    """(step2 config: tasks; step3 config: protocol and models)."""
    return load_yaml("configs/step2.yaml"), load_yaml("configs/step3.yaml")


def load_split(task: dict, split: str, fraction: float | None = None):
    """ids, texts (list[str]), y (int64) for one split; optional stratified subsample (seed 0)."""
    df = pd.read_csv(DATA / task["dataset"] / f"{split}.csv", dtype=str, keep_default_na=False)
    names = df[task["label_column"]].map(task["label_map"])
    if names.isna().any():
        raise ValueError(f"unmapped labels in {task['dataset']}/{split}")
    df["y"] = names.map({c: i for i, c in enumerate(task["classes"])}).astype(np.int64)
    if fraction is not None and fraction < 1.0:
        parts = []
        for _, g in df.groupby("y", sort=True):
            n = max(1, int(round(len(g) * fraction)))
            parts.append(g.sample(n=n, random_state=0))
        df = pd.concat(parts).sort_values("id")
    return df["id"].tolist(), df["text"].tolist(), df["y"].to_numpy()


def set_seed(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def suggest(trial, space: dict) -> dict:
    out = {}
    for name, spec in space.items():
        if spec["type"] == "float":
            out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "int":
            out[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
        elif spec["type"] == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(spec)
    return out


def distributions(space: dict) -> dict:
    import optuna.distributions as d

    out = {}
    for name, spec in space.items():
        if spec["type"] == "float":
            out[name] = d.FloatDistribution(spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["type"] == "int":
            out[name] = d.IntDistribution(spec["low"], spec["high"], step=spec.get("step", 1))
        else:
            out[name] = d.CategoricalDistribution(spec["choices"])
    return out


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def raw_path(task: str, model: str, seed: int) -> Path:
    return RESULTS / "raw" / task / model / f"seed{seed}.csv"


def save_raw(task: str, model: str, seed: int, ids, y_true, y_pred_names, probs, classes, extra: dict | None = None):
    df = pd.DataFrame({"id": ids, "y_true": [classes[i] for i in y_true], "y_pred": y_pred_names})
    for i, c in enumerate(classes):
        df[f"prob_{c}"] = probs[:, i]
    for k, v in (extra or {}).items():
        df[k] = v
    path = raw_path(task, model, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    tmp.replace(path)


# Unix time after which no new trial / seed / LLM batch is started (set by kaggle_entry.py).
DEADLINE = float(os.environ.get("SMISHFUSE_DEADLINE", "inf"))


class Deferred(Exception):
    """Raised when the session deadline is too close to start the next unit of work."""


def time_left() -> float:
    return DEADLINE - time.time()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
