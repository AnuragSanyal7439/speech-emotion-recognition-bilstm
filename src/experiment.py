"""Reproducible experiment preparation and artifact handling."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from config import EMOTIONS, MODEL_FILES, N_FRAMES, N_MFCC, PREPROCESSING, SCALER_FILES, artifact_name
from src.dataset import inspect_dataset, split_by_actor
from src.preprocessing import AudioError, extract_features


class ExperimentError(Exception):
    """The requested experiment cannot be prepared or loaded."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExperimentError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(result, dict):
        raise ExperimentError(f"Expected a JSON object in {path.name}.")
    return result


@contextmanager
def output_lock(directory: Path):
    """Hold an OS-managed writer lock that is released even if a process exits abruptly."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".experiment.lock").open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ExperimentError(f"Another experiment is writing to {directory}.") from exc
        yield


def prepare_dataset(root: Path, seed: int) -> tuple[pd.DataFrame, dict, str]:
    frame = split_by_actor(inspect_dataset(root), seed=seed)
    print(f"Validated {len(frame):,} RAVDESS speech clips. Checking dataset integrity.", flush=True)
    frame["sha256"] = [sha256(Path(path)) for path in frame["path"]]
    if frame.groupby("sha256")["actor"].nunique().gt(1).any():
        raise ExperimentError("An identical WAV appears under multiple actor IDs. Re-extract the speech archive.")
    frame["filename"] = frame["path"].map(lambda path: Path(path).name)
    frame = frame.sort_values("filename", ignore_index=True)
    digest = hashlib.sha256()
    for row in frame.itertuples(index=False):
        digest.update(f"{row.filename},{row.sha256},{row.split}\n".encode())
    stats = {
        "n_clips": len(frame),
        "n_actors": int(frame["actor"].nunique()),
        "n_unique_recordings": int(frame["sha256"].nunique()),
        "emotion_counts": {label: int((frame["emotion"] == label).sum()) for label in EMOTIONS},
        "splits": {
            split: {
                "n_clips": len(rows),
                "actors": sorted(int(actor) for actor in rows["actor"].unique()),
            }
            for split, rows in frame.groupby("split", sort=False)
        },
        "duration_seconds": {
            "min": float(frame["duration_seconds"].min()),
            "mean": float(frame["duration_seconds"].mean()),
            "max": float(frame["duration_seconds"].max()),
        },
    }
    return frame, stats, digest.hexdigest()


def feature_matrix(frame: pd.DataFrame, label: str) -> np.ndarray:
    count = len(frame)
    result = np.empty((count, N_FRAMES, N_MFCC), dtype=np.float32)
    print(f"Extracting MFCCs for {count:,} {label} clips.", flush=True)
    with threadpool_limits(limits=1):
        for index, path in enumerate(frame["path"]):
            try:
                _, _, result[index] = extract_features(Path(path))
            except AudioError as exc:
                raise ExperimentError(f"Could not preprocess {path}: {exc}") from exc
            if (index + 1) % 120 == 0 or index + 1 == count:
                print(f"  {index + 1:,}/{count:,}", flush=True)
    return result


def load_config(directory: Path, model: str) -> dict:
    config = read_json(directory / artifact_name(model, "training_config.json"))
    if (config.get("model") != model or config.get("emotion_labels") != list(EMOTIONS)
            or config.get("preprocessing") != PREPROCESSING):
        raise ExperimentError(f"The {model} checkpoint uses an incompatible experiment configuration.")
    return config


def verify_checkpoint(directory: Path, model: str, config: dict) -> None:
    for filename, key in (
        (MODEL_FILES[model], "model_sha256"), (SCALER_FILES[model], "scaler_sha256"),
    ):
        path = directory / filename
        if not path.is_file() or sha256(path) != config.get(key):
            raise ExperimentError(f"{path.name} is missing or mismatched. Train {model} again.")


def tensorflow_runtime(threads: int):
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def positive_int(value: str) -> int:
    from argparse import ArgumentTypeError

    number = int(value)
    if number < 1:
        raise ArgumentTypeError("must be a positive integer")
    return number


DEFAULT_THREADS = min(4, os.cpu_count() or 1)
