"""Train either recurrent classifier or both using actor-disjoint validation."""

import argparse
import math
import platform
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from config import (
    COMPARISON_FILES, DATASET_DIR, EMOTIONS, EVALUATION_FILES, MODEL_FILES, MODEL_NAMES,
    OUTPUTS_DIR, PREPROCESSING, SCALER_FILES, artifact_name,
)
from src.dataset import DatasetError
from src.experiment import (
    DEFAULT_THREADS, ExperimentError, feature_matrix, load_config, output_lock,
    positive_int, prepare_dataset, sha256, tensorflow_runtime, write_json,
)
from src.preprocessing import AudioError, FeatureScaler, extract_features
from src.visualization import (
    architecture_figure, dataset_figures, mfcc_figure, save_figure, training_figure,
    waveform_figure,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bilstm", "lstm", "both"), default="bilstm")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--epochs", type=positive_int, default=60)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=DEFAULT_THREADS)
    args = parser.parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    return args


def batches(tf, features, labels, args, *, shuffle: bool = False):
    data = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        data = data.shuffle(len(labels), seed=args.seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return data.batch(args.batch_size).with_options(options).prefetch(1)


def train_model(tf, model_key, args, train, validation, scaler, signature) -> None:
    from src.models import build_model

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    model = build_model(model_key, args.learning_rate)
    name = MODEL_NAMES[model_key]
    training_data = batches(tf, *train, args, shuffle=True)
    validation_data = batches(tf, *validation, args)
    print(f"\nTraining {name} ({model.count_params():,} parameters).", flush=True)

    with TemporaryDirectory(prefix=".training-", dir=args.output_dir) as temp:
        stage = Path(temp)
        checkpoint = stage / MODEL_FILES[model_key]
        scaler_path = stage / SCALER_FILES[model_key]
        manifest_name = artifact_name(model_key, "training_config.json")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_loss", mode="min", save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", mode="min", patience=4, factor=0.5, min_lr=1e-6,
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        start = perf_counter()
        fit = model.fit(training_data, validation_data=validation_data, shuffle=False,
                        epochs=args.epochs, callbacks=callbacks, verbose=2)
        elapsed = perf_counter() - start
        history = pd.DataFrame(fit.history)
        required = ["accuracy", "val_accuracy", "loss", "val_loss"]
        if (history.empty or not set(required).issubset(history)
                or not np.isfinite(history[required].to_numpy()).all() or not checkpoint.is_file()):
            raise ExperimentError(f"{name} training ended without a valid checkpoint and history.")
        history.insert(0, "epoch", np.arange(1, len(history) + 1))
        best = history.loc[history["val_loss"].idxmin()]
        history.to_csv(stage / artifact_name(model_key, "training_history.csv"), index=False)
        scaler.save(scaler_path)
        lines: list[str] = []
        model.summary(print_fn=lines.append, line_length=100)
        (stage / artifact_name(model_key, "model_summary.txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )
        save_figure(architecture_figure(model, name),
                    stage / artifact_name(model_key, "model_architecture.png"))
        for metric in ("accuracy", "loss"):
            save_figure(training_figure(history, metric, name),
                        stage / artifact_name(model_key, f"training_{metric}.png"))
        write_json(stage / manifest_name, {
            "model": model_key, "emotion_labels": list(EMOTIONS), "preprocessing": PREPROCESSING,
            "dataset_split_sha256": signature, "seed": args.seed, "epochs_requested": args.epochs,
            "epochs_run": len(history), "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "threads": args.threads,
            "python_version": platform.python_version(), "tensorflow_version": tf.__version__,
            "keras_version": tf.keras.__version__, "parameters": model.count_params(),
            "early_stopping_patience": 10, "lr_plateau_patience": 4, "checkpoint_metric": "val_loss",
            "best_epoch": int(best["epoch"]), "best_val_loss": float(best["val_loss"]),
            "best_epoch_val_accuracy": float(best["val_accuracy"]), "training_seconds": elapsed,
            "model_sha256": sha256(checkpoint), "scaler_sha256": sha256(scaler_path),
        })
        # Publish the manifest last so readers can reject incomplete bundles.
        (args.output_dir / manifest_name).unlink(missing_ok=True)
        for filename in (*COMPARISON_FILES, *(artifact_name(model_key, p) for p in EVALUATION_FILES)):
            (args.output_dir / filename).unlink(missing_ok=True)
        for path in stage.iterdir():
            if path.name != manifest_name:
                path.replace(args.output_dir / path.name)
        (stage / manifest_name).replace(args.output_dir / manifest_name)
        print(f"{name} best epoch {int(best['epoch'])} by validation loss "
              f"({best['val_loss']:.4f}, validation accuracy {best['val_accuracy']:.2%}).", flush=True)


def run(args: argparse.Namespace) -> None:
    frame, stats, signature = prepare_dataset(args.dataset_dir, args.seed)
    keys = tuple(MODEL_FILES) if args.model == "both" else (args.model,)
    with output_lock(args.output_dir):
        for key in MODEL_FILES:
            if (args.output_dir / artifact_name(key, "training_config.json")).exists():
                other = load_config(args.output_dir, key)
                if other.get("dataset_split_sha256") != signature:
                    raise ExperimentError("The existing model used a different dataset or split. "
                                          "Use --output-dir for the new experiment.")
        training_rows = frame[frame["split"] == "train"]
        validation_rows = frame[frame["split"] == "validation"]
        raw_train = feature_matrix(training_rows, "training")
        raw_validation = feature_matrix(validation_rows, "validation")
        scaler = FeatureScaler.fit(raw_train)
        train = (scaler.transform(raw_train), training_rows["label"].to_numpy(dtype=np.int32))
        validation = (scaler.transform(raw_validation), validation_rows["label"].to_numpy(dtype=np.int32))
        tf = tensorflow_runtime(args.threads)
        tf.config.experimental.enable_op_determinism()
        with TemporaryDirectory(prefix=".dataset-", dir=args.output_dir) as temp:
            stage = Path(temp)
            example = training_rows.iloc[0]
            audio, rate, mfcc = extract_features(Path(example["path"]))
            stats.update(dataset_split_sha256=signature, example_filename=example["filename"])
            write_json(stage / "dataset_statistics.json", stats)
            frame.drop(columns=["path"]).to_csv(stage / "dataset_manifest.csv", index=False)
            dataset_figures(frame, stage)
            save_figure(waveform_figure(audio, rate), stage / "waveform.png")
            save_figure(mfcc_figure(mfcc), stage / "mfcc_heatmap.png")
            for path in stage.iterdir():
                path.replace(args.output_dir / path.name)
        for key in keys:
            train_model(tf, key, args, train, validation, scaler, signature)
    print(f"Training artifacts saved to {args.output_dir.resolve()}. "
          "Run python evaluate.py to score the held-out actors.", flush=True)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (DatasetError, AudioError, ExperimentError, OSError) as exc:
        sys.exit(f"Training could not finish: {exc}")
