"""Evaluate saved checkpoints on the original, held-out RAVDESS actors."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from config import (
    COMPARISON_FILES, DATASET_DIR, EMOTIONS, MODEL_FILES, MODEL_NAMES, N_FRAMES, N_MFCC,
    OUTPUTS_DIR, SCALER_FILES, artifact_name,
)
from src.dataset import DatasetError
from src.experiment import (
    DEFAULT_THREADS, ExperimentError, feature_matrix, load_config, output_lock, positive_int,
    prepare_dataset, read_json, tensorflow_runtime, verify_checkpoint,
)
from src.metrics import save_evaluation, save_research_summary
from src.preprocessing import FeatureScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bilstm", "lstm", "both"), default="bilstm")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--threads", type=positive_int, default=DEFAULT_THREADS)
    return parser.parse_args()


def score_checkpoint(tf, args, key: str, raw_features: np.ndarray) -> tuple[np.ndarray, float]:
    model = tf.keras.models.load_model(args.output_dir / MODEL_FILES[key], compile=False)
    if model.input_shape != (None, N_FRAMES, N_MFCC) or model.output_shape != (None, len(EMOTIONS)):
        raise ExperimentError(f"The {MODEL_NAMES[key]} model has incompatible input or output dimensions.")
    scaler = FeatureScaler.load(args.output_dir / SCALER_FILES[key])
    features = tf.convert_to_tensor(scaler.transform(raw_features), dtype=tf.float32)
    batches = [features[i:i + args.batch_size] for i in range(0, len(features), args.batch_size)]

    @tf.function(input_signature=[tf.TensorSpec((None, N_FRAMES, N_MFCC), tf.float32)])
    def forward(batch):
        return model(batch, training=False)

    # Warm up full and partial batches before timing model execution.
    forward(batches[0]).numpy()
    if batches[-1].shape[0] != batches[0].shape[0]:
        forward(batches[-1]).numpy()
    started = perf_counter()
    parts = [forward(batch).numpy() for batch in batches]
    elapsed_ms = (perf_counter() - started) * 1000 / len(features)
    scores = np.concatenate(parts, axis=0)
    if (scores.shape != (len(features), len(EMOTIONS)) or not np.isfinite(scores).all()
            or (scores < 0).any() or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
        raise ExperimentError(f"The {MODEL_NAMES[key]} checkpoint produced invalid probabilities.")
    return scores, elapsed_ms


def run(args: argparse.Namespace) -> None:
    keys = tuple(MODEL_FILES) if args.model == "both" else (args.model,)
    with output_lock(args.output_dir):
        configs = {}
        for key in keys:
            filename = artifact_name(key, "training_config.json")
            if not (args.output_dir / filename).is_file():
                raise ExperimentError(f"Train {MODEL_NAMES[key]} first with python train.py --model {key}.")
            configs[key] = load_config(args.output_dir, key)
            verify_checkpoint(args.output_dir, key, configs[key])
        seed = configs[keys[0]]["seed"]
        frame, stats, signature = prepare_dataset(args.dataset_dir, seed)
        if any(config.get("dataset_split_sha256") != signature for config in configs.values()):
            raise ExperimentError("The audio files or speaker split differ from the saved training run. "
                                  "Use the original dataset and checkpoints.")
        test_rows = frame[frame["split"] == "test"]
        raw_features = feature_matrix(test_rows, "held-out test")
        tf = tensorflow_runtime(args.threads)
        results = {}
        with TemporaryDirectory(prefix=".evaluation-", dir=args.output_dir) as temp:
            stage = Path(temp)
            for key in keys:
                probabilities, inference_ms = score_checkpoint(tf, args, key, raw_features)
                item = save_evaluation(stage, key, configs[key], test_rows, probabilities,
                                       inference_ms, args.batch_size, args.threads)
                results[key] = item
                print(f"{item['model']} held-out test accuracy {item['accuracy']:.2%} · "
                      f"macro F1 {item['f1_macro']:.4f} · {item['n_samples']} clips", flush=True)

            # When a previously evaluated model matches the current protocol, reuse its results.
            for key in MODEL_FILES:
                metrics_path = args.output_dir / artifact_name(key, "test_metrics.json")
                if key in results or not metrics_path.is_file():
                    continue
                previous = read_json(metrics_path)
                other = load_config(args.output_dir, key)
                if (previous.get("dataset_split_sha256") == signature
                        and previous.get("model_sha256") == other.get("model_sha256")
                        and previous.get("scaler_sha256") == other.get("scaler_sha256")
                        and previous.get("inference_batch_size") == args.batch_size
                        and previous.get("inference_threads") == args.threads):
                    verify_checkpoint(args.output_dir, key, other)
                    configs[key], results[key] = other, previous
            ordered = {key: results[key] for key in MODEL_FILES if key in results}
            save_research_summary(stage, ordered, configs, stats)
            for filename in (*COMPARISON_FILES, *(artifact_name(k, "test_metrics.json") for k in keys)):
                (args.output_dir / filename).unlink(missing_ok=True)
            for path in sorted(stage.iterdir(), key=lambda path: path.name.endswith("test_metrics.json")):
                path.replace(args.output_dir / path.name)
    print(f"Evaluation artifacts saved to {args.output_dir.resolve()}.", flush=True)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (DatasetError, ExperimentError, OSError, ValueError) as exc:
        sys.exit(f"Evaluation could not finish: {exc}")
