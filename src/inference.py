"""Load a completed BiLSTM run and classify an input WAV recording."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

from config import EMOTIONS, MODEL_FILES, N_FRAMES, N_MFCC, PREPROCESSING, SCALER_FILES
from src.preprocessing import FeatureScaler, extract_features

if TYPE_CHECKING:
    from tensorflow import keras


class InferenceError(RuntimeError):
    """The trained model could not be loaded or used for prediction."""


@dataclass(frozen=True)
class Prediction:
    emotion: str
    probabilities: np.ndarray
    audio: np.ndarray
    sample_rate: int
    mfcc: np.ndarray


@dataclass(frozen=True)
class Predictor:
    model: keras.Model
    scaler: FeatureScaler

    def predict(self, source: str | Path | BinaryIO) -> Prediction:
        """Return the winning label, scores in label order, and the audio features."""
        audio, sample_rate, mfcc = extract_features(source)
        try:
            features = self.scaler.transform(mfcc[np.newaxis, ...])
            scores = np.asarray(self.model(features, training=False), dtype=np.float32)
        except Exception as exc:
            raise InferenceError(f"BiLSTM inference failed: {exc}") from exc

        if (
            scores.shape != (1, len(EMOTIONS))
            or not np.isfinite(scores).all()
            or (scores < 0).any()
            or not np.isclose(scores.sum(), 1.0)
        ):
            raise InferenceError("The BiLSTM returned invalid emotion probabilities.")

        probabilities = scores[0]
        emotion = EMOTIONS[int(probabilities.argmax())]
        return Prediction(emotion, probabilities, audio, sample_rate, mfcc)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _check_artifacts(artifacts: dict[str, Path], manifest: dict[str, object]) -> None:
    for digest_key, path in artifacts.items():
        if _file_sha256(path) != manifest.get(digest_key):
            raise InferenceError(
                f"{path.name} does not match training_config.json. "
                "Use the files from the same completed BiLSTM training run."
            )


def load_predictor(output_dir: str | Path) -> Predictor:
    """Load a matching model and training scaler from a completed run directory."""
    output_dir = Path(output_dir)
    model_path = output_dir / MODEL_FILES["bilstm"]
    scaler_path = output_dir / SCALER_FILES["bilstm"]
    manifest_path = output_dir / "training_config.json"
    required_paths = (model_path, scaler_path, manifest_path)
    try:
        missing = [path.name for path in required_paths if not path.is_file()]
        if missing:
            raise InferenceError(
                f"Missing BiLSTM artifacts in {output_dir}: {', '.join(missing)}. "
                "Run python train.py --model bilstm after setting up RAVDESS."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InferenceError(f"Could not read {manifest_path}: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("model") != "bilstm"
        or manifest.get("emotion_labels") != list(EMOTIONS)
        or manifest.get("preprocessing") != PREPROCESSING
    ):
        raise InferenceError(
            "The BiLSTM training metadata does not match this project's labels and preprocessing. "
            "Run python train.py --model bilstm to create compatible artifacts."
        )

    artifacts = {"model_sha256": model_path, "scaler_sha256": scaler_path}
    try:
        _check_artifacts(artifacts, manifest)
        from tensorflow import keras

        scaler = FeatureScaler.load(scaler_path)
        model = keras.models.load_model(model_path, compile=False, safe_mode=True)
        # Recheck the files in case a training process replaced them during deserialization.
        _check_artifacts(artifacts, manifest)
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(f"Could not load BiLSTM artifacts from {output_dir}: {exc}") from exc

    if model.input_shape != (None, N_FRAMES, N_MFCC) or model.output_shape != (None, len(EMOTIONS)):
        raise InferenceError("The saved BiLSTM input or output shape is incompatible with this project.")
    return Predictor(model, scaler)
