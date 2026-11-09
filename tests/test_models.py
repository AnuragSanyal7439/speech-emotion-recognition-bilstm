"""Software tests for Keras architectures and local WAV inference."""

import hashlib
import io
import json

import numpy as np
import pytest
import soundfile as sf
from tensorflow import keras

from config import EMOTIONS, MODEL_FILES, N_FRAMES, N_MFCC, PREPROCESSING, SAMPLE_RATE, SCALER_FILES
from src.inference import InferenceError, load_predictor
from src.models import build_model
from src.preprocessing import FeatureScaler, extract_features


@pytest.mark.parametrize(
    ("variant", "recurrent_type", "parameter_count"),
    [("bilstm", keras.layers.Bidirectional, 97_320), ("lstm", keras.layers.LSTM, 40_616)],
)
def test_model_forward_pass(variant, recurrent_type, parameter_count):
    model = build_model(variant)
    assert model.input_shape == (None, N_FRAMES, N_MFCC)
    assert isinstance(model.layers[0], recurrent_type)
    assert isinstance(model.layers[2], recurrent_type)
    assert model.count_params() == parameter_count

    features = np.zeros((2, N_FRAMES, N_MFCC), dtype=np.float32)
    scores = model(features, training=False).numpy()
    assert scores.shape == (2, len(EMOTIONS))
    assert np.all(scores >= 0)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, atol=1e-6)


def test_emotion_order():
    assert EMOTIONS == (
        "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"
    )


def test_predictor_missing_artifacts(tmp_path):
    with pytest.raises(InferenceError, match="Run python train.py --model bilstm"):
        load_predictor(tmp_path)


def test_saved_model_wav_inference(tmp_path):
    rng = np.random.default_rng(42)
    fixture_features = rng.normal(size=(2, N_FRAMES, N_MFCC)).astype(np.float32)
    scaler = FeatureScaler.fit(fixture_features)
    scaler_path = tmp_path / SCALER_FILES["bilstm"]
    scaler.save(scaler_path)

    keras.utils.set_random_seed(42)
    model = build_model("bilstm")
    model_path = tmp_path / MODEL_FILES["bilstm"]
    model.save(model_path)
    manifest = {
        "model": "bilstm",
        "emotion_labels": list(EMOTIONS),
        "preprocessing": PREPROCESSING,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "scaler_sha256": hashlib.sha256(scaler_path.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "training_config.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    samples = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    wav = io.BytesIO()
    sf.write(wav, 0.2 * np.sin(2 * np.pi * 220 * samples), SAMPLE_RATE, format="WAV")
    wav.seek(0)
    audio, sample_rate, mfcc = extract_features(wav)
    scores = model(scaler.transform(mfcc[np.newaxis, ...]), training=False).numpy()[0]

    result = load_predictor(tmp_path).predict(wav)
    assert result.emotion == EMOTIONS[int(scores.argmax())]
    assert result.sample_rate == sample_rate == SAMPLE_RATE
    np.testing.assert_array_equal(result.audio, audio)
    np.testing.assert_array_equal(result.mfcc, mfcc)
    np.testing.assert_allclose(result.probabilities, scores, atol=1e-6)

    manifest_path.write_text(
        json.dumps({**manifest, "emotion_labels": list(reversed(EMOTIONS))}), encoding="utf-8"
    )
    with pytest.raises(InferenceError, match="labels and preprocessing"):
        load_predictor(tmp_path)

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    FeatureScaler.fit(fixture_features + 1).save(scaler_path)
    with pytest.raises(InferenceError, match="does not match training_config.json"):
        load_predictor(tmp_path)
