from io import BytesIO

import numpy as np
import pytest
import soundfile as sf

from config import N_FRAMES, N_MFCC
from src.preprocessing import AudioError, FeatureScaler, extract_features


@pytest.mark.parametrize("duration", [0.6, 4.0])
def test_mfcc_shape_for_short_and_long_stereo_recordings(duration):
    sample_rate = 16_000
    t = np.arange(round(sample_rate * duration)) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * 220 * t)
    samples = np.column_stack([signal, 0.5 * signal])
    stream = BytesIO()
    sf.write(stream, samples, sample_rate, format="WAV")
    stream.seek(7)

    audio, original_rate, features = extract_features(stream)

    assert stream.tell() == 7
    assert audio.shape == signal.shape
    np.testing.assert_allclose(audio, 0.75 * signal, atol=4e-5)
    assert original_rate == sample_rate
    assert features.shape == (N_FRAMES, N_MFCC)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_invalid_wav_has_readable_error():
    with pytest.raises(AudioError, match="readable, uncorrupted WAV"):
        extract_features(BytesIO(b"invalid wav"))


def test_silence_has_readable_error(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16_000), 16_000)
    with pytest.raises(AudioError, match="No audible signal"):
        extract_features(path)


def test_scaler_preserves_training_statistics_for_single_and_batch_inference(tmp_path):
    train = np.random.default_rng(42).normal(size=(3, N_FRAMES, N_MFCC)).astype(np.float32)
    train[..., 0] = 5
    scaler = FeatureScaler.fit(train)
    path = tmp_path / "scaler.npz"
    scaler.save(path)
    loaded = FeatureScaler.load(path)

    train_scaled = loaded.transform(train)
    np.testing.assert_allclose(train_scaled.mean(axis=(0, 1)), 0, atol=1e-6)
    np.testing.assert_allclose(train_scaled[..., 1:].std(axis=(0, 1)), 1, atol=1e-6)
    np.testing.assert_array_equal(train_scaled[..., 0], 0)

    unseen = train[:1] + 3
    np.testing.assert_array_equal(loaded.transform(unseen), scaler.transform(unseen))
    np.testing.assert_array_equal(loaded.transform(unseen[0]), scaler.transform(unseen)[0])
    assert loaded.transform(unseen).mean() > 2


def test_load_reports_corrupt_scaler(tmp_path):
    path = tmp_path / "scaler.npz"
    path.write_bytes(b"invalid scaler")
    with pytest.raises(ValueError, match="Invalid feature scaler file"):
        FeatureScaler.load(path)
