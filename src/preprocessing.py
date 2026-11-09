"""Shared WAV decoding, fixed-length MFCC extraction, and feature scaling."""

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile

import librosa
import numpy as np
import soundfile as sf
from librosa.util.exceptions import LibrosaError

from config import DURATION_SECONDS, HOP_LENGTH, N_FFT, N_FRAMES, N_MFCC, SAMPLE_RATE, TRIM_TOP_DB


MIN_AUDIO_SECONDS = 0.1
MAX_AUDIO_SECONDS = 60
MAX_DECODED_SAMPLES = 12_000_000
MIN_AUDIO_STD = 1e-4
FEATURE_SHAPE = (N_FRAMES, N_MFCC)


class AudioError(ValueError):
    """The recording could not be decoded or contains no usable audio."""


def _decode_wav(source: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(source) if isinstance(source, Path) else source) as wav:
        if wav.format not in {"WAV", "WAVEX", "RF64"}:
            raise AudioError("Choose a WAV audio recording.")
        duration = wav.frames / wav.samplerate
        if duration < MIN_AUDIO_SECONDS:
            raise AudioError("The recording is too short. Record at least 0.1 seconds of audio.")
        if duration > MAX_AUDIO_SECONDS or wav.frames * wav.channels > MAX_DECODED_SAMPLES:
            raise AudioError("The recording is too large. Choose a WAV of up to 60 seconds with fewer channels.")
        samples = wav.read(dtype="float32", always_2d=True)
        if len(samples) != wav.frames or not np.isfinite(samples).all():
            raise AudioError("The WAV contains incomplete or invalid audio samples.")
        mono = samples.mean(axis=1, dtype=np.float64).astype(np.float32)
        return mono, wav.samplerate


def _read_wav(source: str | Path | BinaryIO) -> tuple[np.ndarray, int]:
    try:
        if isinstance(source, (str, Path)):
            return _decode_wav(source)
        position = source.tell()
        try:
            source.seek(0)
            return _decode_wav(source)
        finally:
            source.seek(position)
    except AudioError:
        raise
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AudioError("Could not open the audio. Choose a readable, uncorrupted WAV file.") from exc


def extract_features(source: str | Path | BinaryIO) -> tuple[np.ndarray, int, np.ndarray]:
    """Return the original mono signal, its sample rate, and fixed-size raw MFCCs."""
    audio, sample_rate = _read_wav(source)
    if audio.std(dtype=np.float64) <= MIN_AUDIO_STD:
        raise AudioError("No audible signal was detected. Record speech closer to the microphone.")

    try:
        resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        trimmed, _ = librosa.effects.trim(
            resampled, top_db=TRIM_TOP_DB, frame_length=N_FFT, hop_length=HOP_LENGTH
        )
        samples = int(DURATION_SECONDS * SAMPLE_RATE)
        fixed = librosa.util.fix_length(trimmed, size=samples)
        mfcc = librosa.feature.mfcc(
            y=fixed,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            center=True,
        )
    except (LibrosaError, ValueError) as exc:
        raise AudioError("Could not process the audio. Try recording or exporting the WAV again.") from exc
    features = np.ascontiguousarray(mfcc.T, dtype=np.float32)
    if features.shape != FEATURE_SHAPE or not np.isfinite(features).all():
        raise AudioError("Could not extract a valid MFCC matrix from the recording.")
    return audio, sample_rate, features


@dataclass(frozen=True)
class FeatureScaler:
    """Per-coefficient standardization over the training clips and their time frames."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.mean.shape != (N_MFCC,)
            or self.scale.shape != (N_MFCC,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError("Feature scaler must contain 40 finite means and positive finite scales.")

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureScaler":
        features = np.asarray(features, dtype=np.float32)
        if (
            features.ndim != 3
            or features.shape[1:] != FEATURE_SHAPE
            or len(features) == 0
            or not np.isfinite(features).all()
        ):
            raise ValueError(f"Expected a nonempty batch of finite MFCCs with shape (N, {N_FRAMES}, {N_MFCC}).")
        mean = features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std = features.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        return cls(mean, np.where(std >= 1e-6, std, np.float32(1)))

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if (
            features.ndim not in (2, 3)
            or features.shape[-2:] != FEATURE_SHAPE
            or not np.isfinite(features).all()
        ):
            raise ValueError(f"Expected finite MFCCs with trailing shape {FEATURE_SHAPE}.")
        return (features - self.mean) / self.scale

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as target:
            np.savez(target, mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        with Path(path).open("rb") as source:
            try:
                with np.load(source, allow_pickle=False) as values:
                    return cls(values["mean"].astype(np.float32), values["scale"].astype(np.float32))
            except (BadZipFile, EOFError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid feature scaler file {path}. Run training to regenerate it.") from exc
