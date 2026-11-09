"""Shared dataset, preprocessing, and artifact settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTPUTS_DIR = ROOT / "outputs"
DATASET_URL = "https://zenodo.org/records/1188976"

EMOTIONS = ("Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised")
SAMPLE_RATE = 22_050
DURATION_SECONDS = 3
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2_048
TRIM_TOP_DB = 40
N_FRAMES = 1 + SAMPLE_RATE * DURATION_SECONDS // HOP_LENGTH
PREPROCESSING = {
    "version": 1,
    "sample_rate": SAMPLE_RATE,
    "duration_seconds": DURATION_SECONDS,
    "n_mfcc": N_MFCC,
    "hop_length": HOP_LENGTH,
    "n_fft": N_FFT,
    "n_frames": N_FRAMES,
    "trim_top_db": TRIM_TOP_DB,
}

MODEL_FILES = {"bilstm": "bilstm_emotion.keras", "lstm": "lstm_baseline.keras"}
SCALER_FILES = {"bilstm": "scaler.npz", "lstm": "baseline_scaler.npz"}
MODEL_NAMES = {"bilstm": "BiLSTM", "lstm": "LSTM"}
EVALUATION_FILES = (
    "test_metrics.json", "test_predictions.csv", "classification_report.txt",
    "classification_report.csv", "confusion_matrix.csv", "confusion_matrix.png",
    "emotion_f1_scores.png",
)
COMPARISON_FILES = ("model_comparison.csv", "model_comparison.png", "research_results.md")


def artifact_name(model: str, name: str) -> str:
    return f"baseline_{name}" if model == "lstm" else name
