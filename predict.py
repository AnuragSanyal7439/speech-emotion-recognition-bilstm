"""Classify a local WAV file using the trained BiLSTM and its feature scaler."""

import argparse
import json
import sys
from pathlib import Path

from config import EMOTIONS, OUTPUTS_DIR
from src.inference import InferenceError, load_predictor
from src.preprocessing import AudioError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="Path to a WAV recording")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args()
    if not args.wav.is_file():
        parser.error(f"WAV file not found: {args.wav}")
    result = load_predictor(args.output_dir).predict(args.wav)
    print(json.dumps({
        "predicted_emotion": result.emotion,
        "softmax_probabilities": {
            label: round(float(score), 6) for label, score in zip(EMOTIONS, result.probabilities)
        },
        "note": "Scores are uncalibrated. The model was trained on acted English speech.",
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (AudioError, InferenceError) as exc:
        sys.exit(f"Prediction could not finish: {exc}")
