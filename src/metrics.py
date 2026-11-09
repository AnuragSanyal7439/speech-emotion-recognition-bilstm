"""Held-out metrics and research reporting from actual predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import DATASET_URL, EMOTIONS, MODEL_NAMES, artifact_name
from src.experiment import write_json
from src.visualization import comparison_figure, confusion_figure, f1_figure, save_figure


def save_evaluation(
    stage: Path, key: str, config: dict, rows: pd.DataFrame, probabilities: np.ndarray,
    inference_ms: float, batch_size: int, threads: int,
) -> dict:
    def path(filename: str) -> Path:
        return stage / artifact_name(key, filename)

    truth = rows["label"].to_numpy()
    predicted = probabilities.argmax(axis=1)
    labels = list(range(len(EMOTIONS)))
    report_settings = {"labels": labels, "target_names": EMOTIONS, "zero_division": 0}
    report = classification_report(truth, predicted, output_dict=True, **report_settings)
    macro = report["macro avg"]
    metrics = {
        "model": MODEL_NAMES[key],
        "accuracy": accuracy_score(truth, predicted),
        "precision_macro": macro["precision"],
        "recall_macro": macro["recall"],
        "f1_macro": macro["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "n_samples": len(rows),
        "actors": sorted(int(actor) for actor in rows["actor"].unique()),
        "parameters": config["parameters"],
        "training_seconds": config["training_seconds"],
        "inference_ms_per_clip": inference_ms,
        "inference_batch_size": batch_size,
        "inference_threads": threads,
        "dataset_split_sha256": config["dataset_split_sha256"],
        "model_sha256": config["model_sha256"],
        "scaler_sha256": config["scaler_sha256"],
    }
    predictions = rows[["filename", "actor", "emotion"]].reset_index(drop=True)
    predictions = predictions.rename(columns={"emotion": "true_emotion"})
    predictions["predicted_emotion"] = np.asarray(EMOTIONS)[predicted]
    for index, emotion in enumerate(EMOTIONS):
        predictions[f"probability_{emotion.lower()}"] = probabilities[:, index]
    predictions.to_csv(path("test_predictions.csv"), index=False)

    report_text = classification_report(truth, predicted, digits=4, **report_settings)
    path("classification_report.txt").write_text(
        f"{MODEL_NAMES[key]} | Held-out RAVDESS speech test set ({len(rows)} clips)\n\n{report_text}",
        encoding="utf-8",
    )
    detailed = pd.DataFrame.from_dict(
        {name: report[name] for name in (*EMOTIONS, "macro avg", "weighted avg")}, orient="index",
    )
    detailed.to_csv(path("classification_report.csv"), index_label="emotion")
    matrix = confusion_matrix(truth, predicted, labels=labels)
    pd.DataFrame(matrix, index=EMOTIONS, columns=EMOTIONS).to_csv(
        path("confusion_matrix.csv"), index_label="true_emotion",
    )
    save_figure(confusion_figure(matrix, MODEL_NAMES[key]), path("confusion_matrix.png"))
    save_figure(f1_figure([report[emotion]["f1-score"] for emotion in EMOTIONS], MODEL_NAMES[key]),
                path("emotion_f1_scores.png"))
    write_json(path("test_metrics.json"), metrics)
    return metrics


def save_research_summary(stage: Path, results: dict, configs: dict, stats: dict) -> None:
    fields = ("model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
              "parameters", "training_seconds", "inference_ms_per_clip", "n_samples",
              "inference_batch_size", "inference_threads")
    comparison = pd.DataFrame([{field: item[field] for field in fields} for item in results.values()])
    if len(results) == 2:
        comparison.to_csv(stage / "model_comparison.csv", index=False)
        save_figure(comparison_figure(comparison), stage / "model_comparison.png")
    lines = [
        "# Speech Emotion Recognition Using BiLSTM",
        "",
        "## Research question",
        "",
        "How do unidirectional and bidirectional LSTMs classify eight acted speech emotions "
        "on speakers held out from training and checkpoint selection?",
        "",
        "## Dataset and method",
        "",
        f"The [RAVDESS audio-only speech corpus]({DATASET_URL}) contains {stats['n_clips']:,} clips "
        f"from {stats['n_actors']} actors speaking two English sentences. Actor assignments are "
        "mutually exclusive across partitions.",
        "",
    ]
    for split in ("train", "validation", "test"):
        item = stats["splits"][split]
        actors = ", ".join(f"{actor:02}" for actor in item["actors"])
        lines.append(f"- {split.capitalize()} · {item['n_clips']} clips from actors {actors}")
    if stats["n_unique_recordings"] < stats["n_clips"]:
        lines += [
            "",
            f"The archive contains {stats['n_unique_recordings']:,} unique WAV byte sequences. "
            "The experiment preserves the published clips and checks that no identical WAVs cross actors.",
        ]
    lines += [
        "",
        "Each WAV was converted to mono, resampled to 22,050 Hz, trimmed at 40 dB relative to "
        "peak frame RMS, and truncated or right-padded to 3 seconds. Each sequence contained "
        "130 frames and 40 MFCCs. Per-coefficient standardization was fitted on training clips.",
        "",
        "The models use recurrent widths of 64 and 32, two 30% dropout layers, a 32-unit "
        "ReLU layer and an eight-class softmax. The BiLSTM uses both temporal directions and "
        "has more parameters. Adam optimized sparse categorical cross-entropy. The saved "
        "checkpoint for each model was selected by minimum validation loss.",
        "",
        "## Held-out results",
        "",
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | "
        "Parameters | Fit time (s) | Forward pass (ms/clip) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.accuracy:.2%} | {row.precision_macro:.3f} | {row.recall_macro:.3f} "
            f"| {row.f1_macro:.3f} | {row.f1_weighted:.3f} | {row.parameters:,} "
            f"| {row.training_seconds:.1f} | {row.inference_ms_per_clip:.3f} |"
        )
    lines += ["", "Fit time includes first-epoch tracing, validation, and checkpointing. Forward pass timing is warmed, "
              "batched TensorFlow inference and excludes audio processing and model loading.", ""]
    for key, row in results.items():
        config = configs[key]
        lines.append(
            f"- {row['model']} ran {config['epochs_run']} of {config['epochs_requested']} requested "
            f"epochs. Epoch {config['best_epoch']} reached minimum validation loss "
            f"({config['best_val_loss']:.4f}) with {config['best_epoch_val_accuracy']:.2%} validation "
            f"accuracy. Evaluation used batches of {row['inference_batch_size']}."
        )
    if len(results) == 2:
        difference = results["bilstm"]["accuracy"] - results["lstm"]["accuracy"]
        f1_difference = results["bilstm"]["f1_macro"] - results["lstm"]["f1_macro"]
        lines += ["", f"On this split, BiLSTM minus LSTM was {100 * difference:+.2f} percentage "
                  f"points in accuracy and {f1_difference:+.3f} in macro F1."]
    lines += [
        "",
        "## Limitations and future work",
        "",
        "The single test split covers four actors and two repeated sentences from controlled "
        "recordings. Natural speech, other languages and noisy microphones may yield less reliable "
        "predictions. Softmax scores are uncalibrated. Directionality and parameter count vary "
        "together in this comparison. Future work can compare parameter-matched models, estimate "
        "variation across actor splits, and evaluate on spontaneous multilingual speech.",
        "",
        "## Dataset citation",
        "",
        "Livingstone SR and Russo FA (2018). The Ryerson Audio-Visual Database of Emotional Speech "
        "and Song (RAVDESS). PLOS ONE 13(5), e0196391. "
        "[DOI](https://doi.org/10.1371/journal.pone.0196391).",
        "",
    ]
    (stage / "research_results.md").write_text("\n".join(lines), encoding="utf-8")
