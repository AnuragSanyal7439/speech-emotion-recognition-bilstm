"""Local Streamlit interface for the RAVDESS speech emotion experiment."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from config import (
    DATASET_URL,
    DURATION_SECONDS,
    EMOTIONS,
    MODEL_FILES,
    N_FRAMES,
    N_MFCC,
    OUTPUTS_DIR,
    SAMPLE_RATE,
    SCALER_FILES,
    artifact_name,
)

if TYPE_CHECKING:
    from src.inference import Prediction, Predictor

TITLE = "Speech Emotion Recognition Using BiLSTM"
MODEL_NAMES = {"bilstm": "Bidirectional LSTM", "lstm": "Standard LSTM"}
TRAINING_COMMANDS = """python train.py --model bilstm
python train.py --model lstm
python evaluate.py --model both"""
GENERALIZATION = (
    "The models learn from acted English speech in RAVDESS. Predictions on natural speech, "
    "other languages, or noisy microphone recordings may be unreliable."
)

FIGURES = {
    "training_accuracy.png": "BiLSTM training and validation accuracy",
    "training_loss.png": "BiLSTM training and validation loss",
    "confusion_matrix.png": "BiLSTM held-out test confusion matrix",
    "emotion_f1_scores.png": "BiLSTM held-out F1 by emotion",
    "baseline_training_accuracy.png": "Standard LSTM training and validation accuracy",
    "baseline_training_loss.png": "Standard LSTM training and validation loss",
    "baseline_confusion_matrix.png": "Standard LSTM held-out test confusion matrix",
    "baseline_emotion_f1_scores.png": "Standard LSTM held-out F1 by emotion",
    "model_comparison.png": "Held-out model comparison",
    "emotion_distribution.png": "Dataset emotion distribution",
    "audio_duration_distribution.png": "Dataset recording duration distribution",
    "waveform.png": "Example speech waveform",
    "mfcc_heatmap.png": "Example MFCC heatmap",
    "model_architecture.png": "BiLSTM architecture",
    "baseline_model_architecture.png": "Standard LSTM architecture",
}
CSV_EXPORTS = {
    "model_comparison.csv": "Model comparison metrics",
    "training_history.csv": "BiLSTM training history",
    "test_predictions.csv": "BiLSTM test predictions",
    "classification_report.csv": "BiLSTM test classification report",
    "confusion_matrix.csv": "BiLSTM confusion matrix counts",
    "baseline_training_history.csv": "Standard LSTM training history",
    "baseline_test_predictions.csv": "Standard LSTM test predictions",
    "baseline_classification_report.csv": "Standard LSTM test classification report",
    "baseline_confusion_matrix.csv": "Standard LSTM confusion matrix counts",
    "dataset_manifest.csv": "Dataset manifest and actor split",
}

CSS = """
<style>
.block-container { max-width: 1160px; padding-top: 4.25rem; padding-bottom: 4rem; }
.research-kicker {
    color: #1e5ab2; font-size: .75rem; font-weight: 750; letter-spacing: .14em;
    margin-bottom: .45rem; text-transform: uppercase;
}
.research-hero {
    padding: clamp(1.4rem, 4vw, 3rem);
    margin: 0 0 1.35rem;
    border: 1px solid #d6e4fa;
    border-radius: 1.15rem;
    background: linear-gradient(118deg, #f0f6ff 0%, #f8fbff 75%, #eef7f6 100%);
}
.research-hero h1 { color: #102549; font-size: clamp(1.85rem, 5vw, 3.1rem); line-height: 1.08; padding: 0; margin: .45rem 0 .8rem; }
.research-hero p { color: #31466a; font-size: 1.15rem; margin: 0; }
.emotion-list { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1.2rem; }
.emotion-list span {
    background: #f0f6ff; color: #163d75; padding: .32rem .78rem;
    border: 1px solid #d6e4fa; border-radius: 99px; font-size: .9rem; font-weight: 600;
}
</style>
"""


def read_artifact(filename: str) -> bytes | None:
    try:
        return (OUTPUTS_DIR / filename).read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        st.warning(f"Could not read {filename}. {exc}")
        return None


def read_csv(filename: str, required: tuple[str, ...] = ()) -> pd.DataFrame | None:
    data = read_artifact(filename)
    if data is None:
        return None
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except (ValueError, UnicodeError):
        st.warning(f"{filename} is unreadable. Regenerate it from the training or evaluation script.")
        return None
    if not set(required).issubset(frame.columns) or frame.empty:
        st.warning(f"{filename} has no usable data. Regenerate it from the training or evaluation script.")
        return None
    return frame


def read_json(filename: str) -> dict | None:
    data = read_artifact(filename)
    if data is None:
        return None
    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, UnicodeError):
        pass
    st.warning(f"{filename} is unreadable. Regenerate it from the training or evaluation script.")
    return None


def page_header(section: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="research-kicker">{section}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def setup_help(message: str) -> None:
    st.info(message)
    with st.expander("Run the experiment locally"):
        st.markdown(
            f"Download `Audio_Speech_Actors_01-24.zip` from the [RAVDESS record]({DATASET_URL}) "
            "and extract it into `dataset/`. Run these commands in the project folder after "
            "installing the dependencies in `requirements.txt`."
        )
        st.code(TRAINING_COMMANDS, language="powershell")


def show_figure(filename: str) -> None:
    image = read_artifact(filename)
    if image is not None:
        st.image(image, caption=FIGURES[filename])


def home_page() -> None:
    st.markdown(
        f'<div class="research-hero"><div class="research-kicker">Speech intelligence · Research prototype</div>'
        f"<h1>{TITLE}</h1><p>Deep Learning-Based Emotional Speech Analysis</p></div>",
        unsafe_allow_html=True,
    )
    for column, (number, label) in zip(
        st.columns(3),
        (("8", "Emotion classes"), (str(N_MFCC), "MFCCs per frame"), ("2", "Recurrent models")),
    ):
        column.metric(label, number)
    st.subheader("Research objective")
    st.write(
        "Classify the emotion expressed in a speech clip and measure how a bidirectional recurrent network "
        "compares with a standard LSTM on speakers held out from training."
    )
    with st.container(border=True):
        st.subheader("From speech to emotion")
        st.markdown(
            "**Speech Input** → **Audio Preprocessing** → **MFCC Features** → "
            "**BiLSTM** → **Emotion Classification**"
        )
        st.write(
            f"Audio is mixed to mono, resampled to {SAMPLE_RATE:,} Hz, and standardized to "
            f"{DURATION_SECONDS:g} seconds after trimming low-energy edges. Each clip becomes "
            f"{N_FRAMES} time frames of {N_MFCC} MFCCs. The normalizer learns from training actors only."
        )
    st.subheader("Supported emotions")
    chips = "".join(f"<span>{emotion}</span>" for emotion in EMOTIONS)
    st.markdown(f'<div class="emotion-list">{chips}</div>', unsafe_allow_html=True)
    st.caption(GENERALIZATION)


def artifact_fingerprint() -> tuple[tuple[int, int, int, int], ...]:
    versions = []
    for filename in (MODEL_FILES["bilstm"], SCALER_FILES["bilstm"], "training_config.json"):
        info = (OUTPUTS_DIR / filename).stat()
        versions.append((info.st_ino, info.st_ctime_ns, info.st_mtime_ns, info.st_size))
    return tuple(versions)


@st.cache_resource(max_entries=2, show_spinner=False)
def get_predictor(
    output_dir: Path, artifact_version: tuple[tuple[int, int, int, int], ...]
) -> Predictor:
    """The file version is part of the cache key so retraining invalidates the model."""
    from src.inference import load_predictor

    return load_predictor(output_dir)


def show_detection(result: Prediction) -> None:
    from src.visualization import mfcc_figure, waveform_figure
    from matplotlib import pyplot as plt

    st.subheader("Analysis result")
    predicted, score = st.columns(2)
    predicted.metric("Predicted emotion", result.emotion)
    score.metric("Top softmax score", f"{max(result.probabilities):.1%}")
    st.caption(
        "Softmax scores are uncalibrated. A high score can occur even for an out-of-distribution recording."
    )
    probabilities = pd.DataFrame(
        {"Emotion": EMOTIONS, "Softmax score (%)": result.probabilities * 100}
    )
    st.bar_chart(probabilities, x="Emotion", y="Softmax score (%)")
    waveform_column, mfcc_column = st.columns(2)
    charts = (
        (waveform_column, "Original waveform", waveform_figure(result.audio, result.sample_rate)),
        (mfcc_column, "MFCC heatmap", mfcc_figure(result.mfcc)),
    )
    try:
        for column, title, figure in charts:
            column.markdown(f"**{title}**")
            column.pyplot(figure, clear_figure=True)
        mfcc_column.caption(f"Time is measured on the {DURATION_SECONDS:g}-second model input.")
    finally:
        for _, _, figure in charts:
            plt.close(figure)


def detection_page() -> None:
    page_header("01 / Inference", "Emotion Detection", "Analyze a WAV clip with the trained BiLSTM.")
    source = st.radio("Audio source", ("Upload WAV", "Record microphone"), horizontal=True)
    if source == "Upload WAV":
        audio = st.file_uploader("Choose a WAV recording", type=["wav"])
    else:
        st.caption("Allow microphone access in your browser. Open the app on localhost for local recording.")
        audio = st.audio_input("Record speech")
    st.caption(GENERALIZATION)

    try:
        fingerprint = artifact_fingerprint()
    except FileNotFoundError:
        fingerprint = None
        setup_help("The trained BiLSTM and its preprocessing artifacts are required for analysis.")
    except OSError as exc:
        fingerprint = None
        st.error(f"Could not access the model artifacts. {exc}")

    if audio is None:
        st.session_state.pop("detection_result", None)
        st.button("Analyze Emotion", type="primary", disabled=True)
        return

    audio_bytes = audio.getvalue()
    if source == "Upload WAV":
        st.audio(audio_bytes, format="audio/wav")
    key = (source, hashlib.sha256(audio_bytes).hexdigest(), fingerprint)
    cached_key, result = st.session_state.get("detection_result", (None, None))
    if cached_key != key:
        st.session_state.pop("detection_result", None)
        result = None

    if st.button("Analyze Emotion", type="primary", disabled=fingerprint is None):
        from src.inference import InferenceError
        from src.preprocessing import AudioError

        result = None
        st.session_state.pop("detection_result", None)
        try:
            with st.spinner("Extracting MFCC features and running the BiLSTM…"):
                predictor = get_predictor(OUTPUTS_DIR, fingerprint)
                result = predictor.predict(io.BytesIO(audio_bytes))
        except (AudioError, InferenceError) as exc:
            st.error(str(exc))
        else:
            st.session_state["detection_result"] = (key, result)

    if result is not None:
        show_detection(result)


def model_choice(label: str, artifact: str) -> str:
    variants = [
        model for model in MODEL_NAMES if (OUTPUTS_DIR / artifact_name(model, artifact)).is_file()
    ]
    return st.radio(label, variants, format_func=MODEL_NAMES.get, horizontal=True) if variants else "bilstm"


def show_dataset() -> None:
    st.subheader("Dataset and speaker split")
    manifest = read_csv("dataset_manifest.csv", ("split", "actor"))
    if manifest is None:
        st.caption("Dataset statistics appear after a training run inspects the extracted speech corpus.")
        return
    split_counts = (
        manifest.groupby("split")
        .agg(Actors=("actor", "nunique"), Clips=("actor", "size"))
        .reindex(["train", "validation", "test"])
        .rename(index=str.title)
    )
    st.dataframe(split_counts)
    distribution, duration = st.columns(2)
    with distribution:
        show_figure("emotion_distribution.png")
    with duration:
        show_figure("audio_duration_distribution.png")
    stats = read_json("dataset_statistics.json")
    if stats is not None:
        with st.expander("Dataset statistics"):
            st.json(stats, expanded=False)


def training_page() -> None:
    page_header(
        "02 / Training",
        "Training Dashboard",
        "Inspect learning curves, the actor split, and the saved run configuration.",
    )
    model = model_choice("Training run", "training_history.csv")
    history = read_csv(
        artifact_name(model, "training_history.csv"),
        ("epoch", "accuracy", "val_accuracy", "loss", "val_loss"),
    )
    if history is None:
        setup_help("Training history will appear when the first model finishes training.")
        show_dataset()
        return
    settings = read_json(artifact_name(model, "training_config.json"))
    st.subheader(f"{MODEL_NAMES[model]} learning curves")
    epochs, best, validation = st.columns(3)
    epochs.metric("Completed epochs", len(history))
    if settings is not None and settings.get("best_epoch") is not None:
        best.metric("Checkpoint epoch", settings["best_epoch"])
        accuracy_at_checkpoint = settings.get("best_epoch_val_accuracy")
        if isinstance(accuracy_at_checkpoint, (int, float)):
            validation.metric("Validation accuracy at checkpoint", f"{accuracy_at_checkpoint:.1%}")
    accuracy, loss = st.columns(2)
    with accuracy:
        st.markdown("**Accuracy by epoch**")
        st.line_chart(
            history.rename(columns={"accuracy": "Training", "val_accuracy": "Validation"}),
            x="epoch",
            y=["Training", "Validation"],
        )
    with loss:
        st.markdown("**Loss by epoch**")
        st.line_chart(
            history.rename(columns={"loss": "Training", "val_loss": "Validation"}),
            x="epoch",
            y=["Training", "Validation"],
        )

    show_dataset()
    st.subheader("Architecture and configuration")
    show_figure(artifact_name(model, "model_architecture.png"))
    summary = read_artifact(artifact_name(model, "model_summary.txt"))
    if summary is not None:
        with st.expander("Keras model summary"):
            st.code(summary.decode("utf-8", errors="replace"), language=None)
    if settings is not None:
        with st.expander("Saved training configuration"):
            st.json(settings, expanded=False)


def evaluation_page() -> None:
    page_header(
        "03 / Held-out evaluation",
        "Model Evaluation",
        "Scores come from test actors excluded from both training and validation.",
    )
    model = model_choice("Evaluated model", "test_metrics.json")
    metrics = read_json(artifact_name(model, "test_metrics.json"))
    if metrics is None:
        setup_help("Test metrics are available after evaluation finishes.")
        return
    st.subheader(f"{MODEL_NAMES[model]} test metrics")
    measures = (
        ("accuracy", "Test accuracy"),
        ("precision_macro", "Macro precision"),
        ("recall_macro", "Macro recall"),
        ("f1_macro", "Macro F1"),
        ("f1_weighted", "Weighted F1"),
    )
    for column, (key, label) in zip(st.columns(len(measures)), measures):
        value = metrics.get(key)
        column.metric(label, f"{value:.1%}" if isinstance(value, (int, float)) else "—")
    if "n_samples" in metrics:
        st.caption(f"Held-out speech clips evaluated · {metrics['n_samples']}")

    matrix, scores = st.columns(2)
    with matrix:
        show_figure(artifact_name(model, "confusion_matrix.png"))
    with scores:
        show_figure(artifact_name(model, "emotion_f1_scores.png"))
    report = read_csv(artifact_name(model, "classification_report.csv"))
    if report is not None:
        st.subheader("Classification report")
        st.dataframe(
            report,
            hide_index=True,
            column_config={
                "emotion": "Emotion",
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "f1-score": st.column_config.NumberColumn("F1 score", format="%.3f"),
                "support": st.column_config.NumberColumn("Test clips", format="%d"),
            },
        )

    st.subheader("BiLSTM and standard LSTM comparison")
    comparison = read_csv("model_comparison.csv")
    if comparison is None:
        st.caption("Evaluate both trained models on the same actor split to populate the comparison.")
    else:
        columns = {
            "model": "Model",
            "accuracy": st.column_config.NumberColumn("Test accuracy", format="percent"),
            "f1_macro": st.column_config.NumberColumn("Macro F1", format="%.3f"),
            "parameters": st.column_config.NumberColumn("Parameters", format="%d"),
            "training_seconds": st.column_config.NumberColumn("Training (s)", format="%.1f"),
            "inference_ms_per_clip": st.column_config.NumberColumn("Forward pass (ms/clip)", format="%.3f"),
        }
        st.dataframe(comparison, hide_index=True, column_order=list(columns), column_config=columns)
        show_figure("model_comparison.png")
        st.caption("Forward-pass timing uses warmed batches and excludes model loading and audio preprocessing.")


def research_page() -> None:
    page_header("04 / Research exports", "Research Results", "Download figures and tables for the research poster.")
    summary = read_artifact("research_results.md")
    if summary is None:
        st.subheader(TITLE)
        st.write(
            "This study asks how bidirectional recurrence affects emotion recognition on held-out speakers "
            "when both recurrent models receive the same MFCC representation."
        )
        st.markdown(
            f"**Dataset.** The speech-only RAVDESS corpus contains acted English utterances from 24 actors. "
            f"[RAVDESS dataset record]({DATASET_URL})"
        )
        st.markdown(
            f"**Preprocessing.** Mono audio at {SAMPLE_RATE:,} Hz is trimmed, padded or truncated to "
            f"{DURATION_SECONDS:g} seconds, and represented by {N_MFCC} MFCCs per frame. "
            "Feature standardization is fitted only on the training split."
        )
        st.markdown(
            "**Models and evaluation.** The BiLSTM uses two bidirectional LSTM layers and an eight-class "
            "softmax head. The baseline uses unidirectional LSTMs at the same nominal hidden widths. "
            "The held-out comparison reports accuracy, macro F1, parameter count, training time, and inference time."
        )
        with st.expander("Limitations and future work"):
            st.write(GENERALIZATION)
            st.write(
                "Future experiments can test on spontaneous speech from additional speakers and languages, "
                "evaluate performance under background noise, and measure probability calibration."
            )
        setup_help("Experimental observations will appear after evaluation writes the research summary.")
    else:
        st.download_button(
            "Download results summary (.md)", summary, "research_results.md", "text/markdown", on_click="ignore"
        )
        st.markdown(summary.decode("utf-8", errors="replace"))

    figures = [name for name in FIGURES if (OUTPUTS_DIR / name).is_file()]
    tables = [name for name in CSV_EXPORTS if (OUTPUTS_DIR / name).is_file()]
    if figures:
        st.subheader("Poster figures")
        filename = st.selectbox("Figure to preview", figures, format_func=FIGURES.__getitem__)
        image = read_artifact(filename)
        if image is not None:
            st.image(image, caption=FIGURES[filename])
            st.download_button(f"Download {filename}", image, filename, "image/png", on_click="ignore")
    if tables:
        st.subheader("Data tables")
        filename = st.selectbox("Table to export", tables, format_func=CSV_EXPORTS.__getitem__)
        data = read_artifact(filename)
        if data is not None:
            st.download_button(f"Download {filename}", data, filename, "text/csv", on_click="ignore")


def main() -> None:
    st.set_page_config(page_title=TITLE, page_icon="🎙️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    pages = {
        "Home": home_page,
        "Emotion Detection": detection_page,
        "Training Dashboard": training_page,
        "Model Evaluation": evaluation_page,
        "Research Results": research_page,
    }
    with st.sidebar:
        st.markdown("### Speech Emotion Lab")
        st.caption("MFCC · BiLSTM · RAVDESS")
        current_page = st.radio("Project pages", tuple(pages))
        st.divider()
        st.caption("Local research application")
    pages[current_page]()


if __name__ == "__main__":
    main()
