"""Matplotlib figures shared by the research exports and the dashboard."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from config import EMOTIONS, HOP_LENGTH, SAMPLE_RATE

BLUE = "#275dd2"
GREEN = "#068876"
INK = "#172741"


def _axes(title: str, xlabel: str, ylabel: str) -> tuple[Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.5, 4.3), layout="constrained")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.title.set_fontweight("bold")
    ax.title.set_color(INK)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    return fig, ax


def save_figure(fig: Figure, path: Path) -> None:
    try:
        fig.savefig(path, dpi=240, facecolor="white")
    finally:
        plt.close(fig)


def waveform_figure(audio: np.ndarray, sample_rate: int) -> Figure:
    fig, ax = _axes("Original speech waveform", "Time (seconds)", "Amplitude")
    if len(audio) > 40_000:
        # Plot the peak envelope so isolated transients survive downsampling for display.
        width = (len(audio) + 19_999) // 20_000
        starts = np.arange(0, len(audio), width)
        ax.fill_between(
            starts / sample_rate, np.minimum.reduceat(audio, starts),
            np.maximum.reduceat(audio, starts), color=BLUE,
        )
    else:
        ax.plot(np.arange(len(audio)) / sample_rate, audio, color=BLUE, linewidth=0.65)
    ax.set_xlim(0, len(audio) / sample_rate)
    return fig


def mfcc_figure(mfcc: np.ndarray) -> Figure:
    fig, ax = _axes("Mel-frequency cepstral coefficients", "Time (seconds)", "MFCC index")
    ax.grid(False)
    duration = (len(mfcc) - 1) * HOP_LENGTH / SAMPLE_RATE
    limit = max(10.0, float(np.abs(mfcc).max()))
    # The first coefficient is typically much larger than the others.
    norm = SymLogNorm(linthresh=10, vmin=-limit, vmax=limit)
    heatmap = ax.imshow(
        mfcc.T, origin="lower", aspect="auto", interpolation="nearest", cmap="RdBu_r", norm=norm,
        extent=(0, duration, -0.5, mfcc.shape[1] - 0.5),
    )
    fig.colorbar(heatmap, ax=ax, label="Coefficient value (symmetric log scale)")
    return fig


def dataset_figures(frame: pd.DataFrame, target: Path) -> None:
    fig, ax = _axes("RAVDESS speech clips by emotion", "Emotion", "Number of clips")
    counts = frame["emotion"].value_counts().reindex(EMOTIONS, fill_value=0)
    ax.bar_label(ax.bar(counts.index, counts.values, color=BLUE), padding=3)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, target / "emotion_distribution.png")

    fig, ax = _axes("Original clip duration distribution", "Duration (seconds)", "Number of clips")
    ax.hist(frame["duration_seconds"], bins=24, color=GREEN, edgecolor="white")
    save_figure(fig, target / "audio_duration_distribution.png")


def training_figure(history: pd.DataFrame, metric: str, model: str) -> Figure:
    fig, ax = _axes(f"{model} training and validation {metric}", "Epoch", metric.capitalize())
    ax.plot(history["epoch"], history[metric], label="Training", color=BLUE, linewidth=2)
    ax.plot(history["epoch"], history[f"val_{metric}"], label="Validation", color=GREEN, linewidth=2)
    ax.legend(frameon=False)
    if metric == "accuracy":
        ax.set_ylim(0, 1)
    return fig


def confusion_figure(matrix: np.ndarray, model: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.5), layout="constrained")
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", ax=ax, square=True,
        xticklabels=EMOTIONS, yticklabels=EMOTIONS, cbar_kws={"label": "Clips"},
    )
    ax.set(title=f"{model} confusion matrix on held-out actors",
           xlabel="Predicted emotion", ylabel="True emotion")
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)
    return fig


def f1_figure(scores: list[float], model: str) -> Figure:
    fig, ax = _axes(f"{model} F1 by emotion on held-out actors", "Emotion", "F1-score")
    ax.bar_label(ax.bar(EMOTIONS, scores, color=GREEN), fmt="%.2f", padding=3)
    ax.set_ylim(0, 1.09)
    ax.tick_params(axis="x", rotation=20)
    return fig


def comparison_figure(comparison: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2), layout="constrained")
    fig.suptitle("Recurrent model comparison on the same held-out actors", fontweight="bold", color=INK)
    scores = axes[0, 0]
    x = np.arange(2)
    colors = (BLUE, GREEN)
    for index, (row, color) in enumerate(zip(comparison.itertuples(), colors)):
        bars = scores.bar(x + (index - 0.5) * 0.32, (row.accuracy, row.f1_macro),
                          width=0.32, label=row.model, color=color)
        scores.bar_label(bars, fmt="%.3f", padding=3)
    scores.set(xticks=x, xticklabels=("Accuracy", "Macro F1"), ylim=(0, 1.13), title="Test performance")
    scores.legend(frameon=False, loc="upper right")

    panels = (
        ("parameters", "Trainable parameters", "{:,.0f}"),
        ("training_seconds", "Training time (seconds)", "{:.1f}"),
        ("inference_ms_per_clip", "Forward pass (ms per clip)", "{:.3f}"),
    )
    for ax, (field, title, fmt) in zip(axes.flat[1:], panels):
        values = comparison[field]
        bars = ax.bar(comparison["model"], values, color=colors, width=0.5)
        ax.bar_label(bars, labels=[fmt.format(value) for value in values], padding=3)
        ax.set(title=title, ylim=(0, values.max() * 1.2))
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    return fig


def architecture_figure(model, name: str) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 7.4), layout="constrained")
    ax.set_title(f"{name} speech emotion classifier", fontweight="bold", pad=18)
    layers = [("MFCC sequence", model.input_shape[1:])]
    for layer in model.layers:
        kind = type(layer).__name__
        if kind == "Bidirectional":
            label = f"Bidirectional LSTM · {layer.forward_layer.units} units per direction"
        elif kind == "LSTM":
            label = f"LSTM · {layer.units} units"
        elif kind == "Dropout":
            label = f"Dropout · {layer.rate:.0%}"
        else:
            label = f"Dense · {layer.units} units · {layer.activation.__name__}"
        layers.append((label, tuple(layer.output.shape[1:])))
    for y, (label, shape) in zip(range(len(layers) - 1, -1, -1), layers):
        dims = " × ".join(map(str, shape))
        ax.text(0.5, y, f"{label}\n{dims}", ha="center", va="center", color=INK,
                bbox={"boxstyle": "round,pad=0.45", "facecolor": "#edf3ff", "edgecolor": BLUE})
        if y:
            ax.annotate("", xy=(0.5, y - 0.69), xytext=(0.5, y - 0.30),
                        arrowprops={"arrowstyle": "->", "color": BLUE})
    ax.set(xlim=(0, 1), ylim=(-0.5, len(layers) - 0.5))
    ax.set_axis_off()
    return fig
