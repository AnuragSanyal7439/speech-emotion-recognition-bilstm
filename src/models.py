"""Keras architectures used in the speech emotion experiment."""

from tensorflow import keras

from config import EMOTIONS, N_FRAMES, N_MFCC


def build_model(variant: str, learning_rate: float = 0.001) -> keras.Model:
    """Build a compiled recurrent classifier with the specified directionality."""
    if variant not in {"bilstm", "lstm"}:
        raise ValueError(f"Unknown model variant {variant!r}. Choose 'bilstm' or 'lstm'.")

    first_recurrent: keras.layers.Layer = keras.layers.LSTM(64, return_sequences=True, name="lstm_64")
    second_recurrent: keras.layers.Layer = keras.layers.LSTM(32, name="lstm_32")
    if variant == "bilstm":
        first_recurrent = keras.layers.Bidirectional(first_recurrent, name="bilstm_64")
        second_recurrent = keras.layers.Bidirectional(second_recurrent, name="bilstm_32")

    model = keras.Sequential(
        [
            keras.Input(shape=(N_FRAMES, N_MFCC), name="mfcc_sequence"),
            first_recurrent,
            keras.layers.Dropout(0.3, name="dropout_1"),
            second_recurrent,
            keras.layers.Dropout(0.3, name="dropout_2"),
            keras.layers.Dense(32, activation="relu", name="projection"),
            keras.layers.Dense(len(EMOTIONS), activation="softmax", name="emotion_scores"),
        ],
        name=f"{variant}_emotion",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model
