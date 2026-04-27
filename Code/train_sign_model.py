from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from load_images import build_train_test_split
from word_dataset import (
    LABEL_NAMES_PATH,
    TEST_LABELS_PATH,
    TEST_SEQUENCES_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_SEQUENCES_PATH,
    load_label_names,
    load_pickle,
    save_json,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_language_word_model.keras"
LEGACY_MODEL_PATH = BASE_DIR / "sign_language_word_model.h5"
TRAINING_HISTORY_PATH = BASE_DIR / "training_history.json"

BATCH_SIZE = 16
EPOCHS = 25


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = Counter(labels.tolist())
    total = float(len(labels))
    class_count = float(len(counts))
    return {
        label: total / (class_count * count)
        for label, count in counts.items()
    }


def build_model(num_classes: int, sequence_length: int, landmark_dim: int) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input(shape=(sequence_length, landmark_dim)),
            layers.Masking(mask_value=0.0),
            layers.GaussianNoise(0.01),
            layers.Bidirectional(
                layers.GRU(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)
            ),
            layers.Bidirectional(
                layers.GRU(64, dropout=0.25, recurrent_dropout=0.1)
            ),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def train_model(epochs: int = EPOCHS) -> dict[str, float]:
    if not all(
        path.exists()
        for path in (
            TRAIN_SEQUENCES_PATH,
            TEST_SEQUENCES_PATH,
            TRAIN_LABELS_PATH,
            TEST_LABELS_PATH,
        )
    ):
        build_train_test_split()

    train_sequences = np.array(load_pickle(TRAIN_SEQUENCES_PATH), dtype=np.float32)
    test_sequences = np.array(load_pickle(TEST_SEQUENCES_PATH), dtype=np.float32)
    train_labels = np.array(load_pickle(TRAIN_LABELS_PATH), dtype=np.int32)
    test_labels = np.array(load_pickle(TEST_LABELS_PATH), dtype=np.int32)

    sequence_length = int(train_sequences.shape[1])
    landmark_dim = int(train_sequences.shape[2])
    num_classes = int(len(np.unique(train_labels)))

    model = build_model(num_classes, sequence_length, landmark_dim)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(train_labels)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
        .shuffle(len(train_sequences))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_sequences, test_labels))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    loss, accuracy = model.evaluate(test_dataset, verbose=0)
    model.save(MODEL_PATH)
    model.save(LEGACY_MODEL_PATH)

    history_payload = {
        "labels": load_label_names(),
        "epochs_requested": epochs,
        "val_accuracy": [float(value) for value in history.history.get("val_accuracy", [])],
        "val_loss": [float(value) for value in history.history.get("val_loss", [])],
        "train_accuracy": [float(value) for value in history.history.get("accuracy", [])],
        "train_loss": [float(value) for value in history.history.get("loss", [])],
        "final_test_accuracy": float(accuracy),
        "final_test_loss": float(loss),
    }
    save_json(history_payload, TRAINING_HISTORY_PATH)

    print(f"Saved model to {MODEL_PATH.name}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation loss: {loss:.4f}")

    return {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "classes": float(num_classes),
    }


def main() -> None:
    print(f"Labels file: {LABEL_NAMES_PATH.name}")
    train_model()


if __name__ == "__main__":
    main()