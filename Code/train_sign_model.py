from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from load_images import build_train_test_split
from word_dataset import (
    LABEL_NAMES_PATH,
    LANDMARK_DIM,
    RAW_LANDMARK_DIM,
    TEST_LABELS_PATH,
    TEST_SEQUENCES_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_SEQUENCES_PATH,
    load_label_names,
    load_pickle,
    save_json,
    smooth_landmark_sequence,
    transform_landmark_sequence,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_language_word_model.keras"
LEGACY_MODEL_PATH = BASE_DIR / "sign_language_word_model.h5"
TRAINING_HISTORY_PATH = BASE_DIR / "training_history.json"

BATCH_SIZE = 16
EPOCHS = 60
VALIDATION_SIZE = 0.15
AUGMENTATIONS_PER_SAMPLE = 2
RANDOM_SEED = 42


def compute_class_weights(labels: np.ndarray) -> dict[int, float]:
    counts = Counter(labels.tolist())
    total = float(len(labels))
    class_count = float(len(counts))
    return {
        label: total / (class_count * count)
        for label, count in counts.items()
    }


def resample_sequence(sequence: np.ndarray, positions: np.ndarray) -> np.ndarray:
    clamped = np.clip(np.round(positions), 0, len(sequence) - 1).astype(int)
    return np.asarray(sequence, dtype=np.float32)[clamped]


def augment_raw_sequence(sequence: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    augmented = np.asarray(sequence, dtype=np.float32).reshape(-1, RAW_LANDMARK_DIM)
    if len(augmented) == 0:
        return augmented

    if rng.random() < 0.85:
        base = np.linspace(0.0, 1.0, num=len(augmented))
        stretch = rng.uniform(0.84, 1.16)
        offset = rng.uniform(-0.08, 0.08)
        warped = np.clip(((base - 0.5) * stretch) + 0.5 + offset, 0.0, 1.0)
        augmented = resample_sequence(augmented, warped * (len(augmented) - 1))

    coords = augmented.reshape(-1, 21, 3).copy()
    valid_mask = np.any(np.abs(coords) > 1e-6, axis=(1, 2))
    if not np.any(valid_mask):
        return augmented

    angle = np.deg2rad(rng.uniform(-12.0, 12.0))
    scale = rng.uniform(0.92, 1.08)
    translation = np.array(
        [rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)],
        dtype=np.float32,
    )
    z_scale = rng.uniform(0.95, 1.05)
    jitter_std = rng.uniform(0.0015, 0.0045)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )

    for index, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue

        wrist_xy = coords[index, 0, :2].copy()
        centered_xy = coords[index, :, :2] - wrist_xy
        centered_xy = (centered_xy @ rotation.T) * scale
        coords[index, :, :2] = centered_xy + wrist_xy + translation

        wrist_z = coords[index, 0, 2].copy()
        coords[index, :, 2] = ((coords[index, :, 2] - wrist_z) * z_scale) + wrist_z
        coords[index] += rng.normal(0.0, jitter_std, size=coords[index].shape).astype(np.float32)

    if rng.random() < 0.35 and len(coords) >= 4:
        freeze_length = int(rng.integers(2, min(5, len(coords))))
        start = int(rng.integers(0, len(coords) - freeze_length + 1))
        coords[start : start + freeze_length] = coords[start]

    return smooth_landmark_sequence(coords.reshape(-1, RAW_LANDMARK_DIM))


def prepare_feature_sequences(sequences: np.ndarray) -> np.ndarray:
    transformed = [
        transform_landmark_sequence(sequence)
        for sequence in np.asarray(sequences, dtype=np.float32)
    ]
    return np.asarray(transformed, dtype=np.float32)


def build_augmented_training_set(
    train_sequences: np.ndarray,
    train_labels: np.ndarray,
    augmentations_per_sample: int = AUGMENTATIONS_PER_SAMPLE,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_sequences: list[np.ndarray] = []
    augmented_labels: list[int] = []

    for sequence, label in zip(train_sequences, train_labels, strict=False):
        feature_sequences.append(transform_landmark_sequence(sequence))
        augmented_labels.append(int(label))

        for _ in range(augmentations_per_sample):
            augmented_sequence = augment_raw_sequence(sequence, rng)
            feature_sequences.append(transform_landmark_sequence(augmented_sequence))
            augmented_labels.append(int(label))

    return (
        np.asarray(feature_sequences, dtype=np.float32),
        np.asarray(augmented_labels, dtype=np.int32),
    )


def build_model(num_classes: int, sequence_length: int, landmark_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length, landmark_dim))
    x = layers.LayerNormalization()(inputs)
    x = layers.GaussianNoise(0.02)(x)
    x = layers.Conv1D(128, 5, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Bidirectional(layers.GRU(160, return_sequences=True, dropout=0.25))(x)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.15)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.GRU(96, return_sequences=True, dropout=0.2))(x)

    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dense(256, activation="swish", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(128, activation="swish", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="sign_language_word_model")


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

    train_sequences = np.asarray(load_pickle(TRAIN_SEQUENCES_PATH), dtype=np.float32)
    test_sequences = np.asarray(load_pickle(TEST_SEQUENCES_PATH), dtype=np.float32)
    train_labels = np.asarray(load_pickle(TRAIN_LABELS_PATH), dtype=np.int32)
    test_labels = np.asarray(load_pickle(TEST_LABELS_PATH), dtype=np.int32)

    if len(train_sequences) == 0 or len(test_sequences) == 0:
        raise RuntimeError("Training data was empty. Rebuild the landmark dataset before training.")

    base_train_sequences, val_sequences, base_train_labels, val_labels = train_test_split(
        train_sequences,
        train_labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=train_labels,
    )

    augmented_train_sequences, augmented_train_labels = build_augmented_training_set(
        base_train_sequences,
        base_train_labels,
    )
    val_feature_sequences = prepare_feature_sequences(val_sequences)
    test_feature_sequences = prepare_feature_sequences(test_sequences)

    sequence_length = int(augmented_train_sequences.shape[1])
    landmark_dim = int(augmented_train_sequences.shape[2])
    num_classes = int(len(np.unique(train_labels)))

    model = build_model(num_classes, sequence_length, landmark_dim)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=7e-4,
            weight_decay=1e-4,
            clipnorm=1.0,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

    class_weights = compute_class_weights(base_train_labels)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((augmented_train_sequences, augmented_train_labels))
        .shuffle(len(augmented_train_sequences), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_feature_sequences, val_labels))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((test_feature_sequences, test_labels))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
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
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2,
    )

    loss, accuracy, top3_accuracy = model.evaluate(test_dataset, verbose=0)
    model.save(MODEL_PATH)
    model.save(LEGACY_MODEL_PATH)

    history_payload = {
        "labels": load_label_names(),
        "raw_feature_dim": RAW_LANDMARK_DIM,
        "model_feature_dim": LANDMARK_DIM,
        "epochs_requested": epochs,
        "augmentation_multiplier": AUGMENTATIONS_PER_SAMPLE + 1,
        "train_samples_original": int(len(base_train_sequences)),
        "train_samples_augmented": int(len(augmented_train_sequences)),
        "validation_samples": int(len(val_feature_sequences)),
        "test_samples": int(len(test_feature_sequences)),
        "val_accuracy": [float(value) for value in history.history.get("val_accuracy", [])],
        "val_loss": [float(value) for value in history.history.get("val_loss", [])],
        "val_top3_accuracy": [
            float(value) for value in history.history.get("val_top3_accuracy", [])
        ],
        "train_accuracy": [float(value) for value in history.history.get("accuracy", [])],
        "train_loss": [float(value) for value in history.history.get("loss", [])],
        "train_top3_accuracy": [
            float(value) for value in history.history.get("top3_accuracy", [])
        ],
        "final_test_accuracy": float(accuracy),
        "final_test_top3_accuracy": float(top3_accuracy),
        "final_test_loss": float(loss),
    }
    save_json(history_payload, TRAINING_HISTORY_PATH)

    print(f"Saved model to {MODEL_PATH.name}")
    print(f"Feature dim: {landmark_dim}")
    print(f"Validation split size: {len(val_feature_sequences)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test top-3 accuracy: {top3_accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")

    return {
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_accuracy),
        "loss": float(loss),
        "classes": float(num_classes),
    }


def main() -> None:
    print(f"Labels file: {LABEL_NAMES_PATH.name}")
    train_model()


if __name__ == "__main__":
    main()
