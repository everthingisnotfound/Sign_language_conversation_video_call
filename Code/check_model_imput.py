from __future__ import annotations

from pathlib import Path

from tensorflow.keras.models import load_model

from word_dataset import LABEL_NAMES_PATH, load_label_names


BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "sign_language_word_model.keras",
    BASE_DIR / "sign_language_word_model.h5",
    BASE_DIR / "sign_language_model.keras",
]


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No trained word model file was found.")


def main() -> None:
    model_path = resolve_model_path()
    model = load_model(model_path)

    print(f"Loaded model: {model_path.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Labels file: {LABEL_NAMES_PATH.name}")
    print(f"Classes ({len(load_label_names())}): {load_label_names()}")
    model.summary()


if __name__ == "__main__":
    main()