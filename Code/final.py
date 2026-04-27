from __future__ import annotations

from collect_landmark import build_landmark_dataset
from load_images import build_train_test_split
from train_sign_model import train_model


def main() -> None:
    print("Step 1/3: extracting landmark sequences from gesture videos...")
    build_landmark_dataset(force=True)

    print("\nStep 2/3: building train and test splits...")
    build_train_test_split()

    print("\nStep 3/3: training the word-level sign model...")
    train_model()

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
