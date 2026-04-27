from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from collect_landmark import build_landmark_dataset
from word_dataset import (
    ALL_LABELS_PATH,
    ALL_SEQUENCES_PATH,
    TEST_LABELS_PATH,
    TEST_SEQUENCES_PATH,
    TRAIN_LABELS_PATH,
    TRAIN_SEQUENCES_PATH,
    load_pickle,
    save_pickle,
)


TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_train_test_split(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, int]:
    if not ALL_SEQUENCES_PATH.exists() or not ALL_LABELS_PATH.exists():
        build_landmark_dataset(force=True)

    sequences = np.array(load_pickle(ALL_SEQUENCES_PATH), dtype=np.float32)
    labels = np.array(load_pickle(ALL_LABELS_PATH), dtype=np.int32)

    if len(sequences) == 0:
        raise RuntimeError("No landmark sequences were available to split.")

    train_sequences, test_sequences, train_labels, test_labels = train_test_split(
        sequences,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    save_pickle(train_sequences, TRAIN_SEQUENCES_PATH)
    save_pickle(test_sequences, TEST_SEQUENCES_PATH)
    save_pickle(train_labels, TRAIN_LABELS_PATH)
    save_pickle(test_labels, TEST_LABELS_PATH)

    summary = {
        "train_samples": int(len(train_sequences)),
        "test_samples": int(len(test_sequences)),
        "class_count": int(len(np.unique(labels))),
    }

    print(
        f"Saved {summary['train_samples']} training samples and "
        f"{summary['test_samples']} test samples."
    )
    return summary


def main() -> None:
    build_train_test_split()


if __name__ == "__main__":
    main()