from __future__ import annotations

import csv
from pathlib import Path

from word_dataset import (
    ALL_LABELS_PATH,
    ALL_SEQUENCES_PATH,
    DATASET_METADATA_PATH,
    LABEL_NAMES_PATH,
    SEQUENCE_LENGTH,
    create_hands_detector,
    extract_landmark_sequence,
    list_video_samples,
    save_json,
    save_pickle,
)


OUTPUT_CSV = Path(__file__).resolve().parent / "landmark_data.csv"


def build_landmark_dataset(force: bool = True) -> dict[str, object]:
    if (
        not force
        and ALL_SEQUENCES_PATH.exists()
        and ALL_LABELS_PATH.exists()
        and LABEL_NAMES_PATH.exists()
        and DATASET_METADATA_PATH.exists()
    ):
        print("Using existing landmark dataset files.")
        return {}

    samples = list_video_samples()
    if not samples:
        raise FileNotFoundError("No gesture videos were found in the gesture folders.")

    label_names = sorted({sample.label for sample in samples}, key=str.lower)
    label_to_index = {label: index for index, label in enumerate(label_names)}

    hands = create_hands_detector()
    sequences = []
    labels = []
    manifest_rows = []
    skipped = 0

    print(f"Extracting landmark sequences from {len(samples)} videos...")

    try:
        for position, sample in enumerate(samples, start=1):
            sequence, detection_ratio = extract_landmark_sequence(sample.path, hands=hands)
            status = "ok"
            if sequence is None:
                skipped += 1
                status = "skipped"
            else:
                sequences.append(sequence)
                labels.append(label_to_index[sample.label])

            manifest_rows.append(
                {
                    "label": sample.label,
                    "video": str(sample.path),
                    "detection_ratio": f"{detection_ratio:.3f}",
                    "status": status,
                }
            )
            print(
                f"[{position:03d}/{len(samples)}] {sample.label:<12} "
                f"{sample.path.name:<20} {status} ({detection_ratio:.2f})"
            )
    finally:
        hands.close()

    save_pickle(sequences, ALL_SEQUENCES_PATH)
    save_pickle(labels, ALL_LABELS_PATH)
    save_json(label_names, LABEL_NAMES_PATH)

    metadata = {
        "class_count": len(label_names),
        "video_count": len(samples),
        "usable_sequences": len(sequences),
        "skipped_videos": skipped,
        "sequence_length": SEQUENCE_LENGTH,
        "landmark_dim": 63,
    }
    save_json(metadata, DATASET_METADATA_PATH)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "video", "detection_ratio", "status"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved {len(sequences)} sequences across {len(label_names)} classes.")
    print(f"Skipped {skipped} videos with weak or missing hand detection.")
    print(f"Saved manifest to {OUTPUT_CSV.name}")
    return metadata


def main() -> None:
    build_landmark_dataset(force=True)


if __name__ == "__main__":
    main()