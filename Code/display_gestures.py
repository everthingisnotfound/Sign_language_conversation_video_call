from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from word_dataset import list_video_samples, read_representative_frame


THUMBNAIL_WIDTH = 220
THUMBNAIL_HEIGHT = 160
COLUMNS = 4
OUTPUT_FILE = Path(__file__).resolve().parent / "gesture_overview.jpg"


def create_gesture_grid() -> np.ndarray:
    samples_by_label = {}
    for sample in list_video_samples():
        samples_by_label.setdefault(sample.label, sample.path)

    labels = sorted(samples_by_label.keys(), key=str.lower)
    if not labels:
        raise FileNotFoundError("No gesture videos were found to preview.")

    rows = math.ceil(len(labels) / COLUMNS)
    canvas = np.zeros(
        (rows * THUMBNAIL_HEIGHT, COLUMNS * THUMBNAIL_WIDTH, 3),
        dtype=np.uint8,
    )

    for index, label in enumerate(labels):
        frame = read_representative_frame(samples_by_label[label])
        if frame is None:
            frame = np.zeros((THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))

        cv2.rectangle(frame, (0, THUMBNAIL_HEIGHT - 28), (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label[:20],
            (8, THUMBNAIL_HEIGHT - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        row = index // COLUMNS
        column = index % COLUMNS
        y1 = row * THUMBNAIL_HEIGHT
        y2 = y1 + THUMBNAIL_HEIGHT
        x1 = column * THUMBNAIL_WIDTH
        x2 = x1 + THUMBNAIL_WIDTH
        canvas[y1:y2, x1:x2] = frame

    return canvas


def main() -> None:
    grid_image = create_gesture_grid()
    cv2.imwrite(str(OUTPUT_FILE), grid_image)
    print(f"Saved gesture preview to {OUTPUT_FILE.name}")
    cv2.imshow("Gesture Overview", grid_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()