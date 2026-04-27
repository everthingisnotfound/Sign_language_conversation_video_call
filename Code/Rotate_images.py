from __future__ import annotations

from pathlib import Path

import cv2

from word_dataset import list_video_samples


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "gestures_rotated"
ROTATION_ANGLES = (-8, 8)


def rotate_frame(frame, angle: float):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def augment_videos() -> None:
    samples = list_video_samples()
    if not samples:
        raise FileNotFoundError("No gesture videos were found to augment.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Saving rotated video copies under {OUTPUT_ROOT}")

    for sample in samples:
        destination_dir = OUTPUT_ROOT / sample.label
        destination_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(sample.path))
        if not capture.isOpened():
            print(f"Skipped unreadable video: {sample.path.name}")
            continue

        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writers = {}
        for angle in ROTATION_ANGLES:
            suffix = f"rot{angle:+d}".replace("+", "p").replace("-", "m")
            output_path = destination_dir / f"{sample.path.stem}_{suffix}.mp4"
            writers[angle] = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            for angle, writer in writers.items():
                writer.write(rotate_frame(frame, angle))

        capture.release()
        for writer in writers.values():
            writer.release()

        print(f"Augmented {sample.label}/{sample.path.name}")

    print("Finished writing rotated video copies.")


def main() -> None:
    augment_videos()


if __name__ == "__main__":
    main()