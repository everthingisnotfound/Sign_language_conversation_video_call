from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RAW_GESTURE_DIR = BASE_DIR / "gestures"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SEQUENCE_LENGTH = 30
LANDMARK_DIM = 63

ALL_SEQUENCES_PATH = BASE_DIR / "all_sequences.pkl"
ALL_LABELS_PATH = BASE_DIR / "all_labels.pkl"
TRAIN_SEQUENCES_PATH = BASE_DIR / "train_sequences.pkl"
TEST_SEQUENCES_PATH = BASE_DIR / "test_sequences.pkl"
TRAIN_LABELS_PATH = BASE_DIR / "train_labels.pkl"
TEST_LABELS_PATH = BASE_DIR / "test_labels.pkl"
LABEL_NAMES_PATH = BASE_DIR / "label_names.json"
DATASET_METADATA_PATH = BASE_DIR / "dataset_metadata.json"


@dataclass(frozen=True)
class VideoSample:
    label: str
    path: Path


def _has_video_files(directory: Path) -> bool:
    return any(
        item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        for item in directory.iterdir()
    )


def resolve_gesture_root(root: Path | None = None) -> Path:
    root = root or RAW_GESTURE_DIR
    if not root.exists():
        raise FileNotFoundError(f"Gesture directory '{root}' not found.")

    direct_children = [item for item in root.iterdir() if item.is_dir()]
    if any(_has_video_files(child) for child in direct_children):
        return root

    for child in direct_children:
        nested_children = [item for item in child.iterdir() if item.is_dir()]
        if any(_has_video_files(nested) for nested in nested_children):
            return child

    raise FileNotFoundError(
        "No gesture folders containing video files were found inside the gestures directory."
    )


def list_word_directories(root: Path | None = None) -> list[Path]:
    gesture_root = resolve_gesture_root(root)
    return sorted(
        [
            directory
            for directory in gesture_root.iterdir()
            if directory.is_dir() and _has_video_files(directory)
        ],
        key=lambda path: path.name.lower(),
    )


def list_video_samples(root: Path | None = None) -> list[VideoSample]:
    samples: list[VideoSample] = []
    for directory in list_word_directories(root):
        for video_path in sorted(directory.iterdir(), key=lambda path: path.name.lower()):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append(VideoSample(label=directory.name, path=video_path))
    return samples


def load_label_names() -> list[str]:
    if LABEL_NAMES_PATH.exists():
        return json.loads(LABEL_NAMES_PATH.read_text(encoding="utf-8"))
    return [directory.name for directory in list_word_directories()]


def save_pickle(data, path: Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_json(data, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_hands_detector():
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    coords = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
        dtype=np.float32,
    )
    wrist = coords[0].copy()
    coords -= wrist

    spread = np.max(np.linalg.norm(coords[:, :2], axis=1))
    if spread < 1e-6:
        spread = float(np.max(np.abs(coords)))
    if spread < 1e-6:
        spread = 1.0

    coords /= spread
    return coords.reshape(-1)


def sample_frame_indices(frame_count: int, sequence_length: int = SEQUENCE_LENGTH) -> list[int]:
    if frame_count <= 1:
        return [0] * sequence_length

    indices = np.linspace(0, frame_count - 1, num=sequence_length)
    return [int(round(index)) for index in indices]


def extract_landmark_sequence(
    video_path: Path,
    hands=None,
    sequence_length: int = SEQUENCE_LENGTH,
    min_detection_ratio: float = 0.25,
):
    owns_detector = hands is None
    if hands is None:
        hands = create_hands_detector()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        if owns_detector:
            hands.close()
        return None, 0.0

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(frame_count, sequence_length)

    sequence = []
    detections = 0
    previous_landmarks = np.zeros(LANDMARK_DIM, dtype=np.float32)

    for frame_index in frame_indices:
        if frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = capture.read()
        if not ok:
            sequence.append(previous_landmarks.copy())
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            previous_landmarks = normalize_landmarks(result.multi_hand_landmarks[0])
            detections += 1

        sequence.append(previous_landmarks.copy())

    capture.release()
    if owns_detector:
        hands.close()

    detection_ratio = detections / float(sequence_length)
    if detection_ratio < min_detection_ratio:
        return None, detection_ratio

    return np.array(sequence, dtype=np.float32), detection_ratio


def read_representative_frame(video_path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_index = max(frame_count // 2, 0)
    if frame_count > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)

    ok, frame = capture.read()
    capture.release()
    return frame if ok else None
