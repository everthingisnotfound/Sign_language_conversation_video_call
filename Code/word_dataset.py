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
RAW_LANDMARK_DIM = 63
LANDMARK_DIM = 132

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
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )


def extract_raw_landmarks(hand_landmarks) -> np.ndarray:
    return np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
        dtype=np.float32,
    ).reshape(-1)


def _reshape_landmarks(flat_landmarks: np.ndarray) -> np.ndarray:
    return np.asarray(flat_landmarks, dtype=np.float32).reshape(-1, 21, 3)


def _compute_reference_scale(coords: np.ndarray, valid_mask: np.ndarray) -> float:
    scales: list[float] = []
    for frame, is_valid in zip(coords, valid_mask, strict=False):
        if not is_valid:
            continue

        centered = frame - frame[0]
        spread = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
        if spread > 1e-6:
            scales.append(spread)

    if not scales:
        return 1.0

    scale = float(np.median(scales))
    return scale if scale > 1e-6 else 1.0


def normalize_flat_landmarks(flat_landmarks: np.ndarray) -> np.ndarray:
    coords = _reshape_landmarks(flat_landmarks)[0]
    centered = coords - coords[0]
    scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    if scale < 1e-6:
        scale = float(np.max(np.abs(centered)))
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).reshape(-1)


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    return normalize_flat_landmarks(extract_raw_landmarks(hand_landmarks))


def smooth_landmark_sequence(sequence: np.ndarray, window_size: int = 3) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if len(values) < 3 or window_size <= 1:
        return values

    pad = window_size // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(values)
    for index in range(len(values)):
        smoothed[index] = padded[index : index + window_size].mean(axis=0)
    return smoothed


def transform_landmark_sequence(sequence: np.ndarray) -> np.ndarray:
    raw_sequence = np.asarray(sequence, dtype=np.float32).reshape(-1, RAW_LANDMARK_DIM)
    frame_count = len(raw_sequence)
    if frame_count == 0:
        return np.zeros((0, LANDMARK_DIM), dtype=np.float32)

    coords = raw_sequence.reshape(frame_count, 21, 3)
    valid_mask = np.any(np.abs(coords) > 1e-6, axis=(1, 2))
    if not np.any(valid_mask):
        return np.zeros((frame_count, LANDMARK_DIM), dtype=np.float32)

    reference_scale = _compute_reference_scale(coords, valid_mask)
    first_valid_index = int(np.argmax(valid_mask))
    reference_wrist = coords[first_valid_index, 0].copy()

    features = np.zeros((frame_count, LANDMARK_DIM), dtype=np.float32)
    previous_shape: np.ndarray | None = None
    previous_motion: np.ndarray | None = None

    for index, frame in enumerate(coords):
        if not valid_mask[index]:
            if previous_shape is None or previous_motion is None:
                continue

            features[index, :RAW_LANDMARK_DIM] = previous_shape
            features[index, RAW_LANDMARK_DIM : RAW_LANDMARK_DIM + 3] = previous_motion
            continue

        wrist = frame[0].copy()
        centered = (frame - wrist) / reference_scale
        wrist_motion = (wrist - reference_wrist) / reference_scale
        shape = centered.reshape(-1)

        if previous_shape is None:
            shape_delta = np.zeros_like(shape)
            motion_delta = np.zeros_like(wrist_motion)
        else:
            shape_delta = shape - previous_shape
            motion_delta = wrist_motion - previous_motion

        feature_vector = np.concatenate(
            [shape, wrist_motion, shape_delta, motion_delta],
            axis=0,
        )
        features[index] = feature_vector.astype(np.float32)
        previous_shape = shape
        previous_motion = wrist_motion

    return features


def sample_frame_indices(frame_count: int, sequence_length: int = SEQUENCE_LENGTH) -> list[int]:
    if frame_count <= 1:
        return [0] * sequence_length

    indices = np.linspace(0, frame_count - 1, num=sequence_length)
    return [int(round(index)) for index in indices]


def _fill_missing_landmarks(raw_landmarks: list[np.ndarray | None]) -> np.ndarray | None:
    first_valid = next((entry for entry in raw_landmarks if entry is not None), None)
    if first_valid is None:
        return None

    filled: list[np.ndarray] = []
    previous = np.asarray(first_valid, dtype=np.float32)
    seen_valid = False

    for entry in raw_landmarks:
        if entry is None:
            filled.append(previous.copy())
            continue

        previous = np.asarray(entry, dtype=np.float32)
        seen_valid = True
        filled.append(previous.copy())

    if not seen_valid:
        return None

    return smooth_landmark_sequence(np.array(filled, dtype=np.float32))


def extract_landmark_sequence(
    video_path: Path,
    hands=None,
    sequence_length: int = SEQUENCE_LENGTH,
    min_detection_ratio: float = 0.18,
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

    sequence_landmarks: list[np.ndarray | None] = []
    detections = 0

    for frame_index in frame_indices:
        if frame_count > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = capture.read()
        if not ok:
            sequence_landmarks.append(None)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            sequence_landmarks.append(extract_raw_landmarks(result.multi_hand_landmarks[0]))
            detections += 1
        else:
            sequence_landmarks.append(None)

    capture.release()
    if owns_detector:
        hands.close()

    detection_ratio = detections / float(sequence_length)
    if detection_ratio < min_detection_ratio:
        return None, detection_ratio

    filled_sequence = _fill_missing_landmarks(sequence_landmarks)
    if filled_sequence is None:
        return None, detection_ratio

    return filled_sequence, detection_ratio


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
