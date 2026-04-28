from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

from word_dataset import (
    LANDMARK_DIM,
    RAW_LANDMARK_DIM,
    SEQUENCE_LENGTH,
    create_hands_detector,
    extract_raw_landmarks,
    load_label_names,
    transform_landmark_sequence,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "sign_language_word_model.keras",
    BASE_DIR / "sign_language_word_model.h5",
    BASE_DIR / "sign_language_model.keras",
    BASE_DIR / "sign_language_model.h5",
]
DEFAULT_MIN_CONFIDENCE = 0.15
DEFAULT_MIN_FRAMES = 8
DEFAULT_TRANSCRIPT_THRESHOLD = 0.15
DEFAULT_STABLE_FRAMES = 3
DEFAULT_COOLDOWN_SECONDS = 0.8


@dataclass
class PredictionResult:
    label: str | None
    confidence: float
    bbox: tuple[int, int, int, int] | None
    top_predictions: list[dict[str, float | str]]
    has_hand: bool
    ready: bool
    frames_buffered: int
    detection_ratio: float


@dataclass
class TranscriptSnapshot:
    transcript: str
    candidate_label: str | None
    committed_label: str | None
    candidate_hits: int
    updated: bool


class TranscriptBuilder:
    def __init__(
        self,
        stable_frames: int = DEFAULT_STABLE_FRAMES,
        confidence_threshold: float = DEFAULT_TRANSCRIPT_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        self.stable_frames = stable_frames
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.transcript = ""
        self.candidate_label: str | None = None
        self.candidate_hits = 0
        self.last_committed_label: str | None = None
        self.last_commit_time = 0.0
        self.has_reset_since_last_commit = True

    def clear(self) -> None:
        self.transcript = ""
        self.candidate_label = None
        self.candidate_hits = 0
        self.last_committed_label = None
        self.last_commit_time = 0.0
        self.has_reset_since_last_commit = True

    def update(self, label: str | None, confidence: float) -> TranscriptSnapshot:
        committed_label = None
        updated = False

        if not label or confidence < self.confidence_threshold:
            self.candidate_label = None
            self.candidate_hits = 0
            self.has_reset_since_last_commit = True
            return TranscriptSnapshot(
                transcript=self.transcript,
                candidate_label=self.candidate_label,
                committed_label=committed_label,
                candidate_hits=self.candidate_hits,
                updated=updated,
            )

        if label == self.candidate_label:
            self.candidate_hits += 1
        else:
            self.candidate_label = label
            self.candidate_hits = 1
            if label != self.last_committed_label:
                self.has_reset_since_last_commit = True

        enough_frames = self.candidate_hits >= self.stable_frames
        cooldown_complete = (
            time.time() - self.last_commit_time
        ) >= self.cooldown_seconds
        is_new_label = label != self.last_committed_label

        can_commit = False
        if enough_frames:
            if is_new_label:
                can_commit = True
            elif cooldown_complete and self.has_reset_since_last_commit:
                can_commit = True

        if can_commit:
            self._append_label(label)
            self.last_committed_label = label
            self.last_commit_time = time.time()
            self.has_reset_since_last_commit = False
            committed_label = label
            updated = True

        return TranscriptSnapshot(
            transcript=self.transcript,
            candidate_label=self.candidate_label,
            committed_label=committed_label,
            candidate_hits=self.candidate_hits,
            updated=updated,
        )

    def _append_label(self, label: str) -> None:
        token = label.strip()
        if not token:
            return

        if self.transcript:
            self.transcript += " "
        self.transcript += token


class LiveRecognitionSession:
    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        min_frames: int = DEFAULT_MIN_FRAMES,
        transcript_builder: TranscriptBuilder | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.min_frames = min_frames
        self.transcript_builder = transcript_builder or TranscriptBuilder()
        self.sequence_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
        self.hand_history: deque[int] = deque(maxlen=sequence_length)
        self.last_bbox: tuple[int, int, int, int] | None = None

    def clear(self) -> None:
        self.sequence_buffer.clear()
        self.hand_history.clear()
        self.last_bbox = None
        self.transcript_builder.clear()

    @property
    def frames_buffered(self) -> int:
        return len(self.sequence_buffer)

    @property
    def detection_ratio(self) -> float:
        if not self.hand_history:
            return 0.0
        return float(sum(self.hand_history) / len(self.hand_history))

    @property
    def ready(self) -> bool:
        return self.frames_buffered >= self.min_frames and self.detection_ratio >= 0.15

    def add_observation(
        self,
        landmarks: np.ndarray | None,
        has_hand: bool,
        bbox: tuple[int, int, int, int] | None,
    ) -> None:
        feature_vector = (
            np.asarray(landmarks, dtype=np.float32)
            if landmarks is not None
            else np.zeros(RAW_LANDMARK_DIM, dtype=np.float32)
        )

        self.sequence_buffer.append(feature_vector)
        self.hand_history.append(1 if has_hand else 0)
        if bbox is not None:
            self.last_bbox = bbox

    def build_model_input(self) -> np.ndarray:
        if not self.sequence_buffer:
            return np.zeros((1, self.sequence_length, LANDMARK_DIM), dtype=np.float32)

        sequence = np.stack(list(self.sequence_buffer), axis=0).astype(np.float32)
        if len(sequence) == self.sequence_length:
            return transform_landmark_sequence(sequence)[np.newaxis, ...]

        indices = np.linspace(0, len(sequence) - 1, num=self.sequence_length)
        resampled = sequence[np.round(indices).astype(int)]
        return transform_landmark_sequence(resampled)[np.newaxis, ...]


class SignLanguageInterpreter:
    def __init__(
        self,
        model_path: str | Path | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.min_confidence = min_confidence
        self.labels = load_label_names()
        self.model = load_model(self.model_path, compile=False)
        self.hands = create_hands_detector()

    def close(self) -> None:
        self.hands.close()

    def create_session(
        self,
        min_frames: int = DEFAULT_MIN_FRAMES,
        transcript_threshold: float = DEFAULT_TRANSCRIPT_THRESHOLD,
        stable_frames: int = DEFAULT_STABLE_FRAMES,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> LiveRecognitionSession:
        return LiveRecognitionSession(
            sequence_length=SEQUENCE_LENGTH,
            min_frames=min_frames,
            transcript_builder=TranscriptBuilder(
                stable_frames=stable_frames,
                confidence_threshold=transcript_threshold,
                cooldown_seconds=cooldown_seconds,
            ),
        )

    def _resolve_model_path(self, model_path: str | Path | None) -> Path:
        if model_path is not None:
            return Path(model_path)

        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "No compatible word-level model file was found in the project directory."
        )

    def predict_from_frame(
        self,
        frame: np.ndarray,
        session: LiveRecognitionSession,
        min_confidence: float | None = None,
    ) -> PredictionResult:
        if frame is None or frame.size == 0:
            return PredictionResult(
                label=None,
                confidence=0.0,
                bbox=None,
                top_predictions=[],
                has_hand=False,
                ready=False,
                frames_buffered=session.frames_buffered,
                detection_ratio=session.detection_ratio,
            )

        landmarks, bbox, has_hand = self._extract_live_landmarks(frame)
        session.add_observation(landmarks, has_hand, bbox)

        if not session.ready:
            return PredictionResult(
                label=None,
                confidence=0.0,
                bbox=bbox,
                top_predictions=[],
                has_hand=has_hand,
                ready=False,
                frames_buffered=session.frames_buffered,
                detection_ratio=session.detection_ratio,
            )

        probabilities = self.model.predict(session.build_model_input(), verbose=0)[0]
        threshold = self.min_confidence if min_confidence is None else min_confidence
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "label": self.labels[index],
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ]

        best_index = int(top_indices[0])
        best_confidence = float(probabilities[best_index])
        best_label = self.labels[best_index] if best_confidence >= threshold else None

        return PredictionResult(
            label=best_label,
            confidence=best_confidence,
            bbox=bbox,
            top_predictions=top_predictions,
            has_hand=has_hand,
            ready=True,
            frames_buffered=session.frames_buffered,
            detection_ratio=session.detection_ratio,
        )

    def _extract_live_landmarks(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, bool]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return None, None, False

        hand_landmarks = result.multi_hand_landmarks[0]
        bbox = self._landmarks_to_bbox(hand_landmarks.landmark, frame.shape)
        return extract_raw_landmarks(hand_landmarks), bbox, True

    def _landmarks_to_bbox(
        self,
        landmarks: list,
        frame_shape: tuple[int, int, int],
    ) -> tuple[int, int, int, int]:
        height, width, _ = frame_shape
        xs = [landmark.x * width for landmark in landmarks]
        ys = [landmark.y * height for landmark in landmarks]

        x1 = max(int(min(xs)), 0)
        x2 = min(int(max(xs)), width)
        y1 = max(int(min(ys)), 0)
        y2 = min(int(max(ys)), height)

        padding = max(int(max(x2 - x1, y2 - y1) * 0.15), 10)
        return (
            max(x1 - padding, 0),
            max(y1 - padding, 0),
            min(x2 + padding, width),
            min(y2 + padding, height),
        )


class TextSpeaker:
    def __init__(self) -> None:
        self.engine = pyttsx3.init()
        current_rate = self.engine.getProperty("rate")
        self.engine.setProperty("rate", max(120, current_rate - 20))

    def speak(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self.engine.say(cleaned)
        self.engine.runAndWait()
