from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from sign_language_core import LiveRecognitionSession, SignLanguageInterpreter

WEB_MIN_CONFIDENCE = 0.15
WEB_MIN_FRAMES = 6
WEB_STABLE_FRAMES = 3
WEB_COOLDOWN_SECONDS = 1.45
WEB_HAND_ABSENT_MIN_SECONDS = 0.45


def _allowed_origins() -> list[str]:
    configured = os.getenv(
        "SIGN_API_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


interpreter = SignLanguageInterpreter(min_confidence=WEB_MIN_CONFIDENCE)
sessions: dict[str, LiveRecognitionSession] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    interpreter.close()


app = FastAPI(
    title="Sign Language Interpreter API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrameRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image: str
    session_id: str | None = Field(default=None, alias="sessionId")
    min_confidence: float | None = Field(default=None, alias="minConfidence")


class SessionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")


def decode_image(image_payload: str) -> np.ndarray:
    encoded = image_payload.split(",", 1)[-1]

    try:
        binary = base64.b64decode(encoded)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid base64 image payload."
        ) from exc

    image = cv2.imdecode(np.frombuffer(binary, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")
    return image


def get_session(session_id: str) -> LiveRecognitionSession:
    session = sessions.get(session_id)
    if session is None:
        session = interpreter.create_session(
            min_frames=WEB_MIN_FRAMES,
            transcript_threshold=WEB_MIN_CONFIDENCE,
            stable_frames=WEB_STABLE_FRAMES,
            cooldown_seconds=WEB_COOLDOWN_SECONDS,
            hand_absent_min_seconds=WEB_HAND_ABSENT_MIN_SECONDS,
        )
        sessions[session_id] = session
    return session


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "labelsLoaded": len(interpreter.labels),
        "modelPath": str(interpreter.model_path),
        "mode": "word-sequence",
        "sequenceLength": 30,
        "minConfidence": WEB_MIN_CONFIDENCE,
        "minFrames": WEB_MIN_FRAMES,
        "stableFrames": WEB_STABLE_FRAMES,
        "handAbsentMinSeconds": WEB_HAND_ABSENT_MIN_SECONDS,
    }


@app.get("/api/labels")
def labels() -> dict[str, list[str]]:
    return {"labels": interpreter.labels}


@app.post("/api/reset-session")
def reset_session(payload: SessionRequest) -> dict[str, object]:
    session = get_session(payload.session_id)
    session.clear()
    return {"sessionId": payload.session_id, "transcript": ""}


@app.post("/api/predict-frame")
def predict_frame(payload: FrameRequest) -> dict[str, object]:
    frame = decode_image(payload.image)
    session_id = payload.session_id or uuid4().hex
    session = get_session(session_id)

    prediction = interpreter.predict_from_frame(
        frame,
        session=session,
        min_confidence=payload.min_confidence,
    )
    label_for_transcript = prediction.label if prediction.has_hand else None
    confidence_for_transcript = (
        prediction.confidence if prediction.has_hand else 0.0
    )
    transcript_state = session.transcript_builder.update(
        label_for_transcript,
        confidence_for_transcript,
        prediction.has_hand,
    )

    return {
        "sessionId": session_id,
        "hasHand": prediction.has_hand,
        "ready": prediction.ready,
        "framesBuffered": prediction.frames_buffered,
        "detectionRatio": prediction.detection_ratio,
        "label": prediction.label,
        "confidence": prediction.confidence,
        "bbox": prediction.bbox,
        "topPredictions": prediction.top_predictions,
        "transcript": transcript_state.transcript,
        "committedLabel": transcript_state.committed_label,
        "candidateLabel": transcript_state.candidate_label,
        "candidateHits": transcript_state.candidate_hits,
        "updated": transcript_state.updated,
    }


if __name__ == "__main__":
    uvicorn.run(
        "sign_language_api:app",
        host=os.getenv("SIGN_API_HOST", "127.0.0.1"),
        port=int(os.getenv("SIGN_API_PORT", "8000")),
        reload=False,
    )
