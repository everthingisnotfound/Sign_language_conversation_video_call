from __future__ import annotations

import cv2

from sign_language_core import SignLanguageInterpreter, TextSpeaker


def draw_status(frame, prediction, transcript_state):
    active_label = (
        prediction.label
        or (prediction.top_predictions[0]["label"] if prediction.top_predictions else None)
        or transcript_state.candidate_label
        or "Waiting..."
    )
    cv2.putText(
        frame,
        f"Word: {active_label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (57, 255, 20),
        2,
    )
    cv2.putText(
        frame,
        f"Confidence: {prediction.confidence:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Buffered: {prediction.frames_buffered}/30",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Detection ratio: {prediction.detection_ratio:.2f}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Transcript: {transcript_state.transcript[-40:] or '-'}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
    )

    top_text = " | ".join(
        f"{item['label']}:{item['confidence']:.2f}"
        for item in prediction.top_predictions[:3]
    ) or "-"
    cv2.putText(
        frame,
        f"Top: {top_text}",
        (20, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        2,
    )

    if prediction.bbox:
        x1, y1, x2, y2 = prediction.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (57, 255, 20), 2)

    cv2.putText(
        frame,
        "S: speak  C: clear  Q: quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
    )


def main() -> None:
    interpreter = SignLanguageInterpreter(min_confidence=0.15)
    session = interpreter.create_session(min_frames=8, transcript_threshold=0.15)
    speaker = TextSpeaker()
    

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Unable to access webcam.")

    print("Press S to speak, C to clear the transcript, Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            prediction = interpreter.predict_from_frame(frame, session=session)
            transcript_state = session.transcript_builder.update(prediction.label, prediction.confidence)

            draw_status(frame, prediction, transcript_state)
            cv2.imshow("Sign Language Interpreter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                speaker.speak(transcript_state.transcript)
            elif key == ord("c"):
                session.clear()
            elif key == ord("q"):
                break
    finally:
        camera.release()
        interpreter.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()