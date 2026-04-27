from __future__ import annotations

import cv2
import mediapipe as mp


def main() -> None:
    print("The new word-level pipeline uses hand landmarks, not hist.pkl segmentation.")
    print("This tool now acts as a webcam hand-detection preview. Press Q to quit.")

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise SystemExit("Unable to access the webcam.")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils
    connections = mp.solutions.hands.HAND_CONNECTIONS

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    drawing.draw_landmarks(frame, hand_landmarks, connections)
                status_text = "Hand detected"
                status_color = (0, 255, 0)
            else:
                status_text = "Show your signing hand"
                status_color = (0, 180, 255)

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
            )
            cv2.putText(
                frame,
                "Q: quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

            cv2.imshow("Hand Detection Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()