"""Live webcam hand tracking with finger-joint lines.

Press Q or Esc to close the camera window.
"""

from __future__ import annotations

import cv2
import mediapipe as mp


# MediaPipe's standard hand topology: wrist, palm, and each finger's bones.
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def main() -> None:
    hands_api = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera. Check its permissions or try another camera index.")

    # Ask for a practical preview size; cameras may choose the nearest supported mode.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with hands_api.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
    ) as hands:
        try:
            while camera.isOpened():
                ok, frame = camera.read()
                if not ok:
                    print("Could not receive a frame from the camera.")
                    break

                # Mirror the preview so moving left/right feels natural.
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # The connections are the lines between wrist, palm, and finger joints.
                        drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            HAND_CONNECTIONS,
                            drawing_styles.get_default_hand_landmarks_style(),
                            drawing_styles.get_default_hand_connections_style(),
                        )

                cv2.putText(
                    frame,
                    "Show a hand to the camera  |  Q / Esc: quit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Hand Tracking Matrix", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
