from __future__ import annotations

import math
from collections import defaultdict

import cv2
import mediapipe as mp
import numpy as np

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

EXTENDED_DISTANCE_RATIO = 1.15
FINGER_ANGLE_THRESHOLD = 135.0
THUMB_ANGLE_THRESHOLD = 130.0

L_CONFIRM_FRAMES = 4
L_LOST_FRAMES = 15
HAND_MISSING_FRAMES = 5

POSITION_SMOOTHING = 0.35

PINCH_START_DISTANCE = 0.035
PINCH_CONFIRM_FRAMES = 2

FLASH_FRAMES = 6


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def angle(a, b, c):
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)

    dot = v1[0] * v2[0] + v1[1] * v2[1]

    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 == 0 or mag2 == 0:
        return 180.0

    value = max(-1.0, min(1.0, dot / (mag1 * mag2)))

    return math.degrees(math.acos(value))


def finger_extended(landmarks, mcp, pip, tip):
    wrist = landmarks.landmark[WRIST]
    mcp = landmarks.landmark[mcp]
    pip = landmarks.landmark[pip]
    tip = landmarks.landmark[tip]

    distance_ok = (
        distance(wrist, tip)
        > distance(wrist, pip) * EXTENDED_DISTANCE_RATIO
    )

    angle_ok = angle(mcp, pip, tip) > FINGER_ANGLE_THRESHOLD

    return distance_ok or angle_ok


def thumb_extended(landmarks):
    wrist = landmarks.landmark[WRIST]
    cmc = landmarks.landmark[THUMB_CMC]
    mcp = landmarks.landmark[THUMB_MCP]
    ip = landmarks.landmark[THUMB_IP]
    tip = landmarks.landmark[THUMB_TIP]

    distance_ok = (
        distance(wrist, tip)
        > distance(wrist, mcp) * 1.10
    )

    angle_ok = (
        angle(cmc, mcp, ip) > THUMB_ANGLE_THRESHOLD
        and angle(mcp, ip, tip) > THUMB_ANGLE_THRESHOLD
    )

    return distance_ok or angle_ok


def finger_states(landmarks):
    return {
        "thumb": thumb_extended(landmarks),
        "index": finger_extended(
            landmarks, INDEX_MCP, INDEX_PIP, INDEX_TIP
        ),
        "middle": finger_extended(
            landmarks, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP
        ),
        "ring": finger_extended(
            landmarks, RING_MCP, RING_PIP, RING_TIP
        ),
        "pinky": finger_extended(
            landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP
        ),
    }


def is_l_shape(states):
    return (
        states["thumb"]
        and states["index"]
        and not states["middle"]
        and not states["ring"]
        and not states["pinky"]
    )


def is_pinching(landmarks):
    thumb = landmarks.landmark[THUMB_TIP]
    index = landmarks.landmark[INDEX_TIP]

    return distance(thumb, index) < PINCH_START_DISTANCE


def to_px(point, width, height):
    return int(point.x * width), int(point.y * height)


def smooth(old, new):
    if old is None:
        return new

    return (
        int(old[0] + (new[0] - old[0]) * POSITION_SMOOTHING),
        int(old[1] + (new[1] - old[1]) * POSITION_SMOOTHING),
    )


def grayscale_inside(frame, polygon):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask, [polygon], 255)

    result = frame.copy()
    result[mask == 255] = gray[mask == 255]

    return result


def main():
    drawing = mp.solutions.drawing_utils
    hands_api = mp.solutions.hands

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    positive = defaultdict(int)
    negative = defaultdict(int)
    missing = defaultdict(int)
    pinch_frames = defaultdict(int)

    stable = {
        "Left": False,
        "Right": False,
    }

    points = {
        "Left": {
            "thumb": None,
            "index": None,
        },
        "Right": {
            "thumb": None,
            "index": None,
        },
    }

    color_mode = True
    was_pinching = False
    flash_frames = 0

    with hands_api.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        try:
            while camera.isOpened():
                ok, frame = camera.read()

                if not ok:
                    break

                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                debug = []
                detected_hands = set()
                current_landmarks = {}

                if results.multi_hand_landmarks:
                    for i, landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):
                        label = (
                            results.multi_handedness[i]
                            .classification[0]
                            .label
                        )

                        detected_hands.add(label)
                        current_landmarks[label] = landmarks
                        missing[label] = 0

                        drawing.draw_landmarks(
                            frame,
                            landmarks,
                            HAND_CONNECTIONS,
                            drawing.DrawingSpec(
                                color=(255, 255, 255),
                                thickness=1,
                                circle_radius=1,
                            ),
                            drawing.DrawingSpec(
                                color=(255, 255, 255),
                                thickness=1,
                            ),
                        )

                        states = finger_states(landmarks)
                        detected = is_l_shape(states)

                        debug.append(
                            f"{label}: "
                            + " ".join(
                                f"{k}:{int(v)}"
                                for k, v in states.items()
                            )
                        )

                        if detected:
                            positive[label] += 1
                            negative[label] = 0

                            if positive[label] >= L_CONFIRM_FRAMES:
                                stable[label] = True

                        else:
                            positive[label] = 0
                            negative[label] += 1

                            if negative[label] >= L_LOST_FRAMES:
                                stable[label] = False
                                points[label]["thumb"] = None
                                points[label]["index"] = None

                        if stable[label]:
                            thumb = to_px(
                                landmarks.landmark[THUMB_TIP],
                                width,
                                height,
                            )

                            index = to_px(
                                landmarks.landmark[INDEX_TIP],
                                width,
                                height,
                            )

                            points[label]["thumb"] = smooth(
                                points[label]["thumb"],
                                thumb,
                            )

                            points[label]["index"] = smooth(
                                points[label]["index"],
                                index,
                            )

                for label in ("Left", "Right"):
                    if label not in detected_hands:
                        missing[label] += 1
                        positive[label] = 0
                        pinch_frames[label] = 0

                        if missing[label] >= HAND_MISSING_FRAMES:
                            stable[label] = False
                            negative[label] = 0

                            points[label]["thumb"] = None
                            points[label]["index"] = None

                valid = []

                for label in ("Left", "Right"):
                    if stable[label]:
                        thumb = points[label]["thumb"]
                        index = points[label]["index"]

                        if thumb and index:
                            valid.append(
                                (
                                    label,
                                    thumb,
                                    index,
                                )
                            )

                if len(valid) == 2:
                    valid.sort(
                        key=lambda h:
                        (h[1][0] + h[2][0]) / 2
                    )

                    _, left_thumb, left_index = valid[0]
                    _, right_thumb, right_index = valid[1]

                    polygon = np.array(
                        [
                            left_thumb,
                            right_thumb,
                            right_index,
                            left_index,
                        ],
                        dtype=np.int32,
                    )

                    pinching = False

                    for label, _, _ in valid:
                        if label not in current_landmarks:
                            continue

                        if is_pinching(
                            current_landmarks[label]
                        ):
                            pinch_frames[label] += 1
                        else:
                            pinch_frames[label] = 0

                        if (
                            pinch_frames[label]
                            >= PINCH_CONFIRM_FRAMES
                        ):
                            pinching = True

                    if pinching and not was_pinching:
                        color_mode = not color_mode
                        flash_frames = FLASH_FRAMES

                    was_pinching = pinching

                    overlay = frame.copy()

                    cv2.fillPoly(
                        overlay,
                        [polygon],
                        (0, 255, 255),
                    )

                    cv2.addWeighted(
                        overlay,
                        0.12,
                        frame,
                        0.88,
                        0,
                        frame,
                    )

                    if not color_mode:
                        frame = grayscale_inside(
                            frame,
                            polygon,
                        )

                    if flash_frames > 0:
                        progress = flash_frames / FLASH_FRAMES
                        alpha = progress * progress

                        mask = np.zeros(
                            frame.shape[:2],
                            dtype=np.uint8,
                        )

                        cv2.fillPoly(
                            mask,
                            [polygon],
                            255,
                        )

                        white = np.full_like(
                            frame,
                            255,
                        )

                        flashed = cv2.addWeighted(
                            frame,
                            1 - alpha,
                            white,
                            alpha,
                            0,
                        )

                        frame[mask == 255] = flashed[
                            mask == 255
                        ]

                        flash_frames -= 1

                    cv2.line(
                        frame,
                        left_thumb,
                        right_thumb,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.line(
                        frame,
                        left_index,
                        right_index,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.line(
                        frame,
                        left_thumb,
                        left_index,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    cv2.line(
                        frame,
                        right_thumb,
                        right_index,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    status = (
                        "BLACK & WHITE"
                        if not color_mode
                        else "FRAME ACTIVE"
                    )

                    cv2.putText(
                        frame,
                        status,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                else:
                    was_pinching = False
                    pinch_frames["Left"] = 0
                    pinch_frames["Right"] = 0

                    cv2.putText(
                        frame,
                        "Show TWO L-shaped hands",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                for i, text in enumerate(debug):
                    cv2.putText(
                        frame,
                        text,
                        (20, 70 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    frame,
                    f"Hands detected: "
                    f"{len(results.multi_hand_landmarks or [])}",
                    (20, height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    "Q / Esc: quit",
                    (20, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    "Hand Tracking Matrix",
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()