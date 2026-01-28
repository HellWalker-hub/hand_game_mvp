import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# Landmark indices (MediaPipe Hands standard)
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
PINKY_MCP = 17

# Pinch thresholds (tweak later)
PINCH_DOWN = 0.40   # smaller = harder to pinch
PINCH_UP   = 0.55   # release threshold (hysteresis)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,  # start with one hand for simplicity
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (VideoCapture(0)). Try 1 if you have multiple cameras.")

t0 = time.time()

# Simple draggable object
rect_w, rect_h = 140, 90
rect_x, rect_y = 220, 160
dragging = False
grab_offset = (0, 0)

def l2(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - t0) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w = frame_bgr.shape[:2]

    pointer = None
    pinch_ratio = None

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Convert landmarks to pixel coords
        pts = [(lm.x * w, lm.y * h) for lm in hand]

        thumb_tip = pts[THUMB_TIP]
        index_tip = pts[INDEX_TIP]
        # Hand scale proxy (palm width)
        scale = l2(pts[INDEX_MCP], pts[PINKY_MCP]) + 1e-6

        pinch_dist = l2(thumb_tip, index_tip)
        pinch_ratio = pinch_dist / scale

        pointer = (int(index_tip[0]), int(index_tip[1]))

        # Visualize pointer + pinch line
        cv2.circle(frame_bgr, pointer, 6, (0, 255, 0), -1)
        cv2.line(frame_bgr,
                 (int(thumb_tip[0]), int(thumb_tip[1])),
                 (int(index_tip[0]), int(index_tip[1])),
                 (0, 255, 255), 2)

        # Pinch state with hysteresis
        if not dragging and pinch_ratio < PINCH_DOWN:
            # Start drag if pointer is inside rectangle
            if rect_x <= pointer[0] <= rect_x + rect_w and rect_y <= pointer[1] <= rect_y + rect_h:
                dragging = True
                grab_offset = (pointer[0] - rect_x, pointer[1] - rect_y)

        if dragging and pinch_ratio > PINCH_UP:
            dragging = False

        # Update rectangle position while dragging
        if dragging:
            rect_x = clamp(pointer[0] - grab_offset[0], 0, w - rect_w)
            rect_y = clamp(pointer[1] - grab_offset[1], 0, h - rect_h)

        # Draw a few landmarks for confidence
        for (x, y) in pts:
            cv2.circle(frame_bgr, (int(x), int(y)), 2, (255, 255, 255), -1)

    # Draw draggable rectangle
    color = (0, 0, 255) if dragging else (255, 0, 0)
    cv2.rectangle(frame_bgr, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), color, 2)

    # Debug text
    if pinch_ratio is not None:
        cv2.putText(frame_bgr, f"pinch_ratio: {pinch_ratio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_bgr, f"state: {'DRAG' if dragging else 'HOVER'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame_bgr, "No hand", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Pinch Drag Demo (press q to quit)", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()