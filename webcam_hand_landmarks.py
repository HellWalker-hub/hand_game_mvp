import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"

# VIDEO mode is easiest for a first live webcam MVP (loop + timestamp).
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (VideoCapture(0)). Try 1 if you have multiple cameras.")

# Timestamp in milliseconds (VIDEO mode requires it)
t0 = time.time()

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int((time.time() - t0) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w = frame_bgr.shape[:2]

    # Draw landmarks
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for lm in hand:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

    cv2.imshow("Hand Landmarks (press q to quit)", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()