import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = "models/hand_landmarker.task"
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# Landmarks
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
PINKY_MCP = 17

# Interaction Parameters - STICKY GRIP UPDATE
PINCH_DOWN = 0.40       # Easier to grab
PINCH_UP   = 0.60       # Harder to release (prevents accidental drops)
SMOOTHING_FACTOR = 0.5  
DEBOUNCE_FRAMES = 5     # Increased from 3 to 5 (filters motion blur better)
THROW_THRESHOLD = 15.0  
FRICTION = 0.92         

# --- HELPER FUNCTIONS ---
def get_dist(p1, p2):
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.history = deque(maxlen=5) 
        self.pinch_counter = 0
        self.is_pinching = False
        
    def process(self, landmarks, w, h):
        # 1. Extract raw coordinates
        thumb_tip = (landmarks[THUMB_TIP].x * w, landmarks[THUMB_TIP].y * h)
        index_tip = (landmarks[INDEX_TIP].x * w, landmarks[INDEX_TIP].y * h)
        index_mcp = (landmarks[INDEX_MCP].x * w, landmarks[INDEX_MCP].y * h)
        pinky_mcp = (landmarks[PINKY_MCP].x * w, landmarks[PINKY_MCP].y * h)

        # 2. Pinch Logic
        scale = get_dist(index_mcp, pinky_mcp) + 1e-6
        pinch_dist = get_dist(thumb_tip, index_tip)
        pinch_ratio = pinch_dist / scale

        # 3. Smoothing
        raw_x, raw_y = index_tip
        if self.prev_x == 0 and self.prev_y == 0:
            self.prev_x, self.prev_y = raw_x, raw_y
        
        curr_x = SMOOTHING_FACTOR * raw_x + (1 - SMOOTHING_FACTOR) * self.prev_x
        curr_y = SMOOTHING_FACTOR * raw_y + (1 - SMOOTHING_FACTOR) * self.prev_y
        self.prev_x, self.prev_y = curr_x, curr_y

        # 4. Debounce (Hysteresis)
        if pinch_ratio < PINCH_DOWN:
            self.pinch_counter = min(DEBOUNCE_FRAMES, self.pinch_counter + 1)
        elif pinch_ratio > PINCH_UP:
            self.pinch_counter = max(0, self.pinch_counter - 1)
            
        # Only change state if counter hits limits
        if self.pinch_counter == DEBOUNCE_FRAMES:
            self.is_pinching = True
        elif self.pinch_counter == 0:
            self.is_pinching = False

        # 5. Vector Velocity (dx, dy)
        self.history.append((int(curr_x), int(curr_y)))
        
        vx, vy = 0, 0
        if len(self.history) >= 2:
            old_x, old_y = self.history[0]
            vx = curr_x - old_x
            vy = curr_y - old_y

        return (int(curr_x), int(curr_y)), self.is_pinching, (vx, vy)

# --- SETUP ---
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)

# UPDATED: Use Camera Index 1
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Warning: Camera 1 failed. Trying Camera 0...")
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

controller = HandController()

# --- GAME STATE ---
rect_x, rect_y = 200.0, 200.0
rect_w, rect_h = 150, 100
rect_vx, rect_vy = 0.0, 0.0

state = "IDLE" # IDLE, DRAGGING, THROWN
grab_offset = (0, 0)

print("Sticky Physics Demo Started.")
print("- Camera set to index 1.")
print("- Grip is stronger (harder to accidentally release).")

t0 = time.time()

while True:
    ok, frame_bgr = cap.read()
    if not ok: break

    frame_bgr = cv2.flip(frame_bgr, 1)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int((time.time() - t0) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    h, w = frame_bgr.shape[:2]

    pointer = None
    is_pinching = False
    hand_vel = (0, 0)
    
    if result.hand_landmarks:
        pointer, is_pinching, hand_vel = controller.process(result.hand_landmarks[0], w, h)
        
        # Draw Pointer
        color = (0, 255, 0) if is_pinching else (0, 255, 255)
        cv2.circle(frame_bgr, pointer, 8, color, -1)
        
        # Debug: Show pinch strength bar
        # Visualizes when the system thinks you are releasing
        debug_x = int(controller.pinch_counter / DEBOUNCE_FRAMES * 50)
        cv2.rectangle(frame_bgr, (pointer[0]+10, pointer[1]-10), (pointer[0]+10+debug_x, pointer[1]-5), (255,0,255), -1)

    # --- PHYSICS ENGINE ---

    if state == "IDLE":
        if pointer and is_pinching:
            # Hit test
            if rect_x < pointer[0] < rect_x + rect_w and rect_y < pointer[1] < rect_y + rect_h:
                state = "DRAGGING"
                grab_offset = (pointer[0] - rect_x, pointer[1] - rect_y)
                rect_vx, rect_vy = 0, 0

    elif state == "DRAGGING":
        if pointer:
            # Move object
            rect_x = pointer[0] - grab_offset[0]
            rect_y = pointer[1] - grab_offset[1]
            
            # The Critical Fix: Only THROW if explicitly released
            if not is_pinching:
                speed = np.hypot(hand_vel[0], hand_vel[1])
                if speed > THROW_THRESHOLD:
                    state = "THROWN"
                    rect_vx, rect_vy = hand_vel[0], hand_vel[1]
                else:
                    state = "IDLE"

    elif state == "THROWN":
        rect_x += rect_vx
        rect_y += rect_vy
        rect_vx *= FRICTION
        rect_vy *= FRICTION
        
        if np.hypot(rect_vx, rect_vy) < 1.0:
            state = "IDLE"
            rect_vx, rect_vy = 0, 0
            
        # Bounce
        if rect_x <= 0: rect_x, rect_vx = 0, -rect_vx * 0.8
        if rect_x + rect_w >= w: rect_x, rect_vx = w - rect_w, -rect_vx * 0.8
        if rect_y <= 0: rect_y, rect_vy = 0, -rect_vy * 0.8
        if rect_y + rect_h >= h: rect_y, rect_vy = h - rect_h, -rect_vy * 0.8

    # --- RENDER ---
    if state == "DRAGGING": color = (0, 0, 255)
    elif state == "THROWN": color = (0, 255, 0)
    else: color = (255, 0, 0)
    
    cv2.rectangle(frame_bgr, (int(rect_x), int(rect_y)), 
                  (int(rect_x + rect_w), int(rect_y + rect_h)), color, -1)
    
    cv2.putText(frame_bgr, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Sticky Physics Throw", frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()