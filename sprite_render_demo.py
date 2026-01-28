import time
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = "models/hand_landmarker.task"
IMAGE_PATH = "object.png"  # Put your PNG here!
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# Interaction Constants
PINCH_DOWN = 0.40
PINCH_UP   = 0.60
SMOOTHING_FACTOR = 0.5
DEBOUNCE_FRAMES = 5
THROW_THRESHOLD = 15.0
FRICTION = 0.92

# --- GRAPHICS HELPER: ALPHA BLENDING ---
def overlay_transparent(background, overlay, x, y):
    """
    Overlays a 4-channel PNG (overlay) onto a 3-channel image (background)
    at position (x, y) respecting transparency.
    """
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape

    # 1. Safety Checks (Clipping)
    # If the object is completely off-screen, stop
    if x >= bg_w or y >= bg_h or x + ol_w < 0 or y + ol_h < 0:
        return background

    # Calculate intersection limits to avoid errors at screen edges
    # (This prevents the "crash when dragging off screen" bug)
    bg_x = max(0, x)
    bg_y = max(0, y)
    ol_x = max(0, -x)
    ol_y = max(0, -y)
    
    h = min(bg_h - bg_y, ol_h - ol_y)
    w = min(bg_w - bg_x, ol_w - ol_x)

    # 2. Slice the regions
    # The part of the background we will paint over
    roi = background[bg_y:bg_y+h, bg_x:bg_x+w]
    # The part of the PNG we are using
    img_part = overlay[ol_y:ol_y+h, ol_x:ol_x+w]

    # 3. Separate Channels
    # Color channels (BGR) and Alpha channel (A)
    img_color = img_part[:, :, :3]
    alpha_mask = img_part[:, :, 3] / 255.0  # Normalize 0-255 to 0.0-1.0

    # 4. Blend
    # Formula: Final = (Image * Alpha) + (Background * (1 - Alpha))
    # We use [:, :, None] to broadcast the single-channel alpha across 3 color channels
    composite = (img_color * alpha_mask[:, :, None]) + \
                (roi * (1.0 - alpha_mask[:, :, None]))

    # 5. Place back
    background[bg_y:bg_y+h, bg_x:bg_x+w] = composite.astype(np.uint8)
    return background

# --- HAND CONTROLLER (Same as before) ---
def get_dist(p1, p2): return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.history = deque(maxlen=5) 
        self.pinch_counter = 0
        self.is_pinching = False
        
    def process(self, landmarks, w, h):
        thumb_tip = (landmarks[4].x * w, landmarks[4].y * h)
        index_tip = (landmarks[8].x * w, landmarks[8].y * h)
        index_mcp = (landmarks[5].x * w, landmarks[5].y * h)
        pinky_mcp = (landmarks[17].x * w, landmarks[17].y * h)

        scale = get_dist(index_mcp, pinky_mcp) + 1e-6
        pinch_ratio = get_dist(thumb_tip, index_tip) / scale

        raw_x, raw_y = index_tip
        if self.prev_x == 0: self.prev_x, self.prev_y = raw_x, raw_y
        
        curr_x = SMOOTHING_FACTOR * raw_x + (1 - SMOOTHING_FACTOR) * self.prev_x
        curr_y = SMOOTHING_FACTOR * raw_y + (1 - SMOOTHING_FACTOR) * self.prev_y
        self.prev_x, self.prev_y = curr_x, curr_y

        if pinch_ratio < PINCH_DOWN: self.pinch_counter = min(DEBOUNCE_FRAMES, self.pinch_counter + 1)
        elif pinch_ratio > PINCH_UP: self.pinch_counter = max(0, self.pinch_counter - 1)
            
        if self.pinch_counter == DEBOUNCE_FRAMES: self.is_pinching = True
        elif self.pinch_counter == 0: self.is_pinching = False

        self.history.append((int(curr_x), int(curr_y)))
        vx, vy = 0, 0
        if len(self.history) >= 2:
            vx = curr_x - self.history[0][0]
            vy = curr_y - self.history[0][1]

        return (int(curr_x), int(curr_y)), self.is_pinching, (vx, vy)

# --- SETUP ---
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(1)
if not cap.isOpened(): cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
controller = HandController()

# --- LOAD ASSET ---
if os.path.exists(IMAGE_PATH):
    # Load Image (IMREAD_UNCHANGED includes Alpha Channel)
    sprite = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    # Resize if too huge
    if sprite.shape[0] > 200:
        scale = 200 / sprite.shape[0]
        sprite = cv2.resize(sprite, None, fx=scale, fy=scale)
    print(f"Loaded {IMAGE_PATH}")
else:
    print(f"Warning: {IMAGE_PATH} not found. Creating a synthetic circle.")
    # Create a 150x150 transparent image
    sprite = np.zeros((150, 150, 4), dtype=np.uint8)
    # Draw a filled circle: Blue with full opacity
    cv2.circle(sprite, (75, 75), 70, (255, 100, 0, 255), -1)
    # Draw a border
    cv2.circle(sprite, (75, 75), 70, (255, 255, 255, 255), 4)
    # Add a letter 'P'
    cv2.putText(sprite, "P", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255, 255), 5)

obj_h, obj_w = sprite.shape[:2]
obj_x, obj_y = 200.0, 200.0
obj_vx, obj_vy = 0.0, 0.0
state = "IDLE"
grab_offset = (0, 0)

# --- MAIN LOOP ---
t0 = time.time()
while True:
    ok, frame_bgr = cap.read()
    if not ok: break
    frame_bgr = cv2.flip(frame_bgr, 1)
    
    # MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, int((time.time()-t0)*1000))
    
    pointer = None
    is_pinching = False
    
    if result.hand_landmarks:
        pointer, is_pinching, (vx, vy) = controller.process(result.hand_landmarks[0], *frame_bgr.shape[:2][:2][::-1])
        
        # UI: Draw Pointer
        color = (0, 255, 0) if is_pinching else (0, 255, 255)
        cv2.circle(frame_bgr, pointer, 8, color, -1)

    # Physics Update
    if state == "IDLE":
        if pointer and is_pinching:
            # Hit test (Rectangle box approximation for now)
            if obj_x < pointer[0] < obj_x + obj_w and obj_y < pointer[1] < obj_y + obj_h:
                state = "DRAGGING"
                grab_offset = (pointer[0] - obj_x, pointer[1] - obj_y)
                obj_vx, obj_vy = 0, 0
    
    elif state == "DRAGGING":
        if pointer:
            obj_x = pointer[0] - grab_offset[0]
            obj_y = pointer[1] - grab_offset[1]
            if not is_pinching:
                if np.hypot(vx, vy) > THROW_THRESHOLD:
                    state = "THROWN"
                    obj_vx, obj_vy = vx, vy
                else: state = "IDLE"
    
    elif state == "THROWN":
        obj_x += obj_vx
        obj_y += obj_vy
        obj_vx *= FRICTION
        obj_vy *= FRICTION
        if np.hypot(obj_vx, obj_vy) < 1.0: state = "IDLE"
        
        # Bounce
        h, w = frame_bgr.shape[:2]
        if obj_x <= 0: obj_x, obj_vx = 0, -obj_vx * 0.8
        if obj_x + obj_w >= w: obj_x, obj_vx = w - obj_w, -obj_vx * 0.8
        if obj_y <= 0: obj_y, obj_vy = 0, -obj_vy * 0.8
        if obj_y + obj_h >= h: obj_y, obj_vy = h - obj_h, -obj_vy * 0.8

    # RENDER: Replaced cv2.rectangle with overlay_transparent
    frame_bgr = overlay_transparent(frame_bgr, sprite, int(obj_x), int(obj_y))

    cv2.putText(frame_bgr, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow("Sprite Demo", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()