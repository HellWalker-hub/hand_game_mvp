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
IMAGE_PATH = "object.png" 
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720

# Interaction Constants
PINCH_DOWN = 0.40
PINCH_UP   = 0.60
SMOOTHING_FACTOR = 0.5
DEBOUNCE_FRAMES = 5
THROW_THRESHOLD = 15.0
FRICTION = 0.92

# --- GRAPHICS HELPER (Alpha Blend) ---
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape
    
    if x >= bg_w or y >= bg_h or x + ol_w < 0 or y + ol_h < 0: return background
    
    bg_x, bg_y = max(0, x), max(0, y)
    ol_x, ol_y = max(0, -x), max(0, -y)
    
    h = min(bg_h - bg_y, ol_h - ol_y)
    w = min(bg_w - bg_x, ol_w - ol_x)
    
    roi = background[bg_y:bg_y+h, bg_x:bg_x+w]
    img_part = overlay[ol_y:ol_y+h, ol_x:ol_x+w]
    
    img_color = img_part[:, :, :3]
    alpha_mask = img_part[:, :, 3] / 255.0
    
    composite = (img_color * alpha_mask[:, :, None]) + \
                (roi * (1.0 - alpha_mask[:, :, None]))
    
    background[bg_y:bg_y+h, bg_x:bg_x+w] = composite.astype(np.uint8)
    return background

# --- CONTROLLER CLASS (The "Brain") ---
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

        # Smoothing
        raw_x, raw_y = index_tip
        if self.prev_x == 0: self.prev_x, self.prev_y = raw_x, raw_y
        curr_x = SMOOTHING_FACTOR * raw_x + (1 - SMOOTHING_FACTOR) * self.prev_x
        curr_y = SMOOTHING_FACTOR * raw_y + (1 - SMOOTHING_FACTOR) * self.prev_y
        self.prev_x, self.prev_y = curr_x, curr_y

        # Debounce
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

# --- GAME OBJECT CLASS ---
class GameObject:
    def __init__(self, x, y, img, name="Obj"):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.img = img
        self.h, self.w = img.shape[:2]
        self.state = "IDLE" # IDLE, DRAGGING, THROWN
        self.grab_offset = (0, 0)
        self.name = name

    def update(self, w, h):
        # Physics Step
        if self.state == "THROWN":
            self.x += self.vx
            self.y += self.vy
            self.vx *= FRICTION
            self.vy *= FRICTION
            
            # Stop if slow
            if np.hypot(self.vx, self.vy) < 1.0:
                self.state = "IDLE"
            
            # Bounce
            if self.x <= 0: self.x, self.vx = 0, -self.vx * 0.8
            if self.x + self.w >= w: self.x, self.vx = w - self.w, -self.vx * 0.8
            if self.y <= 0: self.y, self.vy = 0, -self.vy * 0.8
            if self.y + self.h >= h: self.y, self.vy = h - self.h, -self.vy * 0.8

    def is_hit(self, px, py):
        return self.x < px < self.x + self.w and self.y < py < self.y + self.h

    def draw(self, frame):
        # Draw green border if dragged
        frame = overlay_transparent(frame, self.img, int(self.x), int(self.y))
        if self.state == "DRAGGING":
            cv2.rectangle(frame, (int(self.x), int(self.y)), 
                         (int(self.x + self.w), int(self.y + self.h)), (0, 255, 0), 2)
        return frame

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

# --- LOAD ASSET (Create fake one if missing) ---
if os.path.exists(IMAGE_PATH):
    base_sprite = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if base_sprite.shape[0] > 150:
        s = 150 / base_sprite.shape[0]
        base_sprite = cv2.resize(base_sprite, None, fx=s, fy=s)
else:
    base_sprite = np.zeros((100, 100, 4), dtype=np.uint8)
    cv2.circle(base_sprite, (50, 50), 45, (255, 100, 0, 255), -1)
    cv2.putText(base_sprite, "Obj", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255,255), 2)

# --- CREATE MULTIPLE OBJECTS ---
objects = [
    GameObject(100, 100, base_sprite, "A"),
    GameObject(400, 200, base_sprite, "B"),
    GameObject(700, 100, base_sprite, "C")
]
active_drag_index = -1

print("Multi-Object Demo Started.")

t0 = time.time()
while True:
    ok, frame_bgr = cap.read()
    if not ok: break
    frame_bgr = cv2.flip(frame_bgr, 1)
    h_screen, w_screen = frame_bgr.shape[:2]
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, int((time.time()-t0)*1000))
    
    pointer = None
    is_pinching = False
    
    # 1. PROCESS HAND
    if result.hand_landmarks:
        pointer, is_pinching, (vx, vy) = controller.process(result.hand_landmarks[0], w_screen, h_screen)
        cv2.circle(frame_bgr, pointer, 8, (0, 255, 0) if is_pinching else (0, 255, 255), -1)

    # 2. GAME LOGIC (The critical part for multiple items)
    if pointer:
        # If we are NOT already dragging something
        if active_drag_index == -1:
            if is_pinching:
                # Iterate BACKWARDS (top to bottom) to grab the top-most item
                for i in range(len(objects) - 1, -1, -1):
                    if objects[i].is_hit(pointer[0], pointer[1]):
                        active_drag_index = i
                        obj = objects[i]
                        obj.state = "DRAGGING"
                        obj.grab_offset = (pointer[0] - obj.x, pointer[1] - obj.y)
                        obj.vx, obj.vy = 0, 0
                        
                        # Move to end of list (bring to front)
                        objects.pop(i)
                        objects.append(obj)
                        active_drag_index = len(objects) - 1
                        break
        
        # If we ARE dragging something
        else:
            obj = objects[active_drag_index]
            obj.x = pointer[0] - obj.grab_offset[0]
            obj.y = pointer[1] - obj.grab_offset[1]
            
            if not is_pinching:
                # Release
                active_drag_index = -1
                if np.hypot(vx, vy) > THROW_THRESHOLD:
                    obj.state = "THROWN"
                    obj.vx, obj.vy = vx, vy
                else:
                    obj.state = "IDLE"

    # 3. UPDATE & DRAW ALL
    for obj in objects:
        obj.update(w_screen, h_screen)
        frame_bgr = obj.draw(frame_bgr)

    cv2.imshow("Multi-Object Manager", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()