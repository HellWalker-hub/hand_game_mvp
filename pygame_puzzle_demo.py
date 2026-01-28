import pygame
import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = "models/hand_landmarker.task"
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
WINDOW_SIZE = (1280, 720)

# Interaction Constants
PINCH_DOWN = 0.40
PINCH_UP   = 0.60
SMOOTHING_FACTOR = 0.5
DEBOUNCE_FRAMES = 3
SNAP_DISTANCE = 50.0  # Pixel distance to trigger "Snap"

# --- CONTROLLER (The Brain - Same as before) ---
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
        return (int(curr_x), int(curr_y)), self.is_pinching

# --- GAME OBJECTS ---
class PuzzlePiece:
    def __init__(self, x, y, w, h, color, is_target=False):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = color
        self.is_target = is_target # If True, this is the "Slot" (immovable)
        self.is_locked = False     # If True, puzzle solved
        self.grab_offset = (0, 0)
        self.is_dragging = False

    def draw(self, surface):
        if self.is_target:
            # Draw outline only (The Slot)
            pygame.draw.rect(surface, self.color, self.rect, 3)
        else:
            # Draw filled shape (The Piece)
            pygame.draw.rect(surface, self.color, self.rect)
            # Add a white border to pop
            pygame.draw.rect(surface, (255, 255, 255), self.rect, 2)

# --- PYGAME SETUP ---
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Puzzle Snap Demo")
font = pygame.font.SysFont("Arial", 36)

# --- MEDIAPIPE SETUP ---
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

# --- LEVEL SETUP ---
# 1. The Slot (Target) - Fixed location
slot = PuzzlePiece(800, 300, 150, 100, (200, 200, 200), is_target=True)

# 2. The Battery (Movable)
battery = PuzzlePiece(200, 300, 150, 100, (0, 255, 0), is_target=False)

# Main Loop
running = True
t0 = time.time()

while running:
    # 1. Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q: running = False

    # 2. Camera Input
    ok, frame_bgr = cap.read()
    if not ok: break
    frame_bgr = cv2.flip(frame_bgr, 1)
    
    # 3. MediaPipe Processing
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int((time.time() - t0) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)

    # 4. Render Webcam as Background
    # Pygame needs Transposed (Rotated) and RGB data
    frame_rgb = np.rot90(rgb_frame)
    frame_surface = pygame.surfarray.make_surface(frame_rgb)
    frame_surface = pygame.transform.flip(frame_surface, True, False) # Fix mirroing
    # Scale to window if needed (optional)
    screen.blit(frame_surface, (0, 0))

    # 5. Hand Logic
    pointer = None
    is_pinching = False
    
    if result.hand_landmarks:
        # Get coordinates scaled to Window Size
        pointer, is_pinching = controller.process(result.hand_landmarks[0], WINDOW_SIZE[0], WINDOW_SIZE[1])
        
        # Draw Hand Cursor
        pygame.draw.circle(screen, (0, 255, 0) if is_pinching else (0, 255, 255), pointer, 10)

        # --- GAME LOGIC ---
        if not battery.is_locked:
            # Check Drag Start
            if is_pinching and not battery.is_dragging:
                if battery.rect.collidepoint(pointer):
                    battery.is_dragging = True
                    battery.grab_offset = (pointer[0] - battery.rect.x, pointer[1] - battery.rect.y)
            
            # Dragging
            if battery.is_dragging:
                if not is_pinching:
                    # Released! Check Snap
                    dist = np.hypot(battery.rect.centerx - slot.rect.centerx, 
                                    battery.rect.centery - slot.rect.centery)
                    
                    if dist < SNAP_DISTANCE:
                        # SNAP!
                        battery.rect.center = slot.rect.center # Force alignment
                        battery.is_locked = True
                        battery.color = (0, 200, 255) # Change color to show success
                    
                    battery.is_dragging = False
                else:
                    # Move
                    battery.rect.x = pointer[0] - battery.grab_offset[0]
                    battery.rect.y = pointer[1] - battery.grab_offset[1]

    # 6. Draw Objects (Slot first so Battery is on top)
    slot.draw(screen)
    battery.draw(screen)
    
    # UI Overlay
    if battery.is_locked:
        text = font.render("SYSTEM ONLINE! (Puzzle Solved)", True, (0, 255, 0))
        screen.blit(text, (WINDOW_SIZE[0]//2 - 200, 50))
    else:
        text = font.render("Drag Battery to Slot...", True, (255, 255, 255))
        screen.blit(text, (50, 50))

    pygame.display.flip()

cap.release()
pygame.quit()