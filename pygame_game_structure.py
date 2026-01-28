import pygame
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# --- CONFIGURATION ---
WINDOW_SIZE = (1280, 720)
MODEL_PATH = "models/hand_landmarker.task"

# --- INPUT SYSTEM (REVERTED TO STANDARD STABLE VERSION) ---
class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.pinch_counter = 0
        self.is_pinching = False
        
        # 1. REVERT: Standard Thresholds
        self.PINCH_DOWN = 0.40
        self.PINCH_UP = 0.85
        
        # 2. REVERT: Standard Debounce Logic (3 Frames)
        self.DEBOUNCE_FRAMES = 3
        
    def process(self, landmarks, w, h):
        thumb = landmarks[4]
        index = landmarks[8]
        
        # Calculate Pinch
        dist = np.hypot(thumb.x - index.x, thumb.y - index.y)
        scale = np.hypot(landmarks[5].x - landmarks[17].x, landmarks[5].y - landmarks[17].y) + 1e-6
        ratio = dist / scale

        # Smoothing
        SMOOTH = 0.5 # Balanced smoothing
        raw_x, raw_y = int(index.x * w), int(index.y * h)
        if self.prev_x == 0: self.prev_x, self.prev_y = raw_x, raw_y
        curr_x = int(SMOOTH * raw_x + (1 - SMOOTH) * self.prev_x)
        curr_y = int(SMOOTH * raw_y + (1 - SMOOTH) * self.prev_y)
        self.prev_x, self.prev_y = curr_x, curr_y

        # Standard Debounce Logic (No fast release)
        if ratio < self.PINCH_DOWN:
            self.pinch_counter = min(self.DEBOUNCE_FRAMES, self.pinch_counter + 1)
        elif ratio > self.PINCH_UP:
            self.pinch_counter = max(0, self.pinch_counter - 1)
        
        if self.pinch_counter == self.DEBOUNCE_FRAMES: self.is_pinching = True
        elif self.pinch_counter == 0: self.is_pinching = False

        return (curr_x, curr_y), self.is_pinching

# --- SCENE SYSTEM ---
class Scene:
    def __init__(self, manager):
        self.manager = manager
    def update(self, pointer, is_pinching): pass
    def draw(self, screen): pass

class TitleScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 60)
        self.subfont = pygame.font.SysFont("Arial", 30)
        self.start_btn = pygame.Rect(WINDOW_SIZE[0]//2 - 100, WINDOW_SIZE[1]//2, 200, 80)
        
    def update(self, pointer, is_pinching):
        if pointer and is_pinching:
            if self.start_btn.collidepoint(pointer):
                self.manager.change_scene("STORY")

    def draw(self, screen):
        screen.fill((30, 30, 50)) # Solid Dark Blue Background
        
        title = self.font.render("ROBOT REPAIR SQUAD", True, (255, 255, 0))
        screen.blit(title, (WINDOW_SIZE[0]//2 - title.get_width()//2, 150))
        
        # Draw Button
        color = (0, 200, 0)
        pygame.draw.rect(screen, color, self.start_btn, border_radius=15)
        pygame.draw.rect(screen, (255, 255, 255), self.start_btn, 3, border_radius=15)
        
        btn_text = self.subfont.render("PINCH START", True, (255, 255, 255))
        screen.blit(btn_text, (self.start_btn.centerx - btn_text.get_width()//2, 
                               self.start_btn.centery - btn_text.get_height()//2))

class StoryScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 40)
        self.timer = None
        
    def update(self, pointer, is_pinching):
        # Start timer on first update
        if self.timer is None: self.timer = time.time()
        
        if time.time() - self.timer > 3.0:
            self.manager.change_scene("GAME")

    def draw(self, screen):
        screen.fill((20, 20, 20)) # Dark Grey
        msg1 = self.font.render("OH NO!", True, (255, 100, 100))
        msg2 = self.font.render("The system is offline.", True, (255, 255, 255))
        msg3 = self.font.render("Put the Battery in the Slot!", True, (100, 255, 100))
        
        screen.blit(msg1, (100, 200))
        screen.blit(msg2, (100, 260))
        screen.blit(msg3, (100, 320))

class GameScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.slot = pygame.Rect(800, 300, 150, 100)
        self.battery = pygame.Rect(200, 300, 150, 100)
        self.dragging = False
        self.offset = (0, 0)
        self.font = pygame.font.SysFont("Arial", 30)
        
    def update(self, pointer, is_pinching):
        if not pointer: return

        if is_pinching:
            if not self.dragging:
                if self.battery.collidepoint(pointer):
                    self.dragging = True
                    self.offset = (pointer[0] - self.battery.x, pointer[1] - self.battery.y)
            else:
                self.battery.x = pointer[0] - self.offset[0]
                self.battery.y = pointer[1] - self.offset[1]
        else:
            if self.dragging:
                self.dragging = False
                dist = np.hypot(self.battery.centerx - self.slot.centerx, 
                                self.battery.centery - self.slot.centery)
                if dist < 60:
                    self.manager.change_scene("WIN")

    def draw(self, screen):
        # 1. Solid Background (No Webcam)
        screen.fill((50, 50, 70)) 
        
        # 2. Draw Floor/Table (Just for style)
        pygame.draw.rect(screen, (40, 40, 60), (0, 400, WINDOW_SIZE[0], 320))

        # 3. Draw Slot
        pygame.draw.rect(screen, (150, 150, 150), self.slot, 4)
        label = self.font.render("SLOT", True, (150, 150, 150))
        screen.blit(label, (self.slot.x, self.slot.y - 40))
        
        # 4. Draw Battery
        color = (0, 255, 0) if not self.dragging else (100, 255, 100)
        pygame.draw.rect(screen, color, self.battery, border_radius=10)
        pygame.draw.rect(screen, (255, 255, 255), self.battery, 3, border_radius=10)
        
        bat_label = self.font.render("BATTERY", True, (0, 100, 0))
        screen.blit(bat_label, (self.battery.x + 10, self.battery.y + 35))

class WinScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 60)
        self.reset_rect = pygame.Rect(0, 0, 150, 150) # Invisible trigger zone
        
    def update(self, pointer, is_pinching):
        if pointer and is_pinching:
            if self.reset_rect.collidepoint(pointer):
                self.manager.change_scene("TITLE")

    def draw(self, screen):
        screen.fill((50, 200, 50))
        msg = self.font.render("SYSTEM ONLINE!", True, (255, 255, 255))
        screen.blit(msg, (WINDOW_SIZE[0]//2 - msg.get_width()//2, 300))
        
        # Draw Reset Hint clearly
        pygame.draw.rect(screen, (255, 255, 255), (20, 20, 220, 50), border_radius=10)
        hint = pygame.font.SysFont("Arial", 25).render("<-- Pinch to Reset", True, (0, 0, 0))
        screen.blit(hint, (30, 30))

class GameManager:
    def __init__(self):
        self.scenes = {
            "TITLE": TitleScene(self),
            "STORY": StoryScene(self),
            "GAME": GameScene(self),
            "WIN": WinScene(self)
        }
        self.current_scene = self.scenes["TITLE"]
        
    def change_scene(self, scene_name):
        # Reset scene state if needed when entering
        if scene_name == "GAME":
            self.scenes["GAME"].battery.topleft = (200, 300)
        
        self.current_scene = self.scenes[scene_name]

# --- MAIN LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Visual Novel Demo")
    clock = pygame.time.Clock()
    
    # Init Vision
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_SIZE[0])
    cap.set(4, WINDOW_SIZE[1])
    
    controller = HandController()
    manager = GameManager()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q: running = False

        # Vision Input
        ok, frame = cap.read()
        if not ok: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, int(time.time()*1000))
        
        # --- LOGIC ---
        pointer = None
        is_pinching = False
        
        if result.hand_landmarks:
            pointer, is_pinching = controller.process(result.hand_landmarks[0], WINDOW_SIZE[0], WINDOW_SIZE[1])
        
        # --- DRAWING ---
        
        # 1. Draw Scene (Handles its own background fill now)
        manager.current_scene.update(pointer, is_pinching)
        manager.current_scene.draw(screen) 
        
        # 2. Draw GLOBAL CURSOR (On top of everything)
        if pointer:
            # Outer Ring (White)
            pygame.draw.circle(screen, (255, 255, 255), pointer, 15, 3)
            # Inner Dot (Color changes on Pinch)
            inner_color = (0, 255, 0) if is_pinching else (255, 255, 0) # Green=Grab, Yellow=Hover
            pygame.draw.circle(screen, inner_color, pointer, 8)
        else:
            # Helper text if hand lost
            warn = pygame.font.SysFont("Arial", 20).render("Show Hand", True, (255, 0, 0))
            screen.blit(warn, (10, WINDOW_SIZE[1] - 30))

        pygame.display.flip()
        clock.tick(30)
        
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()