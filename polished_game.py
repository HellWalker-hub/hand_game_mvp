import pygame
import cv2
import numpy as np
import time
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
WINDOW_SIZE = (1280, 720)
MODEL_PATH = "models/hand_landmarker.task"
# Files
BATTERY_IMG = "battery.png"
SLOT_BATTERY_IMG = "slot.png"
GEAR_IMG = "gear.png"
SLOT_GEAR_IMG = "gear_slot.png"
BG_IMG = "background.png" # The only new addition

# --- ASSET GENERATOR (Safety Check) ---
def generate_placeholder_assets():
    if not os.path.exists(BATTERY_IMG):
        surf = pygame.Surface((100, 150), pygame.SRCALPHA)
        pygame.draw.rect(surf, (50, 200, 50), (10, 20, 80, 120), border_radius=10)
        pygame.image.save(surf, BATTERY_IMG)
    if not os.path.exists(GEAR_IMG):
        surf = pygame.Surface((150, 150), pygame.SRCALPHA)
        pygame.draw.circle(surf, (100, 100, 255), (75, 75), 70)
        pygame.image.save(surf, GEAR_IMG)
    if not os.path.exists(SLOT_BATTERY_IMG):
        surf = pygame.Surface((120, 170), pygame.SRCALPHA)
        pygame.draw.rect(surf, (100, 100, 100), (0, 0, 120, 170), 5, border_radius=15)
        pygame.image.save(surf, SLOT_BATTERY_IMG)
    if not os.path.exists(SLOT_GEAR_IMG):
        surf = pygame.Surface((170, 170), pygame.SRCALPHA)
        pygame.draw.circle(surf, (200, 200, 200), (85, 85), 80, 5)
        pygame.image.save(surf, SLOT_GEAR_IMG)
    # Generate background if missing (Dark Grey Grid)
    if not os.path.exists(BG_IMG):
        surf = pygame.Surface(WINDOW_SIZE)
        surf.fill((40, 40, 50))
        for x in range(0, 1280, 50): pygame.draw.line(surf, (50, 50, 60), (x, 0), (x, 720), 1)
        for y in range(0, 720, 50): pygame.draw.line(surf, (50, 50, 60), (0, y), (1280, y), 1)
        pygame.image.save(surf, BG_IMG)

generate_placeholder_assets()

# --- SOUND MANAGER (Restored to WAV files) ---
class SoundManager:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sounds = {}
        self.load_assets()
        
    def load_assets(self):
        sound_dir = "SoundPack01"
        def load(name, filename):
            path = os.path.join(sound_dir, filename)
            if os.path.exists(path):
                self.sounds[name] = pygame.mixer.Sound(path)
                self.sounds[name].set_volume(0.5)
            else:
                print(f"Warning: Missing sound {filename}")

        load("GRAB", "Coin01.wav")
        load("RELEASE", "Downer01.wav")
        load("WIN", "Rise02.wav")

    def play(self, name):
        if name in self.sounds: self.sounds[name].play()

# --- HAND CONTROLLER (Restored to Robust Logic) ---
class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.grab_state = False 
        self.debounce_counter = 0
        self.DEBOUNCE_FRAMES = 3
        # Strict thresholds for stability
        self.GRAB_THRESHOLD = 0.40
        self.RELEASE_THRESHOLD = 0.60 

    def process(self, landmarks, w, h):
        wrist = landmarks[0]
        # Track Index Knuckle (Stable point)
        tracker_node = landmarks[5] 
        
        SMOOTH = 0.6
        raw_x, raw_y = int(tracker_node.x * w), int(tracker_node.y * h)
        if self.prev_x == 0: self.prev_x, self.prev_y = raw_x, raw_y
        curr_x = int(SMOOTH * raw_x + (1 - SMOOTH) * self.prev_x)
        curr_y = int(SMOOTH * raw_y + (1 - SMOOTH) * self.prev_y)
        self.prev_x, self.prev_y = curr_x, curr_y

        # Fist Logic (Rotation Independent)
        scale = np.hypot(landmarks[9].x - wrist.x, landmarks[9].y - wrist.y) + 1e-6
        pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
        total_curl = sum([np.hypot(landmarks[t].x - landmarks[m].x, landmarks[t].y - landmarks[m].y) for t, m in pairs])
        avg_curl = (total_curl / 4.0) / scale

        if avg_curl < self.GRAB_THRESHOLD:
             self.debounce_counter = min(self.DEBOUNCE_FRAMES, self.debounce_counter + 1)
        elif avg_curl > self.RELEASE_THRESHOLD:
             self.debounce_counter = max(0, self.debounce_counter - 1)
            
        new_state = (self.debounce_counter == self.DEBOUNCE_FRAMES)
        
        just_pressed = new_state and not self.grab_state
        just_released = not new_state and self.grab_state
        self.grab_state = new_state
        
        return (curr_x, curr_y), self.grab_state, just_pressed, just_released

# --- SPRITE CLASS ---
class DraggableSprite(pygame.sprite.Sprite):
    def __init__(self, img_path, center_pos, type_id, scale_size=150):
        super().__init__()
        self.image = self.load_and_scale(img_path, scale_size)
        self.rect = self.image.get_rect(center=center_pos)
        self.start_pos = center_pos
        self.type_id = type_id 
        self.is_dragging = False
        self.is_locked = False 
        self.offset = (0,0)

    def load_and_scale(self, path, max_size):
        if not os.path.exists(path):
            surf = pygame.Surface((max_size, max_size), pygame.SRCALPHA)
            color = (0, 255, 0) if "battery" in path else (0, 0, 255)
            pygame.draw.rect(surf, color, (0,0,max_size,max_size), border_radius=20)
            return surf
        img = pygame.image.load(path).convert_alpha()
        w, h = img.get_size()
        if w > max_size or h > max_size:
            scale = min(max_size/w, max_size/h)
            img = pygame.transform.smoothscale(img, (int(w*scale), int(h*scale)))
        return img

    def reset_pos(self):
        self.rect.center = self.start_pos

# --- SCENES ---
class Scene:
    def __init__(self, manager): self.manager = manager
    def update(self, pointer, is_grabbing, just_pressed, just_released): pass
    def draw(self, screen): pass

class GameScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 30)
        
        # 1. Load Background
        self.bg = pygame.image.load(BG_IMG).convert()
        self.bg = pygame.transform.scale(self.bg, WINDOW_SIZE)
        
        # 2. Setup Objects (Restored original positions)
        self.objects = [
            DraggableSprite(BATTERY_IMG, (300, 300), "BATTERY"),
            DraggableSprite(GEAR_IMG, (300, 500), "GEAR")
        ]
        
        # 3. Setup Slots
        self.slot_battery_img = self.objects[0].load_and_scale(SLOT_BATTERY_IMG, 170)
        self.slot_battery_rect = self.slot_battery_img.get_rect(center=(900, 300))
        
        self.slot_gear_img = self.objects[0].load_and_scale(SLOT_GEAR_IMG, 170)
        self.slot_gear_rect = self.slot_gear_img.get_rect(center=(900, 500))
        
        self.active_obj = None

    def update(self, pointer, is_grabbing, just_pressed, just_released):
        if not pointer: return

        # GRAB LOGIC
        if just_pressed and not self.active_obj:
            # Check collision with all UNLOCKED objects
            for obj in reversed(self.objects):
                if not obj.is_locked and obj.rect.collidepoint(pointer):
                    self.active_obj = obj
                    obj.is_dragging = True
                    obj.offset = (pointer[0] - obj.rect.x, pointer[1] - obj.rect.y)
                    self.manager.sound.play("GRAB")
                    # Bring to front
                    self.objects.remove(obj)
                    self.objects.append(obj)
                    break

        # RELEASE LOGIC
        if just_released and self.active_obj:
            obj = self.active_obj
            obj.is_dragging = False
            self.active_obj = None
            
            # Identify Target
            target_rect = None
            if obj.type_id == "BATTERY": target_rect = self.slot_battery_rect
            elif obj.type_id == "GEAR":  target_rect = self.slot_gear_rect
            
            # Distance Check
            dist = np.hypot(obj.rect.centerx - target_rect.centerx, 
                            obj.rect.centery - target_rect.centery)
            
            if dist < 80:
                # Success
                obj.rect.center = target_rect.center
                obj.is_locked = True
                self.manager.sound.play("WIN")
            else:
                # Fail - Snap Back (REJECT & RESET)
                self.manager.sound.play("RELEASE") # Or Error sound
                obj.reset_pos()
            
            # Check Win Condition
            if all(o.is_locked for o in self.objects):
                self.manager.change_scene("WIN")

        # DRAG MOVEMENT
        if self.active_obj and is_grabbing:
            self.active_obj.rect.x = pointer[0] - self.active_obj.offset[0]
            self.active_obj.rect.y = pointer[1] - self.active_obj.offset[1]

    def draw(self, screen):
        # Draw Background
        screen.blit(self.bg, (0,0))

        # Draw Slots
        screen.blit(self.slot_battery_img, self.slot_battery_rect)
        screen.blit(self.slot_gear_img, self.slot_gear_rect)
        
        # Draw Labels
        lbl1 = self.font.render("POWER", True, (200, 200, 200))
        screen.blit(lbl1, (self.slot_battery_rect.centerx-30, self.slot_battery_rect.y-30))
        lbl2 = self.font.render("ENGINE", True, (200, 200, 200))
        screen.blit(lbl2, (self.slot_gear_rect.centerx-30, self.slot_gear_rect.y-30))

        # Draw Objects
        for obj in self.objects:
            screen.blit(obj.image, obj.rect)
            if obj.is_dragging:
                # Simple highlight
                pygame.draw.rect(screen, (255, 255, 0), obj.rect, 3, border_radius=10)

class TitleScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 60)
        self.start_btn = pygame.Rect(WINDOW_SIZE[0]//2 - 100, WINDOW_SIZE[1]//2, 200, 80)
    def update(self, pointer, is_grabbing, just_pressed, just_released):
        if pointer and just_pressed and self.start_btn.collidepoint(pointer):
            self.manager.sound.play("GRAB")
            self.manager.change_scene("GAME")
    def draw(self, screen):
        screen.fill((30, 30, 50))
        title = self.font.render("ROBOT REPAIR", True, (255, 255, 0))
        screen.blit(title, (WINDOW_SIZE[0]//2 - title.get_width()//2, 150))
        pygame.draw.rect(screen, (0, 200, 0), self.start_btn, border_radius=15)
        msg = pygame.font.SysFont("Arial", 30).render("START", True, (255, 255, 255))
        screen.blit(msg, (self.start_btn.centerx - msg.get_width()//2, self.start_btn.centery - 15))

class WinScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 60)
        self.reset_rect = pygame.Rect(0, 0, 200, 200)
    def update(self, pointer, is_grabbing, just_pressed, just_released):
        if pointer and just_pressed and self.reset_rect.collidepoint(pointer):
            self.manager.change_scene("TITLE")
    def draw(self, screen):
        screen.fill((50, 200, 50))
        msg = self.font.render("ALL SYSTEMS GO!", True, (255, 255, 255))
        screen.blit(msg, (WINDOW_SIZE[0]//2 - msg.get_width()//2, 300))
        screen.blit(pygame.font.SysFont("Arial", 25).render("Reset", True, (0,0,0)), (30, 30))

class GameManager:
    def __init__(self):
        self.sound = SoundManager()
        self.scenes = { "TITLE": TitleScene(self), "GAME": GameScene(self), "WIN": WinScene(self) }
        self.current_scene = self.scenes["TITLE"]
    def change_scene(self, scene_name):
        if scene_name == "GAME": # Reset Level
            self.scenes["GAME"] = GameScene(self)
        self.current_scene = self.scenes[scene_name]

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Stable Sorting Game")
    clock = pygame.time.Clock()
    
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=MODEL_PATH), running_mode=vision.RunningMode.VIDEO, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_SIZE[0]); cap.set(4, WINDOW_SIZE[1])
    
    controller = HandController()
    manager = GameManager()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q): running = False
        
        ok, frame = cap.read()
        if not ok: break
        
        # Vision
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, int(time.time()*1000))
        
        pointer, is_grabbing, just_pressed, just_released = None, False, False, False
        if result.hand_landmarks:
            pointer, is_grabbing, just_pressed, just_released = controller.process(result.hand_landmarks[0], WINDOW_SIZE[0], WINDOW_SIZE[1])
        
        # Update Scene
        manager.current_scene.update(pointer, is_grabbing, just_pressed, just_released)
        manager.current_scene.draw(screen)
        
        # Cursor
        if pointer:
            color = (0, 255, 0) if is_grabbing else (255, 200, 0)
            pygame.draw.circle(screen, color, pointer, 15 if is_grabbing else 10)
            pygame.draw.circle(screen, (255, 255, 255), pointer, 19 if is_grabbing else 14, 2)
            lbl = pygame.font.SysFont("Arial", 18).render("FIST" if is_grabbing else "OPEN", True, color)
            screen.blit(lbl, (pointer[0]+20, pointer[1]-10))
        
        pygame.display.flip()
        clock.tick(30)
    cap.release(); pygame.quit()

if __name__ == "__main__":
    main()