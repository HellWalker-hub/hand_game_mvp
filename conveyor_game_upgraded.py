import pygame
import cv2
import numpy as np
import time
import os
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
WINDOW_SIZE = (1280, 720)
MODEL_PATH = "models/hand_landmarker.task"

# DEV MODE & DISTANCE OPTIMIZATION
DEV_MODE = False  # Toggle for M1 Air testing
DIGITAL_ZOOM = 1.5  # Adjust for 2-3 meter distance
IGNORE_BACKGROUND_HANDS = True  # Ignore background interference

# Files - Placeholders that you can replace
BATTERY_IMG = "battery.png"
SLOT_BATTERY_IMG = "slot_battery.png"
GEAR_IMG = "gear_icon.png"
SLOT_GEAR_IMG = "gear_slot.png"
BOMB_IMG = "bomb.png"
BG_IMG = "background2.png"

# --- ENHANCED ASSETS (2D Design) ---
def generate_placeholder_assets():
    # Initialize pygame if needed
    if not pygame.get_init():
        pygame.init()
    #fsdf
    # Enhanced Battery (green rectangle with details)
    if not os.path.exists(BATTERY_IMG):
        surf = pygame.Surface((100, 150), pygame.SRCALPHA)
        # Body
        pygame.draw.rect(surf, (46, 213, 115), (10, 25, 80, 110), border_radius=10)
        # Left shadow
        pygame.draw.rect(surf, (35, 170, 90), (10, 25, 20, 110), border_radius=10)
        # Right highlight
        pygame.draw.rect(surf, (80, 255, 150), (70, 25, 20, 110), border_radius=10)
        # Terminal
        pygame.draw.rect(surf, (200, 200, 200), (35, 10, 30, 20), border_radius=5)
        # Plus sign
        font = pygame.font.SysFont("Arial", 40, bold=True)
        plus = font.render("+", True, (255, 255, 255))
        surf.blit(plus, (32, 55))
        pygame.image.save(surf, BATTERY_IMG)
    
    # Enhanced Gear (blue with teeth)
    if not os.path.exists(GEAR_IMG):
        surf = pygame.Surface((150, 150), pygame.SRCALPHA)
        center = (75, 75)
        # Outer teeth
        for i in range(8):
            angle = i * 45
            rad = np.radians(angle)
            x = center[0] + 60 * np.cos(rad)
            y = center[1] + 60 * np.sin(rad)
            pygame.draw.circle(surf, (0, 80, 160), (int(x), int(y)), 15)
        # Body
        pygame.draw.circle(surf, (0, 123, 255), center, 50)
        # Left shadow
        pygame.draw.circle(surf, (0, 90, 200), (center[0] - 10, center[1]), 45)
        # Right highlight
        pygame.draw.circle(surf, (50, 150, 255), (center[0] + 12, center[1] - 8), 25)
        # Center hole
        pygame.draw.circle(surf, (40, 40, 50), center, 20)
        pygame.image.save(surf, GEAR_IMG)
    
    # Enhanced Bomb (red/orange with spark)
    if not os.path.exists(BOMB_IMG):
        surf = pygame.Surface((140, 150), pygame.SRCALPHA)
        center = (70, 85)
        # Body with gradient effect
        for r in range(55, 0, -2):
            gray_val = 30 + (55 - r)
            pygame.draw.circle(surf, (gray_val, gray_val, gray_val), center, r)
        # Highlight
        pygame.draw.circle(surf, (100, 100, 100), (center[0] + 12, center[1] - 12), 18)
        # Fuse
        pygame.draw.line(surf, (100, 60, 30), (center[0] - 15, center[1] - 55), 
                        (center[0] - 8, center[1] - 45), 5)
        # Animated spark
        pygame.draw.circle(surf, (255, 200, 50), (center[0] - 15, center[1] - 55), 8)
        pygame.draw.circle(surf, (255, 100, 0), (center[0] - 15, center[1] - 55), 4)
        # Danger symbol
        font = pygame.font.SysFont("Arial", 32, bold=True)
        symbol = font.render("!", True, (255, 50, 50))
        surf.blit(symbol, (center[0] - 8, center[1] - 12))
        pygame.image.save(surf, BOMB_IMG)
    
    # Enhanced Battery Slot
    if not os.path.exists(SLOT_BATTERY_IMG):
        surf = pygame.Surface((120, 170), pygame.SRCALPHA)
        # Outer frame
        pygame.draw.rect(surf, (100, 110, 120), (0, 0, 120, 170), border_radius=15)
        # Glowing border
        pygame.draw.rect(surf, (46, 213, 115), (0, 0, 120, 170), 5, border_radius=15)
        # Inner shadow
        pygame.draw.rect(surf, (60, 65, 70), (10, 10, 100, 150), border_radius=12)
        pygame.image.save(surf, SLOT_BATTERY_IMG)
    
    # Enhanced Gear Slot
    if not os.path.exists(SLOT_GEAR_IMG):
        surf = pygame.Surface((170, 170), pygame.SRCALPHA)
        # Outer frame
        pygame.draw.rect(surf, (100, 110, 120), (0, 0, 170, 170), border_radius=15)
        # Glowing border
        pygame.draw.rect(surf, (0, 123, 255), (0, 0, 170, 170), 5, border_radius=15)
        # Inner shadow
        pygame.draw.circle(surf, (60, 65, 70), (85, 85), 70)
        pygame.image.save(surf, SLOT_GEAR_IMG)
    
    # Enhanced Background with grid
    if not os.path.exists(BG_IMG):
        surf = pygame.Surface(WINDOW_SIZE)
        # Gradient background
        for y in range(WINDOW_SIZE[1]):
            factor = y / WINDOW_SIZE[1]
            r = int(30 + 15 * factor)
            g = int(35 + 20 * factor)
            b = int(45 + 25 * factor)
            pygame.draw.line(surf, (r, g, b), (0, y), (WINDOW_SIZE[0], y))
        # Subtle grid
        for x in range(0, WINDOW_SIZE[0], 100):
            pygame.draw.line(surf, (50, 55, 70), (x, 0), (x, WINDOW_SIZE[1]), 1)
        for y in range(0, WINDOW_SIZE[1], 100):
            pygame.draw.line(surf, (50, 55, 70), (0, y), (WINDOW_SIZE[0], y), 1)
        pygame.image.save(surf, BG_IMG)

generate_placeholder_assets()

# --- SMART LIGHTING FIX ---
def smart_adjust_gamma(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness < 90:
        gamma = 1.8
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table), True
    return image, False

# --- DIGITAL ZOOM ---
def apply_digital_zoom(frame, zoom_factor):
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    cropped = frame[start_y:start_y + crop_h, start_x:start_x + crop_w]
    zoomed = cv2.resize(cropped, (w, h))
    return zoomed

# --- HAND SIZE CALCULATOR ---
def calculate_hand_size(landmarks):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width * height

# --- ENHANCED PARTICLE SYSTEM ---
class ParticleSystem:
    def __init__(self):
        self.particles = []
        
    def explode(self, x, y, color=(255, 100, 0), count=25):
        for _ in range(count):
            vx = random.uniform(-10, 10)
            vy = random.uniform(-12, 8)
            size = random.randint(8, 20)
            self.particles.append([x, y, vx, vy, size, 255, color])

    def update_and_draw(self, screen):
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.4  # Gravity
            p[5] -= 12
            p[4] *= 0.94
            if p[5] <= 0 or p[4] < 1:
                self.particles.remove(p)
            else:
                # Draw with subtle glow
                s = pygame.Surface((int(p[4]*3), int(p[4]*3)), pygame.SRCALPHA)
                # Outer glow
                pygame.draw.circle(s, (*p[6], int(p[5] * 0.3)), (int(p[4]*1.5), int(p[4]*1.5)), int(p[4]*1.5))
                # Core
                pygame.draw.circle(s, (*p[6], int(p[5])), (int(p[4]*1.5), int(p[4]*1.5)), int(p[4]))
                screen.blit(s, (int(p[0]-p[4]*1.5), int(p[1]-p[4]*1.5)))

# --- SOUND MANAGER ---
class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.music_loaded = False
        self.load_assets()
        
    def load_assets(self):
        sound_dir = "SoundPack01"
        def load(name, filename):
            path = os.path.join(sound_dir, filename)
            if os.path.exists(path):
                self.sounds[name] = pygame.mixer.Sound(path)
                self.sounds[name].set_volume(0.5)
        
        load("GRAB", "Coin01.wav")
        load("RELEASE", "Downer01.wav")
        load("WIN", "Rise02.wav")
        load("EXPLODE", "Downer01.wav")
        
        music_path = os.path.join(sound_dir, "ost.wav")
        if os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
            self.music_loaded = True

    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()
            
    def play_music(self):
        if self.music_loaded:
            pygame.mixer.music.play(-1)

# --- HAND CONTROLLER (UNCHANGED - Your smooth mechanics) ---
class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.grab_state = False
        self.debounce_counter = 0
        self.DEBOUNCE_FRAMES = 3
        self.GRAB_THRESHOLD = 0.40
        self.RELEASE_THRESHOLD = 0.60
        self.CONFIDENCE_THRESHOLD = 0.3  # Lower for distance
        self.MARGIN = 0.25

    def process(self, landmarks, handedness_score, w, h):
        if handedness_score < self.CONFIDENCE_THRESHOLD:
            return (self.prev_x, self.prev_y), self.grab_state, False, False, False

        wrist, tracker = landmarks[0], landmarks[5]
        
        clamped_x = max(self.MARGIN, min(1 - self.MARGIN, tracker.x))
        clamped_y = max(self.MARGIN, min(1 - self.MARGIN, tracker.y))
        active_width = 1 - (2 * self.MARGIN)
        normalized_x = (clamped_x - self.MARGIN) / active_width
        normalized_y = (clamped_y - self.MARGIN) / active_width
        target_x, target_y = int(normalized_x * w), int(normalized_y * h)

        SMOOTH = 0.6
        if self.prev_x == 0:
            self.prev_x, self.prev_y = target_x, target_y
        curr_x = int(SMOOTH * target_x + (1 - SMOOTH) * self.prev_x)
        curr_y = int(SMOOTH * target_y + (1 - SMOOTH) * self.prev_y)
        self.prev_x, self.prev_y = curr_x, curr_y

        scale = np.hypot(landmarks[9].x - wrist.x, landmarks[9].y - wrist.y) + 1e-6
        pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
        avg_curl = sum([np.hypot(landmarks[t].x - landmarks[m].x, landmarks[t].y - landmarks[m].y) for t, m in pairs]) / 4.0 / scale

        if avg_curl < self.GRAB_THRESHOLD:
            self.debounce_counter = min(self.DEBOUNCE_FRAMES, self.debounce_counter + 1)
        elif avg_curl > self.RELEASE_THRESHOLD:
            self.debounce_counter = max(0, self.debounce_counter - 1)
            
        new_state = (self.debounce_counter == self.DEBOUNCE_FRAMES)
        just_pressed = new_state and not self.grab_state
        just_released = not new_state and self.grab_state
        self.grab_state = new_state
        return (curr_x, curr_y), self.grab_state, just_pressed, just_released, True

# --- DRAGGABLE SPRITE ---
class DraggableSprite(pygame.sprite.Sprite):
    def __init__(self, img_path, center_pos, type_id, speed_x=0):
        super().__init__()
        self.image = self.load_and_scale(img_path, 130)
        self.rect = self.image.get_rect(center=center_pos)
        self.type_id = type_id
        self.speed_x = speed_x
        self.is_dragging = False
        self.is_locked = False
        self.offset = (0,0)

    def load_and_scale(self, path, max_size):
        if not os.path.exists(path):
            return pygame.Surface((max_size, max_size))
        img = pygame.image.load(path).convert_alpha()
        w, h = img.get_size()
        if w > max_size or h > max_size:
            scale = min(max_size/w, max_size/h)
            img = pygame.transform.smoothscale(img, (int(w*scale), int(h*scale)))
        return img
        
    def update(self):
        if not self.is_dragging and not self.is_locked:
            self.rect.x += self.speed_x

# --- GAME SCENE ---
class GameScene:
    def __init__(self, manager):
        self.manager = manager
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 32, bold=True)
        self.bg = pygame.image.load(BG_IMG).convert()
        self.bg = pygame.transform.scale(self.bg, WINDOW_SIZE)
        self.particles = ParticleSystem()
        self.conveyor_y = 200
        self.conveyor_height = 180
        self.belt_offset = 0
        self.spawn_timer = time.time()
        self.spawn_rate = 2.5
        self.game_speed = 3
        self.objects = []
        self.active_obj = None
        self.lives = 3
        self.score = 0
        
        temp = DraggableSprite(SLOT_BATTERY_IMG, (0,0), "X")
        self.slot_bat_img = temp.load_and_scale(SLOT_BATTERY_IMG, 150)
        self.slot_bat_rect = self.slot_bat_img.get_rect(center=(400, 600))
        self.slot_gear_img = temp.load_and_scale(SLOT_GEAR_IMG, 150)
        self.slot_gear_rect = self.slot_gear_img.get_rect(center=(880, 600))
        self.manager.sound.play_music()

    def spawn_object(self):
        r = random.random()
        if r < 0.2:
            type_id, img = "BOMB", BOMB_IMG
        elif r < 0.6:
            type_id, img = "BATTERY", BATTERY_IMG
        else:
            type_id, img = "GEAR", GEAR_IMG
        obj = DraggableSprite(img, (-60, self.conveyor_y + self.conveyor_height//2), type_id, self.game_speed)
        self.objects.append(obj)

    def update(self, pointer, is_grabbing, just_pressed, just_released, is_reliable):
        if not is_reliable and not self.active_obj:
            return

        if time.time() - self.spawn_timer > self.spawn_rate:
            self.spawn_object()
            self.spawn_timer = time.time()
            self.spawn_rate = max(1.2, self.spawn_rate * 0.98)
            self.game_speed = min(12, self.game_speed + 0.05)

        for obj in self.objects[:]:
            obj.update()
            if obj.rect.left > WINDOW_SIZE[0]:
                self.objects.remove(obj)
                if obj.type_id != "BOMB":
                    self.lives -= 1
                    self.manager.sound.play("RELEASE")
                    if self.lives <= 0:
                        self.manager.change_scene("LOSE")

        if not pointer:
            return

        if just_pressed and not self.active_obj:
            for obj in reversed(self.objects):
                if not obj.is_locked and obj.rect.inflate(60,60).collidepoint(pointer):
                    if obj.type_id == "BOMB":
                        self.lives -= 1
                        self.manager.sound.play("EXPLODE")
                        self.particles.explode(obj.rect.centerx, obj.rect.centery, (255, 80, 50), 40)
                        self.objects.remove(obj)
                        if self.lives <= 0:
                            self.manager.change_scene("LOSE")
                        return
                    self.active_obj = obj
                    obj.is_dragging = True
                    obj.offset = (pointer[0] - obj.rect.x, pointer[1] - obj.rect.y)
                    self.manager.sound.play("GRAB")
                    self.objects.remove(obj)
                    self.objects.append(obj)
                    break

        if just_released and self.active_obj:
            obj = self.active_obj
            obj.is_dragging = False
            self.active_obj = None
            target_rect = self.slot_bat_rect if obj.type_id == "BATTERY" else self.slot_gear_rect
            dist = np.hypot(obj.rect.centerx - target_rect.centerx, obj.rect.centery - target_rect.centery)
            if dist < 80:
                obj.is_locked = True
                obj.rect.center = target_rect.center
                self.manager.sound.play("WIN")
                # Success particles
                color = (46, 213, 115) if obj.type_id == "BATTERY" else (0, 123, 255)
                self.particles.explode(target_rect.centerx, target_rect.centery, color, 30)
                self.score += 10
                self.objects.remove(obj)
            else:
                self.manager.sound.play("RELEASE")
                if obj in self.objects:
                    self.objects.remove(obj)

        if self.active_obj and is_grabbing:
            self.active_obj.rect.x = pointer[0] - self.active_obj.offset[0]
            self.active_obj.rect.y = pointer[1] - self.active_obj.offset[1]

    def draw(self, screen):
        screen.blit(self.bg, (0,0))
        
        # Enhanced conveyor belt
        belt_rect = pygame.Rect(0, self.conveyor_y, WINDOW_SIZE[0], self.conveyor_height)
        pygame.draw.rect(screen, (90, 95, 105), belt_rect)
        pygame.draw.rect(screen, (50, 55, 65), belt_rect, 6)
        
        # Smoother belt animation
        self.belt_offset = (self.belt_offset + self.game_speed) % 100
        for x in range(int(-100 + self.belt_offset), WINDOW_SIZE[0], 100):
            p1 = (x, self.conveyor_y + 20)
            p2 = (x + 30, self.conveyor_y + self.conveyor_height//2)
            p3 = (x, self.conveyor_y + self.conveyor_height - 20)
            pygame.draw.lines(screen, (110, 115, 125), False, [p1, p2, p3], 4)
        
        # Draw slots with glow
        for slot_img, slot_rect in [(self.slot_bat_img, self.slot_bat_rect), 
                                     (self.slot_gear_img, self.slot_gear_rect)]:
            # Subtle glow
            glow_surf = pygame.Surface((slot_rect.width + 30, slot_rect.height + 30), pygame.SRCALPHA)
            glow_color = (46, 213, 115) if slot_img == self.slot_bat_img else (0, 123, 255)
            pygame.draw.rect(glow_surf, (*glow_color, 40), 
                           (0, 0, slot_rect.width + 30, slot_rect.height + 30), border_radius=18)
            screen.blit(glow_surf, (slot_rect.x - 15, slot_rect.y - 15))
            screen.blit(slot_img, slot_rect)
        
        # Draw objects
        for obj in self.objects:
            screen.blit(obj.image, obj.rect)
            if obj.is_dragging:
                # Glowing outline when dragging
                glow_surf = pygame.Surface((obj.rect.width + 20, obj.rect.height + 20), pygame.SRCALPHA)
                pygame.draw.rect(glow_surf, (255, 255, 100, 120), 
                               (0, 0, obj.rect.width + 20, obj.rect.height + 20), border_radius=12)
                screen.blit(glow_surf, (obj.rect.x - 10, obj.rect.y - 10))
                pygame.draw.rect(screen, (255, 255, 100), obj.rect, 3, border_radius=10)
        
        self.particles.update_and_draw(screen)
        
        # Enhanced HUD
        score_text = self.font_large.render(f"SCORE: {self.score}", True, (255, 255, 100))
        score_shadow = self.font_large.render(f"SCORE: {self.score}", True, (80, 80, 40))
        screen.blit(score_shadow, (22, 22))
        screen.blit(score_text, (20, 20))
        
        # Lives with glow
        for i in range(self.lives):
            x_pos = WINDOW_SIZE[0] - 50 - (i * 55)
            # Glow
            glow_surf = pygame.Surface((50, 50), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (255, 80, 80, 60), (25, 25), 22)
            screen.blit(glow_surf, (x_pos - 25, 15))
            # Heart
            pygame.draw.circle(screen, (255, 70, 70), (x_pos, 40), 16)
            pygame.draw.circle(screen, (200, 50, 50), (x_pos - 3, 40), 13)

# --- SCENES ---
class TitleScene:
    def __init__(self, manager):
        self.manager = manager
        self.font_title = pygame.font.SysFont("Arial", 70, bold=True)
        self.font_button = pygame.font.SysFont("Arial", 42, bold=True)
        self.btn = pygame.Rect(490, 480, 300, 100)
        self.pulse = 0
        
    def update(self, p, g, jp, jr, rel):
        self.pulse = (self.pulse + 0.08) % (2 * np.pi)
        if p and jp and self.btn.collidepoint(p):
            self.manager.change_scene("GAME")
            
    def draw(self, s):
        bg = pygame.image.load(BG_IMG).convert()
        bg = pygame.transform.scale(bg, WINDOW_SIZE)
        s.blit(bg, (0, 0))
        
        # Pulsing title
        scale = 1.0 + 0.05 * np.sin(self.pulse)
        title = self.font_title.render("FACTORY FRENZY", True, (255, 255, 100))
        title_shadow = self.font_title.render("FACTORY FRENZY", True, (80, 80, 40))
        scaled = pygame.transform.scale(title, (int(title.get_width() * scale), int(title.get_height() * scale)))
        scaled_shadow = pygame.transform.scale(title_shadow, (int(title_shadow.get_width() * scale), int(title_shadow.get_height() * scale)))
        s.blit(scaled_shadow, (642 - scaled_shadow.get_width() // 2, 182))
        s.blit(scaled, (640 - scaled.get_width() // 2, 180))
        
        # Glowing button
        glow_surf = pygame.Surface((340, 140), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (46, 213, 115, 60), (0, 0, 340, 140), border_radius=25)
        s.blit(glow_surf, (self.btn.x - 20, self.btn.y - 20))
        
        pygame.draw.rect(s, (35, 170, 90), self.btn, border_radius=20)
        pygame.draw.rect(s, (46, 213, 115), self.btn, 5, border_radius=20)
        
        st = self.font_button.render("START", True, (255, 255, 255))
        s.blit(st, (self.btn.centerx - st.get_width() // 2, self.btn.centery - st.get_height() // 2))

class LoseScene:
    def __init__(self, manager):
        self.manager = manager
        self.font_large = pygame.font.SysFont("Arial", 80, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 40)
        
    def update(self, p, g, jp, jr, rel):
        if p and jp:
            self.manager.change_scene("TITLE")
            
    def draw(self, s):
        bg = pygame.image.load(BG_IMG).convert()
        bg = pygame.transform.scale(bg, WINDOW_SIZE)
        s.blit(bg, (0, 0))
        
        # Dark overlay
        overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        s.blit(overlay, (0, 0))
        
        title = self.font_large.render("GAME OVER", True, (255, 80, 80))
        title_shadow = self.font_large.render("GAME OVER", True, (80, 30, 30))
        s.blit(title_shadow, (642 - title_shadow.get_width() // 2, 252))
        s.blit(title, (640 - title.get_width() // 2, 250))
        
        instruction = self.font_medium.render("Click anywhere to restart", True, (200, 200, 200))
        s.blit(instruction, (640 - instruction.get_width() // 2, 380))

class GameManager:
    def __init__(self):
        self.sound = SoundManager()
        self.scenes = {"TITLE": TitleScene(self), "GAME": GameScene(self), "LOSE": LoseScene(self)}
        self.current_scene = self.scenes["TITLE"]
        
    def change_scene(self, name):
        if name == "GAME":
            self.scenes["GAME"] = GameScene(self)
        self.current_scene = self.scenes[name]

# --- MAIN ---
def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Factory Frenzy - Enhanced")
    clock = pygame.time.Clock()
    
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,  # Detect multiple to pick largest
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    # Camera optimization for Elgato Facecam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # Faster shutter
    
    # Processing resolution
    if DEV_MODE:
        PROCESSING_WIDTH = 480
        PROCESSING_HEIGHT = 360
    else:
        PROCESSING_WIDTH = 640
        PROCESSING_HEIGHT = 480
    
    controller = HandController()
    manager = GameManager()
    show_camera = False
    result = None
    
    print(f"=== FACTORY FRENZY - ENHANCED ===")
    print(f"Camera Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"Digital Zoom: {DIGITAL_ZOOM}x")
    print(f"Background Isolation: {'ON' if IGNORE_BACKGROUND_HANDS else 'OFF'}")
    print(f"==================================")
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
                cap.release()
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_d:
                show_camera = not show_camera
        
        ok, frame = cap.read()
        if not ok:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Apply digital zoom
        if DIGITAL_ZOOM > 1.0:
            frame = apply_digital_zoom(frame, DIGITAL_ZOOM)
        
        # Downscale for processing
        small_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
        
        # Skip gamma in DEV_MODE for speed
        if DEV_MODE:
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            gamma_active = False
        else:
            small_frame, gamma_active = smart_adjust_gamma(small_frame)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = pygame.time.get_ticks()
        result = landmarker.detect_for_video(mp_img, timestamp_ms)
        
        # Hand isolation
        selected_hand = None
        selected_score = 0
        
        if result and result.hand_landmarks and IGNORE_BACKGROUND_HANDS:
            if len(result.hand_landmarks) == 1:
                selected_hand = result.hand_landmarks[0]
                selected_score = result.handedness[0][0].score
            else:
                largest_size = 0
                for i, hand in enumerate(result.hand_landmarks):
                    hand_size = calculate_hand_size(hand)
                    if hand_size > largest_size:
                        largest_size = hand_size
                        selected_hand = hand
                        selected_score = result.handedness[i][0].score
        elif result and result.hand_landmarks:
            selected_hand = result.hand_landmarks[0]
            selected_score = result.handedness[0][0].score
        
        pointer, is_grabbing, jp, jr, is_reliable = None, False, False, False, True
        if selected_hand:
            pointer, is_grabbing, jp, jr, is_reliable = controller.process(
                selected_hand, selected_score, WINDOW_SIZE[0], WINDOW_SIZE[1]
            )
        
        if show_camera:
            cam_surface = pygame.surfarray.make_surface(np.rot90(rgb_frame))
            cam_surface = pygame.transform.flip(cam_surface, True, False)
            cam_surface = pygame.transform.scale(cam_surface, WINDOW_SIZE)
            screen.blit(cam_surface, (0, 0))
            
            debug_font = pygame.font.SysFont("Arial", 24)
            zoom_text = debug_font.render(f"Zoom: {DIGITAL_ZOOM}x", True, (0, 255, 255))
            screen.blit(zoom_text, (20, 60))
            
            if pointer:
                pygame.draw.circle(screen, (0, 255, 0), pointer, 10)
        else:
            manager.current_scene.update(pointer, is_grabbing, jp, jr, is_reliable)
            manager.current_scene.draw(screen)
            
            # Enhanced cursor with glow
            if pointer:
                # Determine color
                if not is_reliable:
                    col = (255, 100, 100)
                elif is_grabbing:
                    col = (100, 255, 100)
                else:
                    col = (255, 200, 0)
                
                # Outer glow
                glow_size = 25 if is_grabbing else 18
                glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*col, 80), (glow_size, glow_size), glow_size)
                screen.blit(glow_surf, (pointer[0] - glow_size, pointer[1] - glow_size))
                
                # Core cursor
                pygame.draw.circle(screen, col, pointer, 15 if is_grabbing else 10)
                pygame.draw.circle(screen, (255, 255, 255), pointer, 19 if is_grabbing else 14, 2)
        
        pygame.display.flip()
        clock.tick(30)
    
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()