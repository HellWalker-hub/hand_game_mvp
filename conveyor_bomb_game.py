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
# Files
BATTERY_IMG = "battery.png"
SLOT_BATTERY_IMG = "slot.png"
GEAR_IMG = "gear.png"
SLOT_GEAR_IMG = "gear_slot.png"
BOMB_IMG = "bomb.png"       # NEW
BG_IMG = "background.png"

# --- ASSETS (Auto-Generates Bomb) ---
def generate_placeholder_assets():
    # ... (Keep existing generators for Battery, Gear, Slots, BG) ...
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
    if not os.path.exists(BG_IMG):
        surf = pygame.Surface(WINDOW_SIZE)
        surf.fill((40, 40, 50))
        pygame.image.save(surf, BG_IMG)

    # NEW: Generate Bomb Asset
    if not os.path.exists(BOMB_IMG):
        surf = pygame.Surface((140, 150), pygame.SRCALPHA)
        # Black Ball
        pygame.draw.circle(surf, (20, 20, 20), (70, 80), 60)
        # Fuse
        pygame.draw.line(surf, (200, 150, 50), (70, 20), (70, 40), 5)
        # Shine
        pygame.draw.circle(surf, (255, 255, 255), (90, 60), 10)
        # Skull or X
        pygame.draw.line(surf, (255,0,0), (50, 80), (90, 80), 5)
        pygame.draw.line(surf, (255,0,0), (70, 60), (70, 100), 5)
        pygame.image.save(surf, BOMB_IMG)

generate_placeholder_assets()

# --- SMART LIGHTING FIX ---
def smart_adjust_gamma(image):
    """
    Calculates average brightness. 
    If > 90 (Bright enough), returns original.
    If < 90 (Dark), applies Gamma Boost.
    """
    # Convert to greyscale to check brightness quickly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    if avg_brightness < 90: # Threshold for "Dark Room"
        # Apply Night Vision
        gamma = 1.8
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table), True # True = Gamma Active
    
    return image, False # False = Normal Light

# --- PARTICLE SYSTEM (For Explosions) ---
class ParticleSystem:
    def __init__(self):
        self.particles = []
        
    def explode(self, x, y, color=(255, 100, 0)):
        for _ in range(20):
            vx = random.uniform(-10, 10)
            vy = random.uniform(-10, 10)
            size = random.randint(10, 25)
            self.particles.append([x, y, vx, vy, size, 255, color]) # x, y, vx, vy, size, alpha, color

    def update_and_draw(self, screen):
        for p in self.particles[:]:
            p[0] += p[2] # Move X
            p[1] += p[3] # Move Y
            p[5] -= 15   # Fade fast
            p[4] *= 0.9  # Shrink
            
            if p[5] <= 0:
                self.particles.remove(p)
            else:
                s = pygame.Surface((p[4]*2, p[4]*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p[6], int(p[5])), (p[4], p[4]), p[4])
                screen.blit(s, (p[0]-p[4], p[1]-p[4]))

# --- CLASSES (Sound, Hand, Sprite) ---
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
        load("EXPLODE", "Downer01.wav") # Reuse Downer for boom (or add new file)
        
        music_path = os.path.join(sound_dir, "ost.wav") 
        if os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
            self.music_loaded = True

    def play(self, name):
        if name in self.sounds: self.sounds[name].play()
    def play_music(self):
        if self.music_loaded: pygame.mixer.music.play(-1)

# HandController (With your settings + Sensitivity + Confidence)
class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.grab_state = False 
        self.debounce_counter = 0
        self.DEBOUNCE_FRAMES = 3
        self.GRAB_THRESHOLD = 0.40
        self.RELEASE_THRESHOLD = 0.60 
        self.CONFIDENCE_THRESHOLD = 0.5
        self.MARGIN = 0.25 

    def process(self, landmarks, handedness_score, w, h):
        if handedness_score < self.CONFIDENCE_THRESHOLD:
            return (self.prev_x, self.prev_y), self.grab_state, False, False, False

        wrist, tracker = landmarks[0], landmarks[5]
        
        # Sensitivity Logic
        clamped_x = max(self.MARGIN, min(1 - self.MARGIN, tracker.x))
        clamped_y = max(self.MARGIN, min(1 - self.MARGIN, tracker.y))
        active_width = 1 - (2 * self.MARGIN)
        normalized_x = (clamped_x - self.MARGIN) / active_width
        normalized_y = (clamped_y - self.MARGIN) / active_width
        target_x, target_y = int(normalized_x * w), int(normalized_y * h)

        SMOOTH = 0.6
        if self.prev_x == 0: self.prev_x, self.prev_y = target_x, target_y
        curr_x = int(SMOOTH * target_x + (1 - SMOOTH) * self.prev_x)
        curr_y = int(SMOOTH * target_y + (1 - SMOOTH) * self.prev_y)
        self.prev_x, self.prev_y = curr_x, curr_y

        scale = np.hypot(landmarks[9].x - wrist.x, landmarks[9].y - wrist.y) + 1e-6
        pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
        avg_curl = sum([np.hypot(landmarks[t].x - landmarks[m].x, landmarks[t].y - landmarks[m].y) for t, m in pairs]) / 4.0 / scale

        if avg_curl < self.GRAB_THRESHOLD: self.debounce_counter = min(self.DEBOUNCE_FRAMES, self.debounce_counter + 1)
        elif avg_curl > self.RELEASE_THRESHOLD: self.debounce_counter = max(0, self.debounce_counter - 1)
            
        new_state = (self.debounce_counter == self.DEBOUNCE_FRAMES)
        just_pressed = new_state and not self.grab_state
        just_released = not new_state and self.grab_state
        self.grab_state = new_state
        return (curr_x, curr_y), self.grab_state, just_pressed, just_released, True

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
        if not os.path.exists(path): return pygame.Surface((max_size, max_size))
        img = pygame.image.load(path).convert_alpha()
        w, h = img.get_size()
        if w > max_size or h > max_size:
            scale = min(max_size/w, max_size/h)
            img = pygame.transform.smoothscale(img, (int(w*scale), int(h*scale)))
        return img
    def update(self):
        if not self.is_dragging and not self.is_locked: self.rect.x += self.speed_x

# --- GAME SCENE (Logic Central) ---
class GameScene:
    def __init__(self, manager):
        self.manager = manager
        self.font = pygame.font.SysFont("Arial", 30)
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
        
        # Load Slot Images
        temp = DraggableSprite(SLOT_BATTERY_IMG, (0,0), "X")
        self.slot_bat_img = temp.load_and_scale(SLOT_BATTERY_IMG, 150)
        self.slot_bat_rect = self.slot_bat_img.get_rect(center=(400, 600))
        self.slot_gear_img = temp.load_and_scale(SLOT_GEAR_IMG, 150)
        self.slot_gear_rect = self.slot_gear_img.get_rect(center=(880, 600))
        
        self.manager.sound.play_music() # PLAY MUSIC HERE

    def spawn_object(self):
        r = random.random()
        # 20% Chance for BOMB
        if r < 0.2:
            type_id = "BOMB"
            img = BOMB_IMG
        elif r < 0.6:
            type_id = "BATTERY"
            img = BATTERY_IMG
        else:
            type_id = "GEAR"
            img = GEAR_IMG
            
        obj = DraggableSprite(img, (-60, self.conveyor_y + self.conveyor_height//2), type_id, self.game_speed)
        self.objects.append(obj)

    def update(self, pointer, is_grabbing, just_pressed, just_released, is_reliable):
        if not is_reliable and not self.active_obj: return

        # Spawn & Move
        if time.time() - self.spawn_timer > self.spawn_rate:
            self.spawn_object()
            self.spawn_timer = time.time()
            self.spawn_rate = max(1.2, self.spawn_rate * 0.98)
            self.game_speed = min(12, self.game_speed + 0.05)

        for obj in self.objects[:]:
            obj.update()
            # Off screen check
            if obj.rect.left > WINDOW_SIZE[0]:
                self.objects.remove(obj)
                # Only lose life if it was a GOOD item. Letting bomb pass is good!
                if obj.type_id != "BOMB":
                    self.lives -= 1
                    self.manager.sound.play("RELEASE")
                    if self.lives <= 0: self.manager.change_scene("LOSE")

        if not pointer: return

        # GRAB LOGIC
        if just_pressed and not self.active_obj:
            for obj in reversed(self.objects):
                # Use Inflated Hitbox
                if not obj.is_locked and obj.rect.inflate(60,60).collidepoint(pointer):
                    
                    # BOMB LOGIC: If you grab a bomb, it explodes immediately
                    if obj.type_id == "BOMB":
                        self.lives -= 1
                        self.manager.sound.play("EXPLODE")
                        self.particles.explode(obj.rect.centerx, obj.rect.centery, (255, 50, 0)) # Red explosion
                        self.objects.remove(obj)
                        if self.lives <= 0: self.manager.change_scene("LOSE")
                        return # Stop processing
                        
                    # NORMAL GRAB
                    self.active_obj = obj
                    obj.is_dragging = True
                    obj.offset = (pointer[0] - obj.rect.x, pointer[1] - obj.rect.y)
                    self.manager.sound.play("GRAB")
                    self.objects.remove(obj); self.objects.append(obj)
                    break

        # RELEASE LOGIC
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
                self.score += 10
                self.objects.remove(obj) # Auto-collect for points
            else:
                self.manager.sound.play("RELEASE")
                # Drop it? Let's just snap it back to conveyor (punishment is losing time)
                # Or just delete it (simple punishment)
                if obj in self.objects: self.objects.remove(obj)

        if self.active_obj and is_grabbing:
            self.active_obj.rect.x = pointer[0] - self.active_obj.offset[0]
            self.active_obj.rect.y = pointer[1] - self.active_obj.offset[1]

    def draw(self, screen):
        screen.blit(self.bg, (0,0))
        
        # Draw Conveyor
        belt_rect = pygame.Rect(0, self.conveyor_y, WINDOW_SIZE[0], self.conveyor_height)
        pygame.draw.rect(screen, (80, 80, 90), belt_rect)
        pygame.draw.rect(screen, (40, 40, 50), belt_rect, 5)
        self.belt_offset = (self.belt_offset + self.game_speed) % 100
        for x in range(int(-100 + self.belt_offset), WINDOW_SIZE[0], 100):
            p1 = (x, self.conveyor_y + 20); p2 = (x + 30, self.conveyor_y + self.conveyor_height//2); p3 = (x, self.conveyor_y + self.conveyor_height - 20)
            pygame.draw.lines(screen, (100, 100, 110), False, [p1, p2, p3], 5)

        # Slots & Objects
        screen.blit(self.slot_bat_img, self.slot_bat_rect)
        screen.blit(self.slot_gear_img, self.slot_gear_rect)
        for obj in self.objects:
            screen.blit(obj.image, obj.rect)
            if obj.is_dragging: pygame.draw.rect(screen, (255, 255, 0), obj.rect, 3, border_radius=10)

        # Particles
        self.particles.update_and_draw(screen)

        # HUD
        score_t = self.font.render(f"SCORE: {self.score}", True, (255, 255, 0))
        screen.blit(score_t, (20, 20))
        for i in range(self.lives):
            pygame.draw.circle(screen, (255, 0, 0), (WINDOW_SIZE[0] - 40 - (i*50), 40), 15)

# --- MANAGERS & MAIN (Simplified) ---
class TitleScene:
    def __init__(self, manager):
        self.manager = manager
        self.font = pygame.font.SysFont("Arial", 60)
        self.btn = pygame.Rect(540, 500, 200, 100)
    def update(self, p, g, jp, jr, rel):
        if p and jp and self.btn.collidepoint(p): self.manager.change_scene("GAME")
    def draw(self, s):
        s.fill((30,30,50)); pygame.draw.rect(s,(0,200,0),self.btn,border_radius=20)
        t = self.font.render("FACTORY FRENZY", True, (255,255,0)); s.blit(t, (640-t.get_width()//2, 200))
        st = pygame.font.SysFont("Arial", 40).render("START", True, (255,255,255)); s.blit(st, (self.btn.centerx-st.get_width()//2, self.btn.centery-st.get_height()//2))

class LoseScene:
    def __init__(self, manager):
        self.manager = manager
        self.font = pygame.font.SysFont("Arial", 80)
    def update(self, p, g, jp, jr, rel):
        if p and jp: self.manager.change_scene("TITLE")
    def draw(self, s):
        s.fill((50,0,0)); t = self.font.render("GAME OVER", True, (255,255,255)); s.blit(t, (640-t.get_width()//2, 300))

class GameManager:
    def __init__(self):
        self.sound = SoundManager()
        self.scenes = {"TITLE": TitleScene(self), "GAME": GameScene(self), "LOSE": LoseScene(self)}
        self.current_scene = self.scenes["TITLE"]
    def change_scene(self, name):
        if name == "GAME": self.scenes["GAME"] = GameScene(self)
        self.current_scene = self.scenes[name]

def main():
    pygame.init(); screen = pygame.display.set_mode(WINDOW_SIZE); pygame.display.set_caption("Conveyor + Bomb + SmartLight")
    clock = pygame.time.Clock()
    
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=MODEL_PATH), running_mode=vision.RunningMode.VIDEO, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    # Hardware Lock (Still useful to prevent blur)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); cap.set(cv2.CAP_PROP_EXPOSURE, -3.0) 
    cap.set(3, WINDOW_SIZE[0]); cap.set(4, WINDOW_SIZE[1])
    
    controller = HandController()
    manager = GameManager()
    show_camera = False 
    
    while True:
        for e in pygame.event.get(): 
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q): return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_d: show_camera = not show_camera
        
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        
        # --- ADAPTIVE LIGHTING ---
        frame, gamma_active = smart_adjust_gamma(frame)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_img, int(time.time()*1000))
        
        pointer, is_grabbing, jp, jr, is_reliable = None, False, False, False, True
        if result.hand_landmarks:
            score = result.handedness[0][0].score
            pointer, is_grabbing, jp, jr, is_reliable = controller.process(result.hand_landmarks[0], score, WINDOW_SIZE[0], WINDOW_SIZE[1])
        
        if show_camera:
            cam_surface = pygame.surfarray.make_surface(np.rot90(rgb_frame)); cam_surface = pygame.transform.flip(cam_surface, True, False); screen.blit(cam_surface, (0,0))
            status = "Night Vision: ON" if gamma_active else "Night Vision: OFF"
            lbl = pygame.font.SysFont("Arial", 30).render(status, True, (0, 255, 0) if gamma_active else (255, 255, 0)); screen.blit(lbl, (20, 60))
            if pointer: pygame.draw.circle(screen, (0, 255, 0), pointer, 10)
        else:
            manager.current_scene.update(pointer, is_grabbing, jp, jr, is_reliable)
            manager.current_scene.draw(screen)
            if pointer:
                col = (0, 255, 0) if is_grabbing else (255, 200, 0)
                if not is_reliable: col = (255, 0, 0)
                pygame.draw.circle(screen, col, pointer, 15 if is_grabbing else 10)
                pygame.draw.circle(screen, (255,255,255), pointer, 19 if is_grabbing else 14, 2)
        
        pygame.display.flip(); clock.tick(30)
    cap.release(); pygame.quit()

if __name__ == "__main__": main()