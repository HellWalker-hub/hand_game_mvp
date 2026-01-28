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
BG_IMG = "background.png"

# --- ASSETS & SOUNDS ---
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
    if not os.path.exists(BG_IMG):
        surf = pygame.Surface(WINDOW_SIZE)
        surf.fill((40, 40, 50))
        pygame.image.save(surf, BG_IMG)
generate_placeholder_assets()

class SoundManager:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sounds = {}
        self.music_loaded = False # Flag to check if music exists
        self.load_assets()

    def load_assets(self):
        sound_dir = "SoundPack01"
        
        # 1. Load Sound Effects (Short clips)
        def load(name, filename):
            path = os.path.join(sound_dir, filename)
            if os.path.exists(path): 
                self.sounds[name] = pygame.mixer.Sound(path)
                self.sounds[name].set_volume(0.5) # Set SFX volume to 50%
            else:
                print(f"Warning: Sound {filename} not found.")
                
        load("GRAB", "Coin01.wav")
        load("RELEASE", "Downer01.wav")
        load("WIN", "Rise02.wav")

        # 2. Load Background Music (Streamed)
        # Make sure you have a file named 'ost.wav' or 'ost.mp3' in SoundPack01
        music_path = os.path.join(sound_dir, "ost.wav") 
        if os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
            self.music_loaded = True
        else:
            print(f"Warning: Music file {music_path} not found.")

    def play(self, name):
        """Plays a short sound effect."""
        if name in self.sounds: 
            self.sounds[name].play()

    def play_music(self):
        """Plays the background music on a loop."""
        if self.music_loaded:
            # -1 argument means loop indefinitely
            pygame.mixer.music.play(-1)
            # Set music volume lower (e.g., 0.3) so it doesn't drown out sound effects
            pygame.mixer.music.set_volume(0.3)
# --- HAND CONTROLLER (With Confidence Gating) ---
# --- HAND CONTROLLER (With Sensitivity Upgrade) ---
class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.grab_state = False 
        self.debounce_counter = 0
        self.DEBOUNCE_FRAMES = 3
        self.GRAB_THRESHOLD = 0.40
        self.RELEASE_THRESHOLD = 0.60 
        self.CONFIDENCE_THRESHOLD = 0.5
        
        # SENSITIVITY PARAMETER (NEW)
        # 0.2 means we chop off 20% from each side.
        # Higher number = Faster Cursor (Less movement needed)
        # Try 0.2 or 0.25. Do not go above 0.4.
        self.MARGIN = 0.20 

    def process(self, landmarks, handedness_score, w, h):
        if handedness_score < self.CONFIDENCE_THRESHOLD:
            return (self.prev_x, self.prev_y), self.grab_state, False, False, False

        wrist = landmarks[0]
        tracker_node = landmarks[5] 
        
        # --- 1. SENSITIVITY MAPPING ---
        # Remap the normalized x (0.0 to 1.0) to a smaller range (margin to 1.0-margin)
        
        # Clamp input so it doesn't go below margin or above 1-margin
        clamped_x = max(self.MARGIN, min(1 - self.MARGIN, tracker_node.x))
        clamped_y = max(self.MARGIN, min(1 - self.MARGIN, tracker_node.y))
        
        # Remap to 0.0-1.0 range
        # Formula: (val - min) / (max - min)
        active_width = 1 - (2 * self.MARGIN)
        normalized_x = (clamped_x - self.MARGIN) / active_width
        normalized_y = (clamped_y - self.MARGIN) / active_width
        
        # Convert to pixels
        target_x = int(normalized_x * w)
        target_y = int(normalized_y * h)

        # --- 2. SMOOTHING ---
        SMOOTH = 0.6
        if self.prev_x == 0: self.prev_x, self.prev_y = target_x, target_y
        
        curr_x = int(SMOOTH * target_x + (1 - SMOOTH) * self.prev_x)
        curr_y = int(SMOOTH * target_y + (1 - SMOOTH) * self.prev_y)
        self.prev_x, self.prev_y = curr_x, curr_y

        # --- 3. FIST LOGIC (Unchanged) ---
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
# --- CONVEYOR OBJECTS ---
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
        if not self.is_dragging and not self.is_locked:
            self.rect.x += self.speed_x

# --- SCENES ---
class Scene:
    def __init__(self, manager): self.manager = manager
    def update(self, pointer, is_grabbing, just_pressed, just_released, is_reliable): pass
    def draw(self, screen): pass


class GameScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 30)
        self.bg = pygame.image.load(BG_IMG).convert()
        self.bg = pygame.transform.scale(self.bg, WINDOW_SIZE)
        
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
        if random.random() < 0.5:
            type_id = "BATTERY"
            img = BATTERY_IMG
        else:
            type_id = "GEAR"
            img = GEAR_IMG
        obj = DraggableSprite(img, (-50, self.conveyor_y + self.conveyor_height//2), type_id, self.game_speed)
        self.objects.append(obj)

    def update(self, pointer, is_grabbing, just_pressed, just_released, is_reliable):
        # Only update gameplay if reliable, OR if already dragging (don't drop accidentally)
        if not is_reliable and not self.active_obj:
            return 

        if time.time() - self.spawn_timer > self.spawn_rate:
            self.spawn_object()
            self.spawn_timer = time.time()
            self.spawn_rate = max(1.0, self.spawn_rate * 0.98)
            self.game_speed = min(10, self.game_speed + 0.1)

        for obj in self.objects[:]:
            obj.update()
            if obj.rect.left > WINDOW_SIZE[0]:
                self.objects.remove(obj)
                self.lives -= 1
                self.manager.sound.play("RELEASE") 
                if self.lives <= 0: self.manager.change_scene("LOSE")

        if not pointer: return

        if just_pressed and not self.active_obj:
            for obj in reversed(self.objects):
                if not obj.is_locked and obj.rect.inflate(60, 60).collidepoint(pointer):
                    self.active_obj = obj
                    obj.is_dragging = True
                    obj.offset = (pointer[0] - obj.rect.x, pointer[1] - obj.rect.y)
                    self.manager.sound.play("GRAB")
                    self.objects.remove(obj); self.objects.append(obj) 
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
                self.score += 10
                self.objects.remove(obj)
            else:
                self.manager.sound.play("RELEASE")
                if obj in self.objects: self.objects.remove(obj)

        if self.active_obj and is_grabbing:
            self.active_obj.rect.x = pointer[0] - self.active_obj.offset[0]
            self.active_obj.rect.y = pointer[1] - self.active_obj.offset[1]

    def draw(self, screen):
        screen.blit(self.bg, (0,0))
        belt_rect = pygame.Rect(0, self.conveyor_y, WINDOW_SIZE[0], self.conveyor_height)
        pygame.draw.rect(screen, (80, 80, 90), belt_rect)
        pygame.draw.rect(screen, (40, 40, 50), belt_rect, 5) 
        
        self.belt_offset = (self.belt_offset + self.game_speed) % 100
        for x in range(int(-100 + self.belt_offset), WINDOW_SIZE[0], 100):
            p1 = (x, self.conveyor_y + 20)
            p2 = (x + 30, self.conveyor_y + self.conveyor_height//2)
            p3 = (x, self.conveyor_y + self.conveyor_height - 20)
            pygame.draw.lines(screen, (100, 100, 110), False, [p1, p2, p3], 5)

        screen.blit(self.slot_bat_img, self.slot_bat_rect)
        screen.blit(self.slot_gear_img, self.slot_gear_rect)
        
        sfont = pygame.font.SysFont("Arial", 40, bold=True)
        screen.blit(sfont.render("BATTERIES", True, (200,200,200)), (self.slot_bat_rect.centerx-100, 700))
        screen.blit(sfont.render("GEARS", True, (200,200,200)), (self.slot_gear_rect.centerx-60, 700))

        for obj in self.objects:
            screen.blit(obj.image, obj.rect)
            if obj.is_dragging:
                pygame.draw.rect(screen, (255, 255, 0), obj.rect, 3, border_radius=10)

        score_t = self.font.render(f"SCORE: {self.score}", True, (255, 255, 0))
        screen.blit(score_t, (20, 20))
        for i in range(self.lives):
            pygame.draw.circle(screen, (255, 0, 0), (WINDOW_SIZE[0] - 40 - (i*50), 40), 15)

class LoseScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 80)
        self.subfont = pygame.font.SysFont("Arial", 40)
        self.retry_rect = pygame.Rect(0,0,1280,720) 
    def update(self, pointer, is_grabbing, just_pressed, just_released, is_reliable):
        if pointer and just_pressed: self.manager.change_scene("TITLE")
    def draw(self, screen):
        screen.fill((50, 0, 0))
        t = self.font.render("GAME OVER", True, (255, 255, 255))
        screen.blit(t, (640 - t.get_width()//2, 300))
        s = self.subfont.render("Grab to Retry", True, (200, 200, 200))
        screen.blit(s, (640 - s.get_width()//2, 400))

class TitleScene(Scene):
    def __init__(self, manager):
        super().__init__(manager)
        self.font = pygame.font.SysFont("Arial", 60)
        self.btn = pygame.Rect(540, 500, 200, 100)
    def update(self, pointer, is_grabbing, just_pressed, just_released, is_reliable):
        if pointer and just_pressed and self.btn.collidepoint(pointer):
            self.manager.sound.play("GRAB")
            self.manager.change_scene("GAME")
    def draw(self, screen):
        screen.fill((30, 30, 50))
        t = self.font.render("FACTORY FRENZY", True, (255, 255, 0))
        screen.blit(t, (640 - t.get_width()//2, 200))
        pygame.draw.rect(screen, (0, 200, 0), self.btn, border_radius=20)
        start = pygame.font.SysFont("Arial", 40).render("START", True, (255,255,255))
        screen.blit(start, (self.btn.centerx - start.get_width()//2, self.btn.centery - start.get_height()//2))

class GameManager:
    def __init__(self):
        self.sound = SoundManager()
        self.scenes = { "TITLE": TitleScene(self), "GAME": GameScene(self), "LOSE": LoseScene(self) }
        self.current_scene = self.scenes["TITLE"]
    def change_scene(self, name):
        if name == "GAME": self.scenes["GAME"] = GameScene(self) 
        self.current_scene = self.scenes[name]


def adjust_gamma(image, gamma=1.5):
    """
    Builds a lookup table mapping the pixel values [0, 255] to
    their adjusted gamma values. This effectively brightens shadows.
    gamma > 1.0 = Brighter shadows (Night Vision)
    gamma < 1.0 = Darker image
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply the mapping (Very fast, negligible lag)
    return cv2.LUT(image, table)
def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Conveyor + Night Vision")
    clock = pygame.time.Clock()
    
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=MODEL_PATH), running_mode=vision.RunningMode.VIDEO, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(1) 
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    # 1. HARDWARE FIX: Relax exposure slightly for dark rooms
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    cap.set(cv2.CAP_PROP_EXPOSURE, -3.0) # CHANGED from -5.0 to -3.0
    
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
        
        # 2. SOFTWARE FIX: Apply Night Vision
        # Gamma 1.5 - 2.0 is usually the sweet spot for dark rooms
        frame = adjust_gamma(frame, gamma=1.8) 
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_img, int(time.time()*1000))
        
        pointer, is_grabbing, jp, jr, is_reliable = None, False, False, False, True
        
        if result.hand_landmarks:
            score = result.handedness[0][0].score
            pointer, is_grabbing, jp, jr, is_reliable = controller.process(result.hand_landmarks[0], score, WINDOW_SIZE[0], WINDOW_SIZE[1])
        
        if show_camera:
            # Show the BRIGHTENED frame so you can see what the AI sees
            cam_surface = pygame.surfarray.make_surface(np.rot90(rgb_frame))
            cam_surface = pygame.transform.flip(cam_surface, True, False)
            screen.blit(cam_surface, (0,0))
            
            # Show Gamma Value
            lbl = pygame.font.SysFont("Arial", 30).render("Night Vision: ON (Gamma 1.8)", True, (0, 255, 0))
            screen.blit(lbl, (20, 60))
            
            if pointer: pygame.draw.circle(screen, (0, 255, 0), pointer, 10)
        else:
            manager.current_scene.update(pointer, is_grabbing, jp, jr, is_reliable)
            manager.current_scene.draw(screen)
            
            # (Draw cursor logic same as before...)
            if pointer:
                col = (0, 255, 0) if is_grabbing else (255, 200, 0)
                pygame.draw.circle(screen, col, pointer, 15 if is_grabbing else 10)
                pygame.draw.circle(screen, (255,255,255), pointer, 19 if is_grabbing else 14, 2)

        pygame.display.flip()
        clock.tick(30)
    cap.release(); pygame.quit()

if __name__ == "__main__": main()