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
DEV_MODE = True  # Toggle for M1 Air testing
DIGITAL_ZOOM = 1.5  # Adjust for 2-3 meter distance
IGNORE_BACKGROUND_HANDS = True  # Ignore background interference

# Files
BATTERY_IMG = "battery.png"
SLOT_BATTERY_IMG = "slot.png"
GEAR_IMG = "gear.png"
SLOT_GEAR_IMG = "gear_slot.png"
BOMB_IMG = "bomb.png"
BG_IMG = "background.png"

# --- ASSETS ---
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
    if not os.path.exists(BOMB_IMG):
        surf = pygame.Surface((140, 150), pygame.SRCALPHA)
        pygame.draw.circle(surf, (20, 20, 20), (70, 80), 60)
        pygame.draw.line(surf, (200, 150, 50), (70, 20), (70, 40), 5)
        pygame.draw.circle(surf, (255, 255, 255), (90, 60), 10)
        pygame.draw.line(surf, (255,0,0), (50, 80), (90, 80), 5)
        pygame.draw.line(surf, (255,0,0), (70, 60), (70, 100), 5)
        pygame.image.save(surf, BOMB_IMG)

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

# --- PARTICLE SYSTEM ---
class ParticleSystem:
    def __init__(self):
        self.particles = []
        
    def explode(self, x, y, color=(255, 100, 0)):
        for _ in range(20):
            vx = random.uniform(-10, 10)
            vy = random.uniform(-10, 10)
            size = random.randint(10, 25)
            self.particles.append([x, y, vx, vy, size, 255, color])

    def update_and_draw(self, screen):
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[5] -= 15
            p[4] *= 0.9
            if p[5] <= 0:
                self.particles.remove(p)
            else:
                s = pygame.Surface((int(p[4]*2), int(p[4]*2)), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p[6], int(p[5])), (int(p[4]), int(p[4])), int(p[4]))
                screen.blit(s, (int(p[0]-p[4]), int(p[1]-p[4])))

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

# --- HAND CONTROLLER ---
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
                        self.particles.explode(obj.rect.centerx, obj.rect.centery, (255, 50, 0))
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
        for obj in self.objects:
            screen.blit(obj.image, obj.rect)
            if obj.is_dragging:
                pygame.draw.rect(screen, (255, 255, 0), obj.rect, 3, border_radius=10)
        self.particles.update_and_draw(screen)
        score_t = self.font.render(f"SCORE: {self.score}", True, (255, 255, 0))
        screen.blit(score_t, (20, 20))
        for i in range(self.lives):
            pygame.draw.circle(screen, (255, 0, 0), (WINDOW_SIZE[0] - 40 - (i*50), 40), 15)

# --- SCENES ---
class TitleScene:
    def __init__(self, manager):
        self.manager = manager
        self.font = pygame.font.SysFont("Arial", 60)
        self.btn = pygame.Rect(540, 500, 200, 100)
        
    def update(self, p, g, jp, jr, rel):
        if p and jp and self.btn.collidepoint(p):
            self.manager.change_scene("GAME")
            
    def draw(self, s):
        s.fill((30,30,50))
        pygame.draw.rect(s,(0,200,0),self.btn,border_radius=20)
        t = self.font.render("FACTORY FRENZY", True, (255,255,0))
        s.blit(t, (640-t.get_width()//2, 200))
        st = pygame.font.SysFont("Arial", 40).render("START", True, (255,255,255))
        s.blit(st, (self.btn.centerx-st.get_width()//2, self.btn.centery-st.get_height()//2))

class LoseScene:
    def __init__(self, manager):
        self.manager = manager
        self.font = pygame.font.SysFont("Arial", 80)
        
    def update(self, p, g, jp, jr, rel):
        if p and jp:
            self.manager.change_scene("TITLE")
            
    def draw(self, s):
        s.fill((50,0,0))
        t = self.font.render("GAME OVER", True, (255,255,255))
        s.blit(t, (640-t.get_width()//2, 300))

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
    pygame.display.set_caption("Factory Frenzy")
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
    
    print(f"=== CAMERA DIAGNOSTICS ===")
    print(f"Camera Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"Digital Zoom: {DIGITAL_ZOOM}x")
    print(f"Background Isolation: {'ON' if IGNORE_BACKGROUND_HANDS else 'OFF'}")
    print(f"==========================")
    
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
            if pointer:
                col = (0, 255, 0) if is_grabbing else (255, 200, 0)
                if not is_reliable:
                    col = (255, 0, 0)
                pygame.draw.circle(screen, col, pointer, 15 if is_grabbing else 10)
                pygame.draw.circle(screen, (255,255,255), pointer, 19 if is_grabbing else 14, 2)
        
        pygame.display.flip()
        clock.tick(30)
    
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()