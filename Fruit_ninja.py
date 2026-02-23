import pygame
import cv2
import numpy as np
import time
import os
import random
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# --- CONFIGURATION ---
WINDOW_SIZE = (1280, 720)
MODEL_PATH = "models/hand_landmarker.task"

# DEV MODE: Set to True for testing on slower hardware (M1 Air)
# Set to False for production deployment on strong PCs
DEV_MODE = False  # Toggle this!

# DISTANCE OPTIMIZATION: Fixed digital zoom for 2-3 meter play
# Higher = more zoom (better for distance, but smaller play area)
# 1.0 = no zoom, 1.5 = 50% zoom, 2.0 = 100% zoom
DIGITAL_ZOOM = 1.75  # Adjust this for your setup!

# HAND ISOLATION: Only track the LARGEST/CLOSEST hand (ignore background)
IGNORE_BACKGROUND_HANDS = True

# --- COLORS ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)
GREEN = (100, 255, 100)
PURPLE = (200, 100, 255)

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

# --- DIGITAL ZOOM FUNCTION ---
def apply_digital_zoom(frame, zoom_factor):
    """Crop center of frame and scale up for digital zoom effect"""
    if zoom_factor <= 1.0:
        return frame
    
    h, w = frame.shape[:2]
    
    # Calculate crop dimensions (smaller crop = more zoom)
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)
    
    # Calculate crop position (center)
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    
    # Crop center
    cropped = frame[start_y:start_y + crop_h, start_x:start_x + crop_w]
    
    # Scale back to original size
    zoomed = cv2.resize(cropped, (w, h))
    
    return zoomed

# --- HAND SIZE CALCULATOR (for isolation) ---
def calculate_hand_size(landmarks):
    """Calculate approximate hand size based on landmark spread"""
    # Get bounding box of all landmarks
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    
    # Return area as size metric
    return width * height

# --- PARTICLE SYSTEM ---
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 10)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 3
        self.color = color
        self.size = random.randint(4, 10)
        self.lifetime = 40
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.4  # gravity
        self.lifetime -= 1
        self.size = max(1, self.size - 0.15)
        
    def draw(self, surf):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / 40))
            s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (int(self.size), int(self.size)), int(self.size))
            surf.blit(s, (int(self.x - self.size), int(self.y - self.size)))

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
        
        load("SLICE", "Rise02.wav")
        load("EXPLODE", "Downer01.wav")
        load("MISS", "Coin01.wav")
        
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

# --- HAND CONTROLLER (Finger Tracking for Slicing) ---
class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.CONFIDENCE_THRESHOLD = 0.3  # Lower for distance detection
        self.MARGIN = 0.25
        self.trail = deque(maxlen=15)
        self.velocity = 0
        
        # Kalman-like prediction for dropped frames
        self.vx, self.vy = 0, 0  # Velocity tracking
        self.frames_lost = 0
        self.MAX_PREDICTION_FRAMES = 3  # Predict up to 3 frames
    
    def process(self, landmarks, handedness_score, w, h):
        # If tracking is lost, use prediction
        if handedness_score < self.CONFIDENCE_THRESHOLD:
            self.frames_lost += 1
            if self.frames_lost <= self.MAX_PREDICTION_FRAMES and self.prev_x != 0:
                # Predict position based on velocity
                pred_x = int(self.prev_x + self.vx)
                pred_y = int(self.prev_y + self.vy)
                # Clamp to screen
                pred_x = max(0, min(w, pred_x))
                pred_y = max(0, min(h, pred_y))
                self.trail.append((pred_x, pred_y))
                return (pred_x, pred_y), self.velocity * 0.9, False  # Indicate unreliable
            return (self.prev_x, self.prev_y), 0, False
        
        # Reset lost frame counter
        self.frames_lost = 0
        
        # Use index finger tip for slicing
        index_tip = landmarks[8]
        
        # Sensitivity mapping
        clamped_x = max(self.MARGIN, min(1 - self.MARGIN, index_tip.x))
        clamped_y = max(self.MARGIN, min(1 - self.MARGIN, index_tip.y))
        active_width = 1 - (2 * self.MARGIN)
        normalized_x = (clamped_x - self.MARGIN) / active_width
        normalized_y = (clamped_y - self.MARGIN) / active_width
        target_x, target_y = int(normalized_x * w), int(normalized_y * h)

        # Aggressive smoothing for distance tracking
        SMOOTH = 0.5  # Increased from 0.3
        if self.prev_x == 0:
            self.prev_x, self.prev_y = target_x, target_y
        
        curr_x = int(SMOOTH * target_x + (1 - SMOOTH) * self.prev_x)
        curr_y = int(SMOOTH * target_y + (1 - SMOOTH) * self.prev_y)
        
        # Update velocity for prediction
        self.vx = curr_x - self.prev_x
        self.vy = curr_y - self.prev_y
        self.velocity = math.hypot(self.vx, self.vy)
        
        self.prev_x, self.prev_y = curr_x, curr_y
        self.trail.append((curr_x, curr_y))
        
        return (curr_x, curr_y), self.velocity, True

# --- FRUIT CLASS ---
class Fruit:
    def __init__(self):
        self.fruit_types = [
            {"name": "watermelon", "color": GREEN, "radius": 40, "points": 10},
            {"name": "orange", "color": ORANGE, "radius": 32, "points": 15},
            {"name": "apple", "color": RED, "radius": 30, "points": 15},
            {"name": "grape", "color": PURPLE, "radius": 22, "points": 20},
            {"name": "banana", "color": YELLOW, "radius": 36, "points": 12},
        ]
        self.fruit_type = random.choice(self.fruit_types)
        self.x = random.randint(150, WINDOW_SIZE[0] - 150)
        self.y = WINDOW_SIZE[1] + 50
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-20, -25)
        self.radius = self.fruit_type["radius"]
        self.color = self.fruit_type["color"]
        self.points = self.fruit_type["points"]
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-4, 4)
        self.sliced = False
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.7  # gravity
        self.rotation += self.rotation_speed
        
    def draw(self, surf):
        if not self.sliced:
            # Glow effect
            glow_surf = pygame.Surface((self.radius * 3, self.radius * 3), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.color, 40), (int(self.radius * 1.5), int(self.radius * 1.5)), int(self.radius * 1.3))
            surf.blit(glow_surf, (int(self.x - self.radius * 1.5), int(self.y - self.radius * 1.5)))
            
            # Main fruit
            pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), self.radius)
            
            # Highlight
            highlight_color = tuple(min(c + 80, 255) for c in self.color)
            highlight_offset = self.radius // 3
            pygame.draw.circle(surf, highlight_color, (int(self.x - highlight_offset), int(self.y - highlight_offset)), self.radius // 3)
            
    def is_off_screen(self):
        return self.y > WINDOW_SIZE[1] + 100

# --- BOMB CLASS ---
class Bomb:
    def __init__(self):
        self.x = random.randint(150, WINDOW_SIZE[0] - 150)
        self.y = WINDOW_SIZE[1] + 50
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-18, -22)
        self.radius = 38
        self.sliced = False
        self.spark_phase = 0
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.7
        self.spark_phase = (self.spark_phase + 1) % 20
        
    def draw(self, surf):
        if not self.sliced:
            # Bomb body
            pygame.draw.circle(surf, BLACK, (int(self.x), int(self.y)), self.radius)
            pygame.draw.circle(surf, (60, 60, 60), (int(self.x), int(self.y)), self.radius, 3)
            
            # Fuse
            fuse_x = self.x - self.radius * 0.7
            fuse_y = self.y - self.radius * 0.7
            pygame.draw.line(surf, (139, 69, 19), (int(self.x - self.radius * 0.4), int(self.y - self.radius * 0.4)), 
                           (int(fuse_x), int(fuse_y)), 4)
            
            # Animated spark
            if self.spark_phase < 10:
                spark_size = 6 + (self.spark_phase % 3)
                pygame.draw.circle(surf, YELLOW, (int(fuse_x), int(fuse_y)), spark_size)
                pygame.draw.circle(surf, ORANGE, (int(fuse_x), int(fuse_y)), spark_size - 2)
    
    def is_off_screen(self):
        return self.y > WINDOW_SIZE[1] + 100

# --- GAME SCENE ---
class GameScene:
    def __init__(self, manager):
        self.manager = manager
        self.font_large = pygame.font.SysFont("Arial", 72, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 48)
        self.font_small = pygame.font.SysFont("Arial", 36)
        
        self.fruits = []
        self.bombs = []
        self.particles = []
        
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.combo_timer = 0
        
        self.spawn_timer = 0
        self.spawn_rate = 50  # frames
        
        self.game_over = False
        
        self.manager.sound.play_music()
    
    def spawn_object(self):
        if random.random() < 0.15:  # 15% bomb chance
            self.bombs.append(Bomb())
        else:
            self.fruits.append(Fruit())
    
    def check_slice(self, hand_x, hand_y, prev_x, prev_y, velocity):
        if velocity < 15:  # Minimum swipe speed
            return
        
        # Check fruits
        for fruit in self.fruits[:]:
            if not fruit.sliced:
                # Line segment to circle collision
                dx = hand_x - prev_x
                dy = hand_y - prev_y
                fx = fruit.x - prev_x
                fy = fruit.y - prev_y
                
                if dx == 0 and dy == 0:
                    dist = math.hypot(hand_x - fruit.x, hand_y - fruit.y)
                else:
                    t = max(0, min(1, (fx * dx + fy * dy) / (dx * dx + dy * dy)))
                    closest_x = prev_x + t * dx
                    closest_y = prev_y + t * dy
                    dist = math.hypot(closest_x - fruit.x, closest_y - fruit.y)
                
                if dist < fruit.radius + 20:
                    fruit.sliced = True
                    self.score += fruit.points
                    self.combo += 1
                    self.combo_timer = 60
                    self.manager.sound.play("SLICE")
                    
                    # Create particles
                    for _ in range(25):
                        self.particles.append(Particle(fruit.x, fruit.y, fruit.color))
                    
                    self.fruits.remove(fruit)
        
        # Check bombs
        for bomb in self.bombs[:]:
            if not bomb.sliced:
                dx = hand_x - prev_x
                dy = hand_y - prev_y
                bx = bomb.x - prev_x
                by = bomb.y - prev_y
                
                if dx == 0 and dy == 0:
                    dist = math.hypot(hand_x - bomb.x, hand_y - bomb.y)
                else:
                    t = max(0, min(1, (bx * dx + by * dy) / (dx * dx + dy * dy)))
                    closest_x = prev_x + t * dx
                    closest_y = prev_y + t * dy
                    dist = math.hypot(closest_x - bomb.x, closest_y - bomb.y)
                
                if dist < bomb.radius + 20:
                    bomb.sliced = True
                    self.lives -= 1
                    self.combo = 0
                    self.manager.sound.play("EXPLODE")
                    
                    # Explosion particles
                    for _ in range(60):
                        self.particles.append(Particle(bomb.x, bomb.y, RED))
                    
                    self.bombs.remove(bomb)
                    
                    if self.lives <= 0:
                        self.game_over = True
    
    def update(self, pointer, velocity, is_reliable, prev_pointer):
        if self.game_over:
            return
        
        # Spawn objects
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_object()
            self.spawn_timer = 0
            self.spawn_rate = max(30, self.spawn_rate - 0.2)  # Speed up gradually
        
        # Update fruits
        for fruit in self.fruits[:]:
            fruit.update()
            if fruit.is_off_screen():
                self.fruits.remove(fruit)
                if not fruit.sliced:
                    self.lives -= 1
                    self.combo = 0
                    self.manager.sound.play("MISS")
                    if self.lives <= 0:
                        self.game_over = True
        
        # Update bombs
        for bomb in self.bombs[:]:
            bomb.update()
            if bomb.is_off_screen():
                self.bombs.remove(bomb)
        
        # Update particles
        for particle in self.particles[:]:
            particle.update()
            if particle.lifetime <= 0:
                self.particles.remove(particle)
        
        # Combo timer
        if self.combo_timer > 0:
            self.combo_timer -= 1
            if self.combo_timer == 0:
                self.combo = 0
        
        # Check slicing
        if pointer and prev_pointer and is_reliable:
            self.check_slice(pointer[0], pointer[1], prev_pointer[0], prev_pointer[1], velocity)
    
    def draw(self, screen):
        # Gradient background
        for y in range(WINDOW_SIZE[1]):
            color_factor = y / WINDOW_SIZE[1]
            color = (int(30 + 20 * color_factor), int(20 + 30 * color_factor), int(60 + 40 * color_factor))
            pygame.draw.line(screen, color, (0, y), (WINDOW_SIZE[0], y))
        
        # Draw fruits and bombs
        for fruit in self.fruits:
            fruit.draw(screen)
        for bomb in self.bombs:
            bomb.draw(screen)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(screen)
        
        # HUD
        score_text = self.font_medium.render(f"Score: {self.score}", True, YELLOW)
        screen.blit(score_text, (20, 20))
        
        # Lives
        for i in range(self.lives):
            pygame.draw.circle(screen, RED, (WINDOW_SIZE[0] - 50 - i * 50, 40), 18)
            pygame.draw.circle(screen, (150, 0, 0), (WINDOW_SIZE[0] - 50 - i * 50, 40), 18, 3)
        
        # Combo
        if self.combo > 1:
            combo_text = self.font_large.render(f"COMBO x{self.combo}!", True, YELLOW)
            combo_alpha = min(255, self.combo_timer * 4)
            combo_surf = pygame.Surface(combo_text.get_size(), pygame.SRCALPHA)
            
            # Draw with glow
            for offset in [(2,2), (-2,2), (2,-2), (-2,-2)]:
                glow = self.font_large.render(f"COMBO x{self.combo}!", True, ORANGE)
                combo_surf.blit(glow, offset)
            combo_surf.blit(combo_text, (0, 0))
            combo_surf.set_alpha(combo_alpha)
            
            screen.blit(combo_surf, (WINDOW_SIZE[0] // 2 - combo_text.get_width() // 2, 120))
        
        # Game over
        if self.game_over:
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("GAME OVER", True, RED)
            final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, WHITE)
            restart_text = self.font_small.render("Press SPACE to Restart", True, YELLOW)
            
            screen.blit(game_over_text, (WINDOW_SIZE[0] // 2 - game_over_text.get_width() // 2, WINDOW_SIZE[1] // 2 - 100))
            screen.blit(final_score_text, (WINDOW_SIZE[0] // 2 - final_score_text.get_width() // 2, WINDOW_SIZE[1] // 2))
            screen.blit(restart_text, (WINDOW_SIZE[0] // 2 - restart_text.get_width() // 2, WINDOW_SIZE[1] // 2 + 80))

# --- TITLE SCENE ---
class TitleScene:
    def __init__(self, manager):
        self.manager = manager
        self.font_title = pygame.font.SysFont("Arial", 80, bold=True)
        self.font_subtitle = pygame.font.SysFont("Arial", 40)
        self.pulse = 0
    
    def update(self, pointer, velocity, is_reliable, prev_pointer):
        self.pulse = (self.pulse + 0.1) % (2 * math.pi)
    
    def draw(self, screen):
        # Background
        for y in range(WINDOW_SIZE[1]):
            color_factor = y / WINDOW_SIZE[1]
            color = (int(30 + 20 * color_factor), int(20 + 30 * color_factor), int(60 + 40 * color_factor))
            pygame.draw.line(screen, color, (0, y), (WINDOW_SIZE[0], y))
        
        # Title with pulse effect
        scale = 1.0 + 0.1 * math.sin(self.pulse)
        title_text = self.font_title.render("FRUIT NINJA", True, YELLOW)
        scaled_title = pygame.transform.scale(title_text, (int(title_text.get_width() * scale), int(title_text.get_height() * scale)))
        screen.blit(scaled_title, (WINDOW_SIZE[0] // 2 - scaled_title.get_width() // 2, 150))
        
        # Instructions
        instructions = [
            "Swipe your finger across fruits to slice them!",
            "Avoid the bombs!",
            "",
            "Press SPACE to Start"
        ]
        
        y_offset = 350
        for instruction in instructions:
            if instruction:
                text = self.font_subtitle.render(instruction, True, WHITE)
                screen.blit(text, (WINDOW_SIZE[0] // 2 - text.get_width() // 2, y_offset))
            y_offset += 50

# --- GAME MANAGER ---
class GameManager:
    def __init__(self):
        self.sound = SoundManager()
        self.scenes = {"TITLE": TitleScene(self), "GAME": GameScene(self)}
        self.current_scene = self.scenes["TITLE"]
    
    def change_scene(self, name):
        if name == "GAME":
            self.scenes["GAME"] = GameScene(self)
        self.current_scene = self.scenes[name]

# --- MAIN ---
def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Gesture Fruit Ninja")
    clock = pygame.time.Clock()
    
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,  # Detect up to 2 hands so we can pick the largest
        min_hand_detection_confidence=0.3,  # Lower for distance (was 0.5)
        min_hand_presence_confidence=0.3,   # Lower for distance (was 0.5)
        min_tracking_confidence=0.3         # Lower for distance (was 0.5)
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    # Elgato Facecam optimization - Use 720p for more stable capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request max FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)  # Faster shutter for less motion blur
    
    # Processing resolution (much smaller for speed)
    if DEV_MODE:
        PROCESSING_WIDTH = 480  # Lower for M1 Air testing
        PROCESSING_HEIGHT = 360
        print("🔧 DEV MODE: Using 480p processing for better performance")
    else:
        PROCESSING_WIDTH = 640  # Full quality for production
        PROCESSING_HEIGHT = 480
        print("🚀 PRODUCTION MODE: Using 640p processing")
    
    controller = HandController()
    manager = GameManager()
    show_camera = False
    prev_pointer = None
    result = None  # Initialize result to avoid UnboundLocalError
    
    # DIAGNOSTICS: Track performance
    frame_times = deque(maxlen=60)
    detection_count = 0
    total_frames = 0
    last_diagnostic_print = time.time()
    
    # Timing breakdown
    capture_times = deque(maxlen=60)
    processing_times = deque(maxlen=60)
    render_times = deque(maxlen=60)
    
    # Frame skipping for consistent timing
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 2 if DEV_MODE else 1  # Process every other frame in dev mode
    
    # Print camera info
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"=== CAMERA DIAGNOSTICS ===")
    print(f"Camera Resolution: {actual_width}x{actual_height}")
    print(f"Camera FPS: {actual_fps}")
    print(f"Camera Backend: {cap.getBackendName()}")
    print(f"Digital Zoom: {DIGITAL_ZOOM}x")
    print(f"Background Isolation: {'ON' if IGNORE_BACKGROUND_HANDS else 'OFF'}")
    print(f"Starting game loop...")
    print(f"==========================")
    
    while True:
        frame_start = time.time()
        frame_counter += 1
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
                cap.release()
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_d:
                    show_camera = not show_camera
                if e.key == pygame.K_SPACE:
                    if isinstance(manager.current_scene, TitleScene):
                        manager.change_scene("GAME")
                    elif isinstance(manager.current_scene, GameScene) and manager.current_scene.game_over:
                        manager.change_scene("GAME")
        
        ok, frame = cap.read()
        if not ok:
            break
        
        capture_time = time.time() - frame_start
        processing_start = time.time()
        
        frame = cv2.flip(frame, 1)
        
        # Apply digital zoom BEFORE processing
        if DIGITAL_ZOOM > 1.0:
            frame = apply_digital_zoom(frame, DIGITAL_ZOOM)
        
        # Only process MediaPipe on certain frames for dev mode
        if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
            # OPTIMIZATION: Downscale for MediaPipe processing
            small_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
            
            # Skip gamma correction in DEV_MODE for speed
            if DEV_MODE:
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            else:
                small_frame, gamma_active = smart_adjust_gamma(small_frame)
                rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = pygame.time.get_ticks()
            result = landmarker.detect_for_video(mp_img, timestamp_ms)
        # else: reuse previous result (frame skipping)
        
        processing_time = time.time() - processing_start
        render_start = time.time()
        
        # HAND ISOLATION: Pick the largest/closest hand (ignore background)
        selected_hand = None
        selected_score = 0
        
        if result and result.hand_landmarks and IGNORE_BACKGROUND_HANDS:
            if len(result.hand_landmarks) == 1:
                # Only one hand detected, use it
                selected_hand = result.hand_landmarks[0]
                selected_score = result.handedness[0][0].score
            else:
                # Multiple hands detected - pick the LARGEST (closest to camera)
                largest_size = 0
                for i, hand in enumerate(result.hand_landmarks):
                    hand_size = calculate_hand_size(hand)
                    if hand_size > largest_size:
                        largest_size = hand_size
                        selected_hand = hand
                        selected_score = result.handedness[i][0].score
        elif result and result.hand_landmarks:
            # No isolation, use first hand
            selected_hand = result.hand_landmarks[0]
            selected_score = result.handedness[0][0].score
        
        pointer, velocity, is_reliable = None, 0, False
        if selected_hand:  # Use isolated hand
            pointer, velocity, is_reliable = controller.process(
                selected_hand, selected_score, WINDOW_SIZE[0], WINDOW_SIZE[1]
            )
            detection_count += 1
        
        total_frames += 1
        
        # Print diagnostics every 2 seconds
        if time.time() - last_diagnostic_print > 2.0:
            detection_rate = (detection_count / total_frames) * 100 if total_frames > 0 else 0
            avg_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0
            confidence = selected_score if selected_hand else 0
            
            # Timing breakdown
            avg_capture = sum(capture_times) / len(capture_times) * 1000 if capture_times else 0
            avg_processing = sum(processing_times) / len(processing_times) * 1000 if processing_times else 0
            avg_render = sum(render_times) / len(render_times) * 1000 if render_times else 0
            
            print(f"Detection: {detection_rate:.1f}% | FPS: {avg_fps:.1f} | Conf: {confidence:.2f}")
            print(f"  Capture: {avg_capture:.1f}ms | Processing: {avg_processing:.1f}ms | Render: {avg_render:.1f}ms")
            
            detection_count = 0
            total_frames = 0
            last_diagnostic_print = time.time()
        
        if show_camera:
            # Use small frame for camera view (faster rendering)
            cam_surface = pygame.surfarray.make_surface(np.rot90(rgb_frame))
            cam_surface = pygame.transform.flip(cam_surface, True, False)
            cam_surface = pygame.transform.scale(cam_surface, WINDOW_SIZE)  # Scale up to window
            screen.blit(cam_surface, (0, 0))
            
            # Show zoom level and isolation status
            debug_font = pygame.font.SysFont("Arial", 24)
            zoom_text = debug_font.render(f"Zoom: {DIGITAL_ZOOM}x", True, (0, 255, 255))
            screen.blit(zoom_text, (20, 100))
            
            if IGNORE_BACKGROUND_HANDS and result and result.hand_landmarks:
                hands_text = debug_font.render(f"Hands Detected: {len(result.hand_landmarks)}", True, (255, 255, 0))
                screen.blit(hands_text, (20, 130))
                if len(result.hand_landmarks) > 1:
                    isolated_text = debug_font.render("Using LARGEST hand", True, (0, 255, 0))
                    screen.blit(isolated_text, (20, 160))
            
            # Draw trail
            if len(controller.trail) > 1:
                for i in range(len(controller.trail) - 1):
                    alpha = int(255 * (i / len(controller.trail)))
                    thickness = int(10 * (i / len(controller.trail))) + 2
                    s = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
                    pygame.draw.line(s, (0, 255, 255, alpha), controller.trail[i], controller.trail[i + 1], thickness)
                    screen.blit(s, (0, 0))
            
            if pointer:
                # Velocity indicator
                color_intensity = min(255, int(velocity * 10))
                pygame.draw.circle(screen, (color_intensity, 255 - color_intensity, 0), pointer, 15)
        else:
            manager.current_scene.update(pointer, velocity, is_reliable, prev_pointer)
            manager.current_scene.draw(screen)
            
            # Draw trail
            if len(controller.trail) > 1:
                for i in range(len(controller.trail) - 1):
                    alpha = int(200 * (i / len(controller.trail)))
                    thickness = int(12 * (i / len(controller.trail))) + 3
                    s = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
                    pygame.draw.line(s, (100, 200, 255, alpha), controller.trail[i], controller.trail[i + 1], thickness)
                    screen.blit(s, (0, 0))
            
            # Draw cursor
            if pointer:
                color = (100, 255, 100) if is_reliable else (255, 100, 100)
                pygame.draw.circle(screen, color, pointer, 12)
                pygame.draw.circle(screen, WHITE, pointer, 15, 2)
        
        prev_pointer = pointer
        
        render_time = time.time() - render_start
        
        pygame.display.flip()
        clock.tick(60)
        
        # Track frame time for FPS calculation
        total_time = time.time() - frame_start
        frame_times.append(total_time)
        capture_times.append(capture_time)
        processing_times.append(processing_time)
        render_times.append(render_time)
    
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()