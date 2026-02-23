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
WINDOW_SIZE = (1920, 1080)  # Fullscreen resolution
MODEL_PATH = "models/hand_landmarker.task"

# Distance optimization settings
DEV_MODE = True
DIGITAL_ZOOM = 1.5
IGNORE_BACKGROUND_HANDS = True

# ROI (Region of Interest) settings for background hand filtering
USE_ROI = True  # Only detect hands in center region
ROI_MARGIN = 0.15  # Ignore outer 15% of frame (where background people stand)
USE_Z_DEPTH = True  # Prioritize closest hand using Z-coordinate

# Game balance - BALANCED INTENSITY
ENEMY_THROW_TIME = 2.0  # 2 seconds (was 1.5)
DIFFICULTY_INCREASE_RATE = 0.03  # Moderate ramp (was 0.05)
MAX_ENEMIES_ON_SCREEN = 6  # Slightly fewer (was 8)

# --- ASSET FOLDERS ---
ENEMY_FOLDER = "assets/enemies"
PROJECTILE_FOLDER = "assets/projectiles"
SOUND_FOLDER = "assets/sounds"
MUSIC_FOLDER = "assets/music"
BACKGROUND_FOLDER = "assets/backgrounds"

# --- COLORS ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
DARK_RED = (100, 0, 0)
BLUE = (100, 150, 255)
YELLOW = (255, 200, 0)

# --- HELPER FUNCTIONS ---
def smart_adjust_gamma(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness < 90:
        gamma = 1.8
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table), True
    return image, False

def apply_digital_zoom(frame, zoom_factor):
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)
    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2
    cropped = frame[start_y:start_y + crop_h, start_x:start_x + crop_w]
    return cv2.resize(cropped, (w, h))

def calculate_hand_size(landmarks):
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width * height

def is_hand_in_roi(landmarks, roi_margin=0.15):
    """Check if hand center is within ROI (center region of frame)"""
    # Calculate hand center
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    
    # Check if center is within ROI bounds
    in_x = roi_margin < center_x < (1 - roi_margin)
    in_y = roi_margin < center_y < (1 - roi_margin)
    
    return in_x and in_y

def get_hand_z_depth(landmarks):
    """Get average Z-depth of hand (lower = closer to camera)"""
    z_coords = [lm.z for lm in landmarks]
    return sum(z_coords) / len(z_coords)

def load_sprite_flexible(folder, filename_variants, size, fallback_color):
    """Try multiple filename variations (case-insensitive)"""
    for variant in filename_variants:
        path = os.path.join(folder, variant)
        if os.path.exists(path):
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.scale(img, size)
            except:
                pass
    
    # Fallback: colored circle
    surf = pygame.Surface(size, pygame.SRCALPHA)
    pygame.draw.circle(surf, fallback_color, (size[0]//2, size[1]//2), min(size)//2)
    return surf

def load_background():
    """Load background image or return None for solid color fallback"""
    bg_variants = [
        "background.png", "Background.png", "BACKGROUND.png",
        "bg.png", "BG.png", "Bg.png",
        "background.jpg", "Background.jpg", "BACKGROUND.jpg",
        "bg.jpg", "BG.jpg", "Bg.jpg"
    ]
    
    for variant in bg_variants:
        path = os.path.join(BACKGROUND_FOLDER, variant)
        if os.path.exists(path):
            try:
                img = pygame.image.load(path).convert()
                return pygame.transform.scale(img, WINDOW_SIZE)
            except:
                pass
    
    return None  # No background found, use solid color

# --- PARTICLE SYSTEM (OPTIMIZED) ---
class ParticleSystem:
    """Pre-rendered particle system for better performance"""
    def __init__(self):
        self.particles = []
        # Pre-create particle surfaces (cache them)
        self.particle_surfaces = {}
        for size in range(4, 13):
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 255, 255), (size, size), size)
            self.particle_surfaces[size] = surf
    
    def add_particles(self, x, y, color, count=20):
        """Add multiple particles at once"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 10)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 2
            size = random.randint(4, 12)
            self.particles.append([x, y, vx, vy, size, 40, color])  # x, y, vx, vy, size, lifetime, color
    
    def update_and_draw(self, surf):
        """Batch update and draw"""
        for p in self.particles[:]:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[3] += 0.4   # vy += gravity
            p[5] -= 1     # lifetime -= 1
            
            if p[5] <= 0:
                self.particles.remove(p)
            else:
                # Use pre-rendered surface with color mod
                size = int(p[4])
                if size in self.particle_surfaces:
                    alpha = int(255 * (p[5] / 40))
                    temp_surf = self.particle_surfaces[size].copy()
                    temp_surf.fill((*p[6], alpha), special_flags=pygame.BLEND_RGBA_MULT)
                    surf.blit(temp_surf, (int(p[0] - size), int(p[1] - size)), special_flags=pygame.BLEND_ALPHA_SDL2)

# --- PROJECTILE CLASS ---
class Projectile:
    def __init__(self, x, y, target_x, target_y):
        self.x = x
        self.y = y
        dx = target_x - x
        dy = target_y - y
        dist = math.hypot(dx, dy)
        speed = 8
        self.vx = (dx / dist) * speed if dist > 0 else 0
        self.vy = (dy / dist) * speed if dist > 0 else 0
        self.radius = 15
        self.active = True
        
        # Try to load projectile with flexible naming
        self.image = load_sprite_flexible(
            PROJECTILE_FOLDER,
            ["shuriken.png", "Shuriken.png", "projectile.png", "Projectile.png"],
            (30, 30),
            (50, 50, 50)
        )
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        if self.x < -50 or self.x > WINDOW_SIZE[0] + 50 or self.y < -50 or self.y > WINDOW_SIZE[1] + 50:
            self.active = False
            
    def draw(self, surf):
        surf.blit(self.image, (int(self.x - 15), int(self.y - 15)))

# --- ENEMY CLASS WITH STATE-BASED SPRITES ---
class Enemy:
    TYPES = {
        "bandit": {"hp": 1, "speed": 1.0, "points": 10, "color": (139, 69, 19)},
        "ninja": {"hp": 1, "speed": 1.5, "points": 15, "color": (50, 50, 50)},
        "samurai": {"hp": 2, "speed": 0.8, "points": 25, "color": (150, 150, 150)},
        "oni": {"hp": 2, "speed": 1.8, "points": 30, "color": (200, 0, 0)}
    }
    
    def __init__(self, enemy_type="bandit"):
        self.type = enemy_type
        self.props = self.TYPES[enemy_type]
        self.hp = self.props["hp"]
        self.max_hp = self.props["hp"]
        self.points = self.props["points"]
        
        # Position
        self.x = random.randint(100, WINDOW_SIZE[0] - 100)
        self.y = WINDOW_SIZE[1] + 50
        self.target_y = random.randint(WINDOW_SIZE[1] - 250, WINDOW_SIZE[1] - 150)
        
        # Movement
        self.speed = self.props["speed"]
        self.rising = True
        self.rise_timer = 0
        self.rise_duration = 40  # Faster rise (was 60)
        
        # Throw mechanics
        self.throw_timer = 0
        self.throw_cooldown = ENEMY_THROW_TIME * 60
        self.has_thrown = False
        
        # State management
        self.state = "idle"  # idle, attacking, hurt, death
        self.state_timer = 0
        
        self.radius = 40
        self.sliced = False
        
        # Load all sprite states with flexible naming
        enemy_folder = os.path.join(ENEMY_FOLDER, enemy_type)
        sprite_size = (80, 80)
        
        self.sprites = {
            "idle": load_sprite_flexible(
                enemy_folder,
                ["idle.png", "Idle.png", "IDLE.png"],
                sprite_size,
                self.props["color"]
            ),
            "attack": load_sprite_flexible(
                enemy_folder,
                ["attack1.png", "Attack1.png", "ATTACK1.png", "attack.png", "Attack.png"],
                sprite_size,
                self.props["color"]
            ),
            "hurt": load_sprite_flexible(
                enemy_folder,
                ["take hit.png", "Take Hit.png", "take_hit.png", "Take_Hit.png", "TAKE_HIT.png", "hurt.png", "Hurt.png"],
                sprite_size,
                (255, 100, 100)
            ),
            "death": load_sprite_flexible(
                enemy_folder,
                ["death.png", "Death.png", "DEATH.png"],
                sprite_size,
                (100, 100, 100)
            )
        }
        
        self.current_sprite = self.sprites["idle"]
        
    def set_state(self, new_state, duration=0):
        """Change enemy state and sprite"""
        self.state = new_state
        self.state_timer = duration
        self.current_sprite = self.sprites.get(new_state, self.sprites["idle"])
        
    def update(self):
        # Update state timer
        if self.state_timer > 0:
            self.state_timer -= 1
            if self.state_timer == 0:
                self.set_state("idle")
        
        # Rise up animation
        if self.rising:
            self.rise_timer += 1
            progress = self.rise_timer / self.rise_duration
            self.y = WINDOW_SIZE[1] + 50 - (progress * (WINDOW_SIZE[1] + 50 - self.target_y))
            
            if self.rise_timer >= self.rise_duration:
                self.rising = False
                self.y = self.target_y
        else:
            self.throw_timer += 1
            
    def should_throw(self):
        return not self.rising and self.throw_timer >= self.throw_cooldown and not self.has_thrown
    
    def throw(self):
        self.has_thrown = True
        self.set_state("attack", 30)  # Show attack sprite for 0.5 seconds
        return Projectile(self.x, self.y, WINDOW_SIZE[0] // 2, 50)
    
    def take_damage(self):
        self.hp -= 1
        if self.hp > 0:
            self.set_state("hurt", 15)  # Flash hurt sprite briefly
            return False
        else:
            self.set_state("death", 30)  # Show death sprite before removal
            return True
    
    def draw(self, surf):
        if not self.sliced:
            # Draw current sprite
            surf.blit(self.current_sprite, (int(self.x - 40), int(self.y - 40)))
            
            # Draw HP bar if multi-HP enemy
            if self.max_hp > 1:
                bar_width = 60
                bar_height = 6
                hp_ratio = self.hp / self.max_hp
                pygame.draw.rect(surf, DARK_RED, (int(self.x - bar_width//2), int(self.y - 55), bar_width, bar_height))
                pygame.draw.rect(surf, RED, (int(self.x - bar_width//2), int(self.y - 55), int(bar_width * hp_ratio), bar_height))
            
            # Draw throw indicator
            if not self.rising and not self.has_thrown and self.state != "attack":
                progress = self.throw_timer / self.throw_cooldown
                indicator_size = int(20 * progress)
                color_intensity = int(255 * progress)
                pygame.draw.circle(surf, (color_intensity, 0, 0), (int(self.x), int(self.y)), indicator_size, 2)

# --- HAND CONTROLLER ---
class HandController:
    def __init__(self):
        self.prev_x, self.prev_y = 0, 0
        self.CONFIDENCE_THRESHOLD = 0.3
        self.MARGIN = 0.25
        self.trail = deque(maxlen=15)
        self.velocity = 0
    
    def process(self, landmarks, handedness_score, w, h):
        if handedness_score < self.CONFIDENCE_THRESHOLD:
            return (self.prev_x, self.prev_y), 0, False
        
        index_tip = landmarks[8]
        
        clamped_x = max(self.MARGIN, min(1 - self.MARGIN, index_tip.x))
        clamped_y = max(self.MARGIN, min(1 - self.MARGIN, index_tip.y))
        active_width = 1 - (2 * self.MARGIN)
        normalized_x = (clamped_x - self.MARGIN) / active_width
        normalized_y = (clamped_y - self.MARGIN) / active_width
        target_x, target_y = int(normalized_x * w), int(normalized_y * h)

        SMOOTH = 0.3
        if self.prev_x == 0:
            self.prev_x, self.prev_y = target_x, target_y
        
        curr_x = int(SMOOTH * target_x + (1 - SMOOTH) * self.prev_x)
        curr_y = int(SMOOTH * target_y + (1 - SMOOTH) * self.prev_y)
        
        dx = curr_x - self.prev_x
        dy = curr_y - self.prev_y
        self.velocity = math.hypot(dx, dy)
        
        self.prev_x, self.prev_y = curr_x, curr_y
        self.trail.append((curr_x, curr_y))
        
        return (curr_x, curr_y), self.velocity, True

# --- SOUND MANAGER ---
class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.load_assets()
        
    def load_assets(self):
        sound_files = {
            "SLASH": ["slash.wav", "Slash.wav", "SLASH.wav"],
            "HIT": ["hit.wav", "Hit.wav", "HIT.wav"],
            "THROW": ["throw.wav", "Throw.wav", "THROW.wav"],
            "DAMAGE": ["damage.wav", "Damage.wav", "DAMAGE.wav"]
        }
        
        for name, variants in sound_files.items():
            for variant in variants:
                path = os.path.join(SOUND_FOLDER, variant)
                if os.path.exists(path):
                    try:
                        self.sounds[name] = pygame.mixer.Sound(path)
                        self.sounds[name].set_volume(0.5)
                        break
                    except:
                        pass
        
        # Try to load background music
        music_variants = ["battle.wav", "Battle.wav", "BATTLE.wav", "music.wav", "Music.wav"]
        for variant in music_variants:
            music_path = os.path.join(MUSIC_FOLDER, variant)
            if os.path.exists(music_path):
                try:
                    pygame.mixer.music.load(music_path)
                    break
                except:
                    pass
    
    def play(self, name):
        if name in self.sounds:
            self.sounds[name].play()
            
    def play_music(self):
        try:
            pygame.mixer.music.play(-1)
        except:
            pass

# --- GAME SCENE ---
class GameScene:
    def __init__(self, sound_manager):
        self.sound = sound_manager
        self.font_large = pygame.font.SysFont("Arial", 72, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 48)
        self.font_small = pygame.font.SysFont("Arial", 36)
        
        self.enemies = []
        self.projectiles = []
        self.particle_system = ParticleSystem()
        
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.combo_timer = 0
        
        self.spawn_timer = 0
        self.spawn_rate = 60  # More balanced - spawn every 1 sec (was 0.5)
        self.min_spawn_rate = 20  # Max difficulty - spawn every 0.33 sec (was 0.16)
        
        self.game_over = False
        self.high_score = 0
        
        # Track enemies marked for death
        self.death_animations = []
        
        # Load background image
        self.background = load_background()
        self.bg_color = (20, 15, 30)  # Fallback dark purple
        
        self.sound.play_music()
    
    def spawn_enemy(self):
        # Don't spawn if too many enemies on screen
        if len(self.enemies) >= MAX_ENEMIES_ON_SCREEN:
            return
            
        if self.score < 50:
            enemy_type = random.choice(["bandit", "ninja"])
        elif self.score < 150:
            enemy_type = random.choice(["bandit", "ninja", "samurai"])
        else:
            enemy_type = random.choice(["bandit", "ninja", "samurai", "oni"])
        
        self.enemies.append(Enemy(enemy_type))
    
    def check_slice(self, hand_x, hand_y, prev_x, prev_y, velocity):
        if velocity < 15:
            return
        
        for enemy in self.enemies[:]:
            if not enemy.sliced and enemy.state != "death":
                dx = hand_x - prev_x
                dy = hand_y - prev_y
                ex = enemy.x - prev_x
                ey = enemy.y - prev_y
                
                if dx == 0 and dy == 0:
                    dist = math.hypot(hand_x - enemy.x, hand_y - enemy.y)
                else:
                    t = max(0, min(1, (ex * dx + ey * dy) / (dx * dx + dy * dy)))
                    closest_x = prev_x + t * dx
                    closest_y = prev_y + t * dy
                    dist = math.hypot(closest_x - enemy.x, closest_y - enemy.y)
                
                if dist < enemy.radius + 20:
                    killed = enemy.take_damage()
                    
                    if killed:
                        self.score += enemy.points
                        self.combo += 1
                        self.combo_timer = 60
                        self.sound.play("SLASH")
                        
                        # Optimized particles
                        self.particle_system.add_particles(enemy.x, enemy.y, (200, 0, 0), 15)
                        
                        # Mark for removal after death animation
                        self.death_animations.append((enemy, 30))
                    else:
                        self.sound.play("HIT")
    
    def check_projectile_hit(self):
        for proj in self.projectiles[:]:
            if proj.y < 100:
                self.lives -= 1
                self.combo = 0
                self.sound.play("DAMAGE")
                
                # Optimized particles
                self.particle_system.add_particles(proj.x, proj.y, (255, 100, 0), 20)
                
                proj.active = False
                
                if self.lives <= 0:
                    self.game_over = True
                    self.high_score = max(self.high_score, self.score)
    
    def update(self, pointer, velocity, is_reliable, prev_pointer):
        if self.game_over:
            return
        
        # Spawn enemies
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_enemy()
            self.spawn_timer = 0
            self.spawn_rate = max(self.min_spawn_rate, self.spawn_rate - DIFFICULTY_INCREASE_RATE)
        
        # Update enemies
        for enemy in self.enemies[:]:
            enemy.update()
            
            if enemy.should_throw():
                projectile = enemy.throw()
                self.projectiles.append(projectile)
                self.sound.play("THROW")
        
        # Update death animations
        updated_deaths = []
        for enemy, timer in self.death_animations:
            timer -= 1
            if timer <= 0:
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
            else:
                updated_deaths.append((enemy, timer))
        self.death_animations = updated_deaths
        
        # Update projectiles
        for proj in self.projectiles[:]:
            proj.update()
            if not proj.active:
                self.projectiles.remove(proj)
        
        self.check_projectile_hit()
        
        # Combo timer
        if self.combo_timer > 0:
            self.combo_timer -= 1
            if self.combo_timer == 0:
                self.combo = 0
        
        # Check slicing
        if pointer and prev_pointer and is_reliable:
            self.check_slice(pointer[0], pointer[1], prev_pointer[0], prev_pointer[1], velocity)
    
    def draw(self, screen):
        # Draw background image or solid color
        if self.background:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill(self.bg_color)
        
        for enemy in self.enemies:
            enemy.draw(screen)
        
        for proj in self.projectiles:
            proj.draw(screen)
        
        # Draw particles (optimized batch rendering)
        self.particle_system.update_and_draw(screen)
        
        # HUD
        score_text = self.font_medium.render(f"Score: {self.score}", True, YELLOW)
        screen.blit(score_text, (20, 20))
        
        for i in range(self.lives):
            pygame.draw.circle(screen, RED, (WINDOW_SIZE[0] - 50 - i * 50, 40), 18)
        
        if self.combo > 1:
            combo_text = self.font_large.render(f"{self.combo}x COMBO!", True, YELLOW)
            alpha = min(255, self.combo_timer * 4)
            combo_surf = pygame.Surface(combo_text.get_size(), pygame.SRCALPHA)
            combo_surf.blit(combo_text, (0, 0))
            combo_surf.set_alpha(alpha)
            screen.blit(combo_surf, (WINDOW_SIZE[0] // 2 - combo_text.get_width() // 2, 120))
        
        if self.game_over:
            overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (0, 0))
            
            game_over_text = self.font_large.render("GAME OVER", True, RED)
            score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
            high_text = self.font_small.render(f"High Score: {self.high_score}", True, YELLOW)
            restart_text = self.font_small.render("Press SPACE to Restart", True, WHITE)
            
            screen.blit(game_over_text, (WINDOW_SIZE[0] // 2 - game_over_text.get_width() // 2, 200))
            screen.blit(score_text, (WINDOW_SIZE[0] // 2 - score_text.get_width() // 2, 300))
            screen.blit(high_text, (WINDOW_SIZE[0] // 2 - high_text.get_width() // 2, 360))
            screen.blit(restart_text, (WINDOW_SIZE[0] // 2 - restart_text.get_width() // 2, 450))

# --- MAIN ---
def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Samurai Slash")
    clock = pygame.time.Clock()
    
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
    
    # Lower processing resolution for performance at 1920x1080
    PROCESSING_WIDTH = 320 if DEV_MODE else 480  # Much lower for speed
    PROCESSING_HEIGHT = 240 if DEV_MODE else 360
    
    controller = HandController()
    sound = SoundManager()
    game = GameScene(sound)
    
    prev_pointer = None
    result = None
    
    print("=== SAMURAI SLASH ===")
    print("Swipe to slice enemies before they throw!")
    print("Press SPACE to restart | Press Q to quit")
    
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
                cap.release()
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                if game.game_over:
                    game = GameScene(sound)
        
        ok, frame = cap.read()
        if not ok:
            break
        
        frame = cv2.flip(frame, 1)
        
        if DIGITAL_ZOOM > 1.0:
            frame = apply_digital_zoom(frame, DIGITAL_ZOOM)
        
        small_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
        
        if DEV_MODE:
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        else:
            small_frame, _ = smart_adjust_gamma(small_frame)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = pygame.time.get_ticks()
        result = landmarker.detect_for_video(mp_img, timestamp_ms)
        
        selected_hand = None
        selected_score = 0
        
        if result and result.hand_landmarks:
            # Filter hands using ROI and Z-depth
            valid_hands = []
            
            for i, hand in enumerate(result.hand_landmarks):
                # Check 1: ROI filtering (ignore hands at edges)
                if USE_ROI and not is_hand_in_roi(hand, ROI_MARGIN):
                    continue  # Skip this hand - it's outside center region
                
                # Check 2: Basic confidence
                confidence = result.handedness[i][0].score
                if confidence < 0.3:
                    continue
                
                # Store valid hand with metadata
                z_depth = get_hand_z_depth(hand) if USE_Z_DEPTH else 0
                hand_size = calculate_hand_size(hand)
                valid_hands.append({
                    'hand': hand,
                    'confidence': confidence,
                    'z_depth': z_depth,
                    'size': hand_size,
                    'index': i
                })
            
            # Select best hand from valid candidates
            if len(valid_hands) == 1:
                # Only one valid hand - use it
                selected_hand = valid_hands[0]['hand']
                selected_score = valid_hands[0]['confidence']
            elif len(valid_hands) > 1:
                # Multiple valid hands - prioritize by Z-depth (closest)
                if USE_Z_DEPTH:
                    best_hand = min(valid_hands, key=lambda h: h['z_depth'])
                else:
                    # Fallback: use largest hand
                    best_hand = max(valid_hands, key=lambda h: h['size'])
                
                selected_hand = best_hand['hand']
                selected_score = best_hand['confidence']
        
        pointer, velocity, is_reliable = None, 0, False
        if selected_hand:
            pointer, velocity, is_reliable = controller.process(
                selected_hand, selected_score, WINDOW_SIZE[0], WINDOW_SIZE[1]
            )
        
        game.update(pointer, velocity, is_reliable, prev_pointer)
        game.draw(screen)
        
        # Draw slash trail (optimized - single surface)
        if len(controller.trail) > 1:
            trail_surf = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            for i in range(len(controller.trail) - 1):
                alpha = int(200 * (i / len(controller.trail)))
                thickness = int(12 * (i / len(controller.trail))) + 3
                color = (100, 200, 255, alpha)
                pygame.draw.line(trail_surf, color, controller.trail[i], controller.trail[i + 1], thickness)
            screen.blit(trail_surf, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)
        
        if pointer and is_reliable:
            color = (100, 255, 100) if velocity > 15 else (255, 200, 0)
            pygame.draw.circle(screen, color, pointer, 12)
            pygame.draw.circle(screen, WHITE, pointer, 15, 2)
        
        prev_pointer = pointer
        pygame.display.flip()
        clock.tick(60)
    
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()