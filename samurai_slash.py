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

# Game balance - BALANCED PACE
ENEMY_THROW_TIME = 2.0  # 2 seconds before throw (was 1.5)
DIFFICULTY_INCREASE_RATE = 0.03  # Moderate difficulty ramp (was 0.05)
MAX_ENEMIES_ON_SCREEN = 6  # Moderate enemy count (was 8)

# --- ASSET FOLDERS ---
ENEMY_FOLDER = "assets/enemies"
PROJECTILE_FOLDER = "assets/projectiles"
SOUND_FOLDER = "assets/sounds"
SWORD_SFX_FOLDER = "assets/sword_sfx"  # NEW: Folder for slash variations
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
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
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
    return None

# --- SWORD CURSOR ---
def create_sword_cursor():
    """
    Creates a placeholder sword cursor surface.
    Drop a 'sword_cursor.png' into assets/ to use a custom sprite instead.
    The pivot point (tip of the sword) is at the TOP-LEFT of the surface.
    """
    custom_path = os.path.join("assets", "sword_cursor.png")
    if os.path.exists(custom_path):
        try:
            img = pygame.image.load(custom_path).convert_alpha()
            return pygame.transform.scale(img, (120, 120))
        except:
            pass

    # --- Procedural placeholder sword ---
    size = 64
    surf = pygame.Surface((size, size), pygame.SRCALPHA)

    # Blade (thin white diagonal line from top-left tip to lower-right handle)
    blade_color = (220, 220, 255)
    guard_color = (200, 160, 50)
    handle_color = (120, 60, 20)

    # Blade: diagonal from (4,4) to (44,44)
    blade_points = [
        (4, 4),
        (10, 2),
        (46, 42),
        (44, 46),
    ]
    pygame.draw.polygon(surf, blade_color, blade_points)

    # Blade shine
    pygame.draw.line(surf, (255, 255, 255, 180), (5, 5), (40, 40), 1)

    # Guard (crossguard)
    guard_points = [
        (38, 36),
        (50, 28),
        (52, 32),
        (40, 40),
        (48, 52),
        (44, 54),
        (36, 42),
    ]
    pygame.draw.polygon(surf, guard_color, guard_points)

    # Handle
    handle_points = [
        (44, 46),
        (48, 42),
        (60, 58),
        (56, 62),
    ]
    pygame.draw.polygon(surf, handle_color, handle_points)

    return surf


def draw_sword_cursor(screen, sword_surf, pointer, velocity):
    """Draw the sword cursor at the pointer position, rotated to match movement direction."""
    if pointer is None:
        return

    # Tint based on velocity (yellow = slow, red = fast slash)
    if velocity > 15:
        tint = (255, 80, 80, 220)
    else:
        tint = (255, 255, 180, 200)

    # Apply tint via copy
    tinted = sword_surf.copy()
    tinted.fill(tint, special_flags=pygame.BLEND_RGBA_MULT)

    # Blit so the tip (top-left of surface) is at pointer position
    offset = 4  # small offset so tip aligns with finger
    screen.blit(tinted, (pointer[0] - offset, pointer[1] - offset))

    # Optional: small dot at exact tip for precision
    pygame.draw.circle(screen, (255, 255, 255), pointer, 3)


# --- PARTICLE SYSTEM (OPTIMIZED) ---
class ParticleSystem:
    """Pre-rendered particle system for better performance"""
    def __init__(self):
        self.particles = []
        self.particle_surfaces = {}
        for size in range(4, 13):
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 255, 255), (size, size), size)
            self.particle_surfaces[size] = surf

    def add_particles(self, x, y, color, count=20):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(3, 10)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed - 2
            size = random.randint(4, 12)
            self.particles.append([x, y, vx, vy, size, 40, color])

    def update_and_draw(self, surf):
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += 0.4
            p[5] -= 1
            if p[5] <= 0:
                self.particles.remove(p)
            else:
                size = int(p[4])
                if size in self.particle_surfaces:
                    alpha = int(255 * (p[5] / 40))
                    temp_surf = self.particle_surfaces[size].copy()
                    temp_surf.fill((*p[6], alpha), special_flags=pygame.BLEND_RGBA_MULT)
                    surf.blit(temp_surf, (int(p[0] - size), int(p[1] - size)), special_flags=pygame.BLEND_ALPHA_SDL2)


# --- PROJECTILE CLASS ---
class Projectile:
    def __init__(self, x, y, target_x, target_y, speed=8):
        self.x = x
        self.y = y
        dx = target_x - x
        dy = target_y - y
        dist = math.hypot(dx, dy)
        self.vx = (dx / dist) * speed if dist > 0 else 0
        self.vy = (dy / dist) * speed if dist > 0 else 0
        self.radius = 15
        self.active = True

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

        spawn_direction = random.choice(["bottom", "left", "right"])

        if spawn_direction == "bottom":
            self.x = random.randint(100, WINDOW_SIZE[0] - 100)
            self.y = WINDOW_SIZE[1] + 50
            self.target_y = random.randint(WINDOW_SIZE[1] - 400, WINDOW_SIZE[1] - 150)
            self.target_x = self.x
        elif spawn_direction == "left":
            self.x = -50
            self.y = random.randint(WINDOW_SIZE[1] - 400, WINDOW_SIZE[1] - 150)
            self.target_x = random.randint(200, WINDOW_SIZE[0] // 3)
            self.target_y = self.y
        else:
            self.x = WINDOW_SIZE[0] + 50
            self.y = random.randint(WINDOW_SIZE[1] - 400, WINDOW_SIZE[1] - 150)
            self.target_x = random.randint(WINDOW_SIZE[0] * 2 // 3, WINDOW_SIZE[0] - 200)
            self.target_y = self.y

        self.spawn_direction = spawn_direction
        self.speed = self.props["speed"]
        self.rising = True
        self.rise_timer = 0
        self.rise_duration = 50

        self.throw_timer = 0
        self.throw_cooldown = ENEMY_THROW_TIME * 60
        self.has_thrown = False

        self.state = "idle"
        self.state_timer = 0

        self.radius = 40
        self.sliced = False

        enemy_folder = os.path.join(ENEMY_FOLDER, enemy_type)
        sprite_size = (80, 80)

        self.sprites = {
            "idle": load_sprite_flexible(
                enemy_folder,
                ["idle.png", "Idle.png", "IDLE.png"],
                sprite_size, self.props["color"]
            ),
            "attack": load_sprite_flexible(
                enemy_folder,
                ["attack1.png", "Attack1.png", "ATTACK1.png", "attack.png", "Attack.png"],
                sprite_size, self.props["color"]
            ),
            "hurt": load_sprite_flexible(
                enemy_folder,
                ["take hit.png", "Take Hit.png", "take_hit.png", "Take_Hit.png", "TAKE_HIT.png", "hurt.png", "Hurt.png"],
                sprite_size, (255, 100, 100)
            ),
            "death": load_sprite_flexible(
                enemy_folder,
                ["death.png", "Death.png", "DEATH.png"],
                sprite_size, (100, 100, 100)
            )
        }

        self.current_sprite = self.sprites["idle"]

    def set_state(self, new_state, duration=0):
        self.state = new_state
        self.state_timer = duration
        self.current_sprite = self.sprites.get(new_state, self.sprites["idle"])

    def update(self):
        if self.state_timer > 0:
            self.state_timer -= 1
            if self.state_timer == 0:
                self.set_state("idle")

        if self.rising:
            self.rise_timer += 1
            progress = self.rise_timer / self.rise_duration

            if self.spawn_direction == "bottom":
                self.y = WINDOW_SIZE[1] + 50 - (progress * (WINDOW_SIZE[1] + 50 - self.target_y))
            elif self.spawn_direction == "left":
                self.x = -50 + (progress * (self.target_x + 50))
            else:
                self.x = WINDOW_SIZE[0] + 50 - (progress * (WINDOW_SIZE[0] + 50 - self.target_x))

            if self.rise_timer >= self.rise_duration:
                self.rising = False
                self.x = self.target_x
                self.y = self.target_y
        else:
            self.throw_timer += 1

    def should_throw(self):
        return not self.rising and self.throw_timer >= self.throw_cooldown and not self.has_thrown

    def throw(self, proj_speed=8):
        self.has_thrown = True
        self.set_state("attack", 30)
        return Projectile(self.x, self.y, WINDOW_SIZE[0] // 2, 50, speed=proj_speed)

    def take_damage(self):
        self.hp -= 1
        if self.hp > 0:
            self.set_state("hurt", 15)
            return False
        else:
            self.set_state("death", 30)
            return True

    def draw(self, surf):
        if not self.sliced:
            surf.blit(self.current_sprite, (int(self.x - 40), int(self.y - 40)))

            if self.max_hp > 1:
                bar_width = 60
                bar_height = 6
                hp_ratio = self.hp / self.max_hp
                pygame.draw.rect(surf, DARK_RED, (int(self.x - bar_width//2), int(self.y - 55), bar_width, bar_height))
                pygame.draw.rect(surf, RED, (int(self.x - bar_width//2), int(self.y - 55), int(bar_width * hp_ratio), bar_height))

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

    def reset(self):
        """Reset trail and position so gesture detection starts fresh"""
        self.prev_x, self.prev_y = 0, 0
        self.trail.clear()
        self.velocity = 0


# --- SOUND MANAGER ---
class SoundManager:
    def __init__(self):
        pygame.mixer.init()
        self.sounds = {}
        self.slash_sounds = []
        self.load_assets()

    def load_assets(self):
        sound_files = {
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

        slash_variants = [
            "slash.wav", "Slash.wav", "SLASH.wav",
            "slash1.wav", "Slash1.wav", "SLASH1.wav",
            "slash2.wav", "Slash2.wav", "SLASH2.wav",
            "slash3.wav", "Slash3.wav", "SLASH3.wav"
        ]

        for variant in slash_variants:
            path = os.path.join(SWORD_SFX_FOLDER, variant)
            if os.path.exists(path):
                try:
                    sound = pygame.mixer.Sound(path)
                    sound.set_volume(0.5)
                    self.slash_sounds.append(sound)
                except:
                    pass

        if not self.slash_sounds:
            for variant in slash_variants:
                path = os.path.join(SOUND_FOLDER, variant)
                if os.path.exists(path):
                    try:
                        sound = pygame.mixer.Sound(path)
                        sound.set_volume(0.5)
                        self.slash_sounds.append(sound)
                    except:
                        pass

        music_variants = ["battle.mp3", "Battle.wav", "BATTLE.wav", "music.wav", "Music.wav"]
        for variant in music_variants:
            music_path = os.path.join(MUSIC_FOLDER, variant)
            if os.path.exists(music_path):
                try:
                    pygame.mixer.music.load(music_path)
                    break
                except:
                    pass

    def play(self, name):
        if name == "SLASH":
            if self.slash_sounds:
                random.choice(self.slash_sounds).play()
        elif name in self.sounds:
            self.sounds[name].play()

    def play_music(self):
        try:
            pygame.mixer.music.play(-1)
        except:
            pass


# --- GAME OVER / RESTART SCREEN ---
class RestartScreen:
    """
    Shown when the game is over.
    Player restarts by performing a fast horizontal slash gesture.
    SPACE key still works as fallback.
    """
    SLASH_VELOCITY_THRESHOLD = 20   # Minimum speed to count as a slash
    SLASH_DISTANCE_THRESHOLD = 200  # Minimum horizontal distance for the gesture

    def __init__(self, score, high_score, font_large, font_medium, font_small):
        self.score = score
        self.high_score = high_score
        self.font_large = font_large
        self.font_medium = font_medium
        self.font_small = font_small

        # For gesture detection
        self.gesture_start_x = None
        self.gesture_start_y = None
        self.gesture_active = False
        self.gesture_cooldown = 60          # Frames before a new gesture can register (avoid instant re-trigger)
        self.gesture_cooldown_timer = 90    # Start with a brief delay so the game-over screen has time to show
        self.slash_registered = False

        # Visual feedback
        self.slash_flash_timer = 0
        self.hint_bob_timer = 0

    def update(self, pointer, velocity, is_reliable, prev_pointer):
        """Returns True if player wants to restart."""
        self.hint_bob_timer += 1

        if self.slash_flash_timer > 0:
            self.slash_flash_timer -= 1

        if self.gesture_cooldown_timer > 0:
            self.gesture_cooldown_timer -= 1
            return False

        if not is_reliable or pointer is None or prev_pointer is None:
            self.gesture_active = False
            self.gesture_start_x = None
            return False

        # Begin tracking a new gesture
        if not self.gesture_active:
            if velocity > self.SLASH_VELOCITY_THRESHOLD:
                self.gesture_active = True
                self.gesture_start_x = prev_pointer[0]
                self.gesture_start_y = prev_pointer[1]
        else:
            # Check if slash has covered enough horizontal distance
            horizontal_dist = abs(pointer[0] - self.gesture_start_x)
            vertical_dist = abs(pointer[1] - self.gesture_start_y)

            if horizontal_dist > self.SLASH_DISTANCE_THRESHOLD and horizontal_dist > vertical_dist * 1.5:
                # It's a horizontal slash!
                self.slash_flash_timer = 30
                self.gesture_cooldown_timer = self.gesture_cooldown
                self.gesture_active = False
                self.gesture_start_x = None
                return True

            # Reset if hand slows down without completing gesture
            if velocity < 5:
                self.gesture_active = False
                self.gesture_start_x = None

        return False

    def draw(self, screen, trail, pointer):
        overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))

        # Flash effect on successful slash
        if self.slash_flash_timer > 0:
            flash_surf = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            alpha = int(180 * (self.slash_flash_timer / 30))
            flash_surf.fill((255, 255, 255, alpha))
            screen.blit(flash_surf, (0, 0))

        cx = WINDOW_SIZE[0] // 2

        game_over_text = self.font_large.render("GAME OVER", True, RED)
        score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
        high_text = self.font_small.render(f"High Score: {self.high_score}", True, YELLOW)

        screen.blit(game_over_text, (cx - game_over_text.get_width() // 2, 200))
        screen.blit(score_text, (cx - score_text.get_width() // 2, 300))
        screen.blit(high_text, (cx - high_text.get_width() // 2, 360))

        # Animated "SLASH TO RESTART" hint
        bob = int(math.sin(self.hint_bob_timer * 0.07) * 8)

        if self.gesture_cooldown_timer > 0:
            # Show "get ready" during cooldown
            ready_text = self.font_small.render("Get ready...", True, (180, 180, 180))
            screen.blit(ready_text, (cx - ready_text.get_width() // 2, 460 + bob))
        else:
            # Pulsing color for slash hint
            pulse = int(abs(math.sin(self.hint_bob_timer * 0.05)) * 200 + 55)
            hint_color = (pulse, pulse, 80)
            slash_hint = self.font_medium.render("⚔  SLASH to Restart  ⚔", True, hint_color)
            screen.blit(slash_hint, (cx - slash_hint.get_width() // 2, 460 + bob))

            # Draw slash arrow guide
            arrow_y = 560 + bob
            arrow_color = (hint_color[0], hint_color[1], hint_color[2], 180)
            arrow_surf = pygame.Surface((400, 40), pygame.SRCALPHA)
            pygame.draw.line(arrow_surf, (200, 200, 100, 200), (20, 20), (380, 20), 4)
            pygame.draw.polygon(arrow_surf, (200, 200, 100, 220), [(360, 5), (395, 20), (360, 35)])
            screen.blit(arrow_surf, (cx - 200, arrow_y))

        # Fallback reminder
        space_text = self.font_small.render("(or press SPACE)", True, (120, 120, 120))
        screen.blit(space_text, (cx - space_text.get_width() // 2, 620))

        # Draw active gesture trail on this screen too (from trail deque passed in)
        if len(trail) > 1:
            trail_surf = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            for i in range(len(trail) - 1):
                alpha = int(200 * (i / len(trail)))
                thickness = int(12 * (i / len(trail))) + 3
                pygame.draw.line(trail_surf, (100, 200, 255, alpha), trail[i], trail[i + 1], thickness)
            screen.blit(trail_surf, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)


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
        self.spawn_rate = 50
        self.min_spawn_rate = 20

        self.game_over = False
        self.high_score = 0

        self.death_animations = []

        self.background = load_background()
        self.bg_color = (20, 15, 30)

        # Restart screen (created when game ends)
        self.restart_screen = None

        # --- TIME-BASED DIFFICULTY ---
        # Wave fires every 10 seconds (600 frames @ 60fps)
        self.frame_count = 0
        self.difficulty_level = 0
        self.difficulty_interval = 600     # 10 seconds between waves
        self.next_spike_frame = 600

        # Live difficulty values — these get hammered each wave
        self.throw_time_mult = 1.0         # Shrinks enemy throw cooldown aggressively
        self.projectile_speed_mult = 1.0   # Projectiles fly faster each wave
        self.max_enemies = MAX_ENEMIES_ON_SCREEN

        # On-screen alert
        self.spike_alert_timer = 0
        self.spike_alert_level = 0

        self.sound.play_music()

    def spawn_enemy(self):
        if len(self.enemies) >= self.max_enemies:
            return

        if self.difficulty_level < 2:
            enemy_type = random.choice(["bandit", "ninja"])
        elif self.difficulty_level < 4:
            enemy_type = random.choice(["bandit", "ninja", "samurai"])
        else:
            enemy_type = random.choice(["bandit", "ninja", "samurai", "oni"])

        e = Enemy(enemy_type)
        # Apply difficulty multipliers to this enemy's throw cooldown
        e.throw_cooldown = max(60, int(e.throw_cooldown * self.throw_time_mult))
        self.enemies.append(e)

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
                        self.particle_system.add_particles(enemy.x, enemy.y, (200, 0, 0), 15)
                        self.death_animations.append((enemy, 30))
                    else:
                        self.sound.play("HIT")

    def check_projectile_hit(self):
        for proj in self.projectiles[:]:
            if proj.y < 100:
                self.lives -= 1
                self.combo = 0
                self.sound.play("DAMAGE")
                self.particle_system.add_particles(proj.x, proj.y, (255, 100, 0), 20)
                proj.active = False

                if self.lives <= 0:
                    self.game_over = True
                    self.high_score = max(self.high_score, self.score)
                    # Create restart screen
                    self.restart_screen = RestartScreen(
                        self.score, self.high_score,
                        self.font_large, self.font_medium, self.font_small
                    )

    def apply_difficulty_spike(self):
        """
        Called every 10 seconds. Aggressive scaling every wave:
          Wave 1: throw 25% faster, projectiles 20% faster
          Wave 2: another 25% faster charge, +1 enemy cap
          Wave 3: throw at ~42% original speed, projectiles 60% faster
          Wave 4+: near-instant charge, bullet hell projectiles
        """
        self.difficulty_level += 1
        lvl = self.difficulty_level

        # Throw cooldown: multiplicative 25% shrink per wave — felt immediately
        # Wave 1: 0.75x  Wave 2: 0.56x  Wave 3: 0.42x  Wave 4: 0.32x  floor: 0.25x
        self.throw_time_mult = max(0.25, self.throw_time_mult * 0.75)

        # Projectile speed: +20% additive per wave, cap at 2.5x
        self.projectile_speed_mult = min(2.5, self.projectile_speed_mult + 0.20)

        # Spawn rate: drop 8 frames per wave (was 6)
        self.spawn_rate = max(self.min_spawn_rate, self.spawn_rate - 8)

        # Max enemies: +1 per wave from wave 2 onward
        if lvl >= 2:
            self.max_enemies = min(MAX_ENEMIES_ON_SCREEN + 8, self.max_enemies + 1)

        # Immediately tighten existing enemies on screen so you feel it NOW
        for enemy in self.enemies:
            if not enemy.has_thrown:
                enemy.throw_cooldown = max(45, int(enemy.throw_cooldown * 0.80))

        # Trigger on-screen warning
        self.spike_alert_timer = 180
        self.spike_alert_level = lvl

        print(f"[DIFFICULTY] Wave {lvl} — throw_mult={self.throw_time_mult:.2f}, "
              f"proj_speed={self.projectile_speed_mult:.2f}x, "
              f"spawn_rate={self.spawn_rate}, max_enemies={self.max_enemies}")

    def update(self, pointer, velocity, is_reliable, prev_pointer):
        if self.game_over:
            return

        # --- Time-based difficulty spike ---
        self.frame_count += 1
        if self.frame_count >= self.next_spike_frame:
            self.apply_difficulty_spike()
            self.next_spike_frame += self.difficulty_interval

        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_enemy()
            self.spawn_timer = 0
            self.spawn_rate = max(self.min_spawn_rate, self.spawn_rate - DIFFICULTY_INCREASE_RATE)

        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.should_throw():
                proj_speed = 8 * self.projectile_speed_mult
                projectile = enemy.throw(proj_speed=proj_speed)
                self.projectiles.append(projectile)
                self.sound.play("THROW")

        updated_deaths = []
        for enemy, timer in self.death_animations:
            timer -= 1
            if timer <= 0:
                if enemy in self.enemies:
                    self.enemies.remove(enemy)
            else:
                updated_deaths.append((enemy, timer))
        self.death_animations = updated_deaths

        for proj in self.projectiles[:]:
            proj.update()
            if not proj.active:
                self.projectiles.remove(proj)

        self.check_projectile_hit()

        if self.combo_timer > 0:
            self.combo_timer -= 1
            if self.combo_timer == 0:
                self.combo = 0

        if pointer and prev_pointer and is_reliable:
            self.check_slice(pointer[0], pointer[1], prev_pointer[0], prev_pointer[1], velocity)

    def draw(self, screen):
        if self.background:
            screen.blit(self.background, (0, 0))
        else:
            screen.fill(self.bg_color)

        for enemy in self.enemies:
            enemy.draw(screen)

        for proj in self.projectiles:
            proj.draw(screen)

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

        # --- Difficulty spike alert ---
        if self.spike_alert_timer > 0:
            self.spike_alert_timer -= 1
            progress = self.spike_alert_timer / 180  # 1.0 → 0.0

            # Pulsing red border flash
            if self.spike_alert_timer > 120:
                border_alpha = int(180 * (1.0 - progress) * 2)
                border_surf = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
                pygame.draw.rect(border_surf, (220, 0, 0, border_alpha), (0, 0, WINDOW_SIZE[0], WINDOW_SIZE[1]), 18)
                screen.blit(border_surf, (0, 0))

            # "WAVE X — DANGER!" text fades out
            text_alpha = int(255 * min(1.0, progress * 3))
            if text_alpha > 10:
                label = f"— WAVE {self.spike_alert_level} —"
                sub_label = "ENEMIES ARE FASTER!"
                alert_text = self.font_large.render(label, True, RED)
                sub_text = self.font_medium.render(sub_label, True, (255, 120, 0))

                for txt, y_pos in [(alert_text, WINDOW_SIZE[1] // 2 - 60),
                                   (sub_text,  WINDOW_SIZE[1] // 2 + 20)]:
                    s = txt.copy()
                    s.set_alpha(text_alpha)
                    screen.blit(s, (WINDOW_SIZE[0] // 2 - s.get_width() // 2, y_pos))

        # Difficulty level indicator (small, top-center)
        if self.difficulty_level > 0:
            wave_txt = self.font_small.render(f"Wave {self.difficulty_level}", True, (180, 100, 100))
            screen.blit(wave_txt, (WINDOW_SIZE[0] // 2 - wave_txt.get_width() // 2, 20))


# --- MAIN ---
def main():
    pygame.init()
    pygame.mouse.set_visible(False)  # Hide system cursor — we draw our own sword

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

    PROCESSING_WIDTH = 320 if DEV_MODE else 480
    PROCESSING_HEIGHT = 240 if DEV_MODE else 360

    controller = HandController()
    sound = SoundManager()
    game = GameScene(sound)

    # Build sword cursor (procedural placeholder — swap with a sprite via assets/sword_cursor.png)
    sword_cursor = create_sword_cursor()

    prev_pointer = None
    result = None

    print("=== SAMURAI SLASH ===")
    print("Swipe to slice enemies before they throw!")
    print("Press SPACE to restart | Press Q to quit")
    print("On Game Over: perform a horizontal slash gesture to restart!")

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
                cap.release()
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                if game.game_over:
                    high_score = game.high_score
                    game = GameScene(sound)
                    game.high_score = high_score
                    controller.reset()

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
            valid_hands = []

            for i, hand in enumerate(result.hand_landmarks):
                if USE_ROI and not is_hand_in_roi(hand, ROI_MARGIN):
                    continue

                confidence = result.handedness[i][0].score
                if confidence < 0.3:
                    continue

                z_depth = get_hand_z_depth(hand) if USE_Z_DEPTH else 0
                hand_size = calculate_hand_size(hand)
                valid_hands.append({
                    'hand': hand,
                    'confidence': confidence,
                    'z_depth': z_depth,
                    'size': hand_size,
                    'index': i
                })

            if len(valid_hands) == 1:
                selected_hand = valid_hands[0]['hand']
                selected_score = valid_hands[0]['confidence']
            elif len(valid_hands) > 1:
                if USE_Z_DEPTH:
                    best_hand = min(valid_hands, key=lambda h: h['z_depth'])
                else:
                    best_hand = max(valid_hands, key=lambda h: h['size'])
                selected_hand = best_hand['hand']
                selected_score = best_hand['confidence']

        pointer, velocity, is_reliable = None, 0, False
        if selected_hand:
            pointer, velocity, is_reliable = controller.process(
                selected_hand, selected_score, WINDOW_SIZE[0], WINDOW_SIZE[1]
            )

        # --- GAME OVER: gesture restart check ---
        if game.game_over and game.restart_screen is not None:
            wants_restart = game.restart_screen.update(pointer, velocity, is_reliable, prev_pointer)
            if wants_restart:
                high_score = game.high_score
                game = GameScene(sound)
                game.high_score = high_score
                controller.reset()
        else:
            game.update(pointer, velocity, is_reliable, prev_pointer)

        # --- DRAW ---
        game.draw(screen)

        # If game over, draw restart screen overlay
        if game.game_over and game.restart_screen is not None:
            game.restart_screen.draw(screen, controller.trail, pointer)

        # Draw slash trail (only during active gameplay)
        if not game.game_over and len(controller.trail) > 1:
            trail_surf = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
            for i in range(len(controller.trail) - 1):
                alpha = int(200 * (i / len(controller.trail)))
                thickness = int(12 * (i / len(controller.trail))) + 3
                color = (100, 200, 255, alpha)
                pygame.draw.line(trail_surf, color, controller.trail[i], controller.trail[i + 1], thickness)
            screen.blit(trail_surf, (0, 0), special_flags=pygame.BLEND_ALPHA_SDL2)

        # Draw sword cursor (replaces the old green circle)
        if pointer and is_reliable:
            draw_sword_cursor(screen, sword_cursor, pointer, velocity)

        prev_pointer = pointer
        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()