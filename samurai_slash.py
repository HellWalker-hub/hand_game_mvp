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

# Distance optimization settings
DEV_MODE = True
DIGITAL_ZOOM = 1.5
IGNORE_BACKGROUND_HANDS = True

# Game balance
ENEMY_THROW_TIME = 3.0  # seconds before enemy throws projectile
DIFFICULTY_INCREASE_RATE = 0.02  # How fast spawn rate increases

# --- ASSET FOLDERS ---
# Create these folders and put your downloaded assets here:
ENEMY_FOLDER = "assets/enemies"  # Put enemy sprites here
PROJECTILE_FOLDER = "assets/projectiles"  # Put shuriken/kunai sprites here
SOUND_FOLDER = "assets/sounds"  # Put sound effects here
MUSIC_FOLDER = "assets/music"  # Put background music here

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

def load_image_or_placeholder(path, size, color):
    """Try to load image, create colored placeholder if not found"""
    if os.path.exists(path):
        img = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(img, size)
    else:
        # Create placeholder
        surf = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.circle(surf, color, (size[0]//2, size[1]//2), min(size)//2)
        return surf

# --- PARTICLE SYSTEM ---
class Particle:
    def __init__(self, x, y, color, velocity_range=(-5, 5)):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 10)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - 2
        self.color = color
        self.size = random.randint(4, 12)
        self.lifetime = 40
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.4
        self.lifetime -= 1
        self.size = max(1, self.size - 0.2)
        
    def draw(self, surf):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / 40))
            s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (int(self.size), int(self.size)), int(self.size))
            surf.blit(s, (int(self.x - self.size), int(self.y - self.size)))

# --- PROJECTILE CLASS ---
class Projectile:
    def __init__(self, x, y, target_x, target_y):
        self.x = x
        self.y = y
        # Calculate direction toward player
        dx = target_x - x
        dy = target_y - y
        dist = math.hypot(dx, dy)
        speed = 8
        self.vx = (dx / dist) * speed if dist > 0 else 0
        self.vy = (dy / dist) * speed if dist > 0 else 0
        self.radius = 15
        self.active = True
        
        # Try to load projectile image
        self.image = load_image_or_placeholder(
            os.path.join(PROJECTILE_FOLDER, "shuriken.png"),
            (30, 30),
            (50, 50, 50)
        )
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        # Remove if off screen
        if self.x < -50 or self.x > WINDOW_SIZE[0] + 50 or self.y < -50 or self.y > WINDOW_SIZE[1] + 50:
            self.active = False
            
    def draw(self, surf):
        surf.blit(self.image, (int(self.x - 15), int(self.y - 15)))

# --- ENEMY CLASS ---
class Enemy:
    # Enemy types with different properties
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
        
        # Random spawn position at bottom of screen
        self.x = random.randint(100, WINDOW_SIZE[0] - 100)
        self.y = WINDOW_SIZE[1] + 50
        self.target_y = random.randint(WINDOW_SIZE[1] - 250, WINDOW_SIZE[1] - 150)
        
        # Movement
        self.speed = self.props["speed"]
        self.rising = True
        self.rise_timer = 0
        self.rise_duration = 60  # frames to rise
        
        # Throw mechanics
        self.throw_timer = 0
        self.throw_cooldown = ENEMY_THROW_TIME * 60  # Convert to frames (60 fps)
        self.has_thrown = False
        
        self.radius = 40
        self.sliced = False
        
        # Try to load enemy sprite
        sprite_name = f"{enemy_type}.png"
        self.image = load_image_or_placeholder(
            os.path.join(ENEMY_FOLDER, sprite_name),
            (80, 80),
            self.props["color"]
        )
        
    def update(self):
        # Rise up animation
        if self.rising:
            self.rise_timer += 1
            progress = self.rise_timer / self.rise_duration
            self.y = WINDOW_SIZE[1] + 50 - (progress * (WINDOW_SIZE[1] + 50 - self.target_y))
            
            if self.rise_timer >= self.rise_duration:
                self.rising = False
                self.y = self.target_y
        else:
            # Count down to throw
            self.throw_timer += 1
            
    def should_throw(self):
        return not self.rising and self.throw_timer >= self.throw_cooldown and not self.has_thrown
    
    def throw(self):
        self.has_thrown = True
        # Return projectile aimed at top-center of screen (player position)
        return Projectile(self.x, self.y, WINDOW_SIZE[0] // 2, 50)
    
    def take_damage(self):
        self.hp -= 1
        return self.hp <= 0  # Returns True if enemy is killed
    
    def draw(self, surf):
        if not self.sliced:
            # Draw enemy sprite
            surf.blit(self.image, (int(self.x - 40), int(self.y - 40)))
            
            # Draw HP bar if multi-HP enemy
            if self.max_hp > 1:
                bar_width = 60
                bar_height = 6
                hp_ratio = self.hp / self.max_hp
                pygame.draw.rect(surf, DARK_RED, (int(self.x - bar_width//2), int(self.y - 55), bar_width, bar_height))
                pygame.draw.rect(surf, RED, (int(self.x - bar_width//2), int(self.y - 55), int(bar_width * hp_ratio), bar_height))
            
            # Draw throw indicator
            if not self.rising and not self.has_thrown:
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
        
        # Use index finger tip
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
            "SLASH": "slash.wav",
            "HIT": "hit.wav",
            "THROW": "throw.wav",
            "DAMAGE": "damage.wav"
        }
        
        for name, filename in sound_files.items():
            path = os.path.join(SOUND_FOLDER, filename)
            if os.path.exists(path):
                self.sounds[name] = pygame.mixer.Sound(path)
                self.sounds[name].set_volume(0.5)
        
        # Try to load background music
        music_path = os.path.join(MUSIC_FOLDER, "battle.wav")
        if os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
    
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
        self.particles = []
        
        self.score = 0
        self.lives = 3
        self.combo = 0
        self.combo_timer = 0
        
        # Spawn control
        self.spawn_timer = 0
        self.spawn_rate = 120  # Start: spawn every 2 seconds
        self.min_spawn_rate = 30  # Max difficulty: spawn every 0.5 seconds
        
        self.game_over = False
        self.high_score = 0
        
        self.sound.play_music()
    
    def spawn_enemy(self):
        # Random enemy type based on score (harder enemies appear later)
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
            if not enemy.sliced:
                # Line-to-circle collision
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
                        enemy.sliced = True
                        self.score += enemy.points
                        self.combo += 1
                        self.combo_timer = 60
                        self.sound.play("SLASH")
                        
                        # Particles
                        for _ in range(20):
                            self.particles.append(Particle(enemy.x, enemy.y, (200, 0, 0)))
                        
                        self.enemies.remove(enemy)
                    else:
                        self.sound.play("HIT")
    
    def check_projectile_hit(self):
        """Check if any projectile reached the player"""
        for proj in self.projectiles[:]:
            if proj.y < 100:  # Reached top of screen (player area)
                self.lives -= 1
                self.combo = 0
                self.sound.play("DAMAGE")
                
                # Explosion at top
                for _ in range(30):
                    self.particles.append(Particle(proj.x, proj.y, (255, 100, 0)))
                
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
            # Gradually increase difficulty
            self.spawn_rate = max(self.min_spawn_rate, self.spawn_rate - DIFFICULTY_INCREASE_RATE)
        
        # Update enemies
        for enemy in self.enemies[:]:
            enemy.update()
            
            # Check if enemy should throw
            if enemy.should_throw():
                projectile = enemy.throw()
                self.projectiles.append(projectile)
                self.sound.play("THROW")
        
        # Update projectiles
        for proj in self.projectiles[:]:
            proj.update()
            if not proj.active:
                self.projectiles.remove(proj)
        
        # Check projectile hits
        self.check_projectile_hit()
        
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
        # Dark background
        screen.fill((20, 15, 30))
        
        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(screen)
        
        # Draw projectiles
        for proj in self.projectiles:
            proj.draw(screen)
        
        # Draw particles
        for particle in self.particles:
            particle.draw(screen)
        
        # HUD
        score_text = self.font_medium.render(f"Score: {self.score}", True, YELLOW)
        screen.blit(score_text, (20, 20))
        
        # Lives
        for i in range(self.lives):
            pygame.draw.circle(screen, RED, (WINDOW_SIZE[0] - 50 - i * 50, 40), 18)
        
        # Combo
        if self.combo > 1:
            combo_text = self.font_large.render(f"{self.combo}x COMBO!", True, YELLOW)
            alpha = min(255, self.combo_timer * 4)
            combo_surf = pygame.Surface(combo_text.get_size(), pygame.SRCALPHA)
            combo_surf.blit(combo_text, (0, 0))
            combo_surf.set_alpha(alpha)
            screen.blit(combo_surf, (WINDOW_SIZE[0] // 2 - combo_text.get_width() // 2, 120))
        
        # Game over
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
    
    # MediaPipe setup
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    # Camera setup
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
    
    PROCESSING_WIDTH = 480 if DEV_MODE else 640
    PROCESSING_HEIGHT = 360 if DEV_MODE else 480
    
    controller = HandController()
    sound = SoundManager()
    game = GameScene(sound)
    
    prev_pointer = None
    result = None
    
    print("=== SAMURAI SLASH ===")
    print("Controls: Swipe to slice enemies")
    print("Press SPACE to restart")
    print("Press Q to quit")
    
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
        
        pointer, velocity, is_reliable = None, 0, False
        if selected_hand:
            pointer, velocity, is_reliable = controller.process(
                selected_hand, selected_score, WINDOW_SIZE[0], WINDOW_SIZE[1]
            )
        
        # Update and draw game
        game.update(pointer, velocity, is_reliable, prev_pointer)
        game.draw(screen)
        
        # Draw slash trail
        if len(controller.trail) > 1:
            for i in range(len(controller.trail) - 1):
                alpha = int(200 * (i / len(controller.trail)))
                thickness = int(12 * (i / len(controller.trail))) + 3
                s = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
                pygame.draw.line(s, (100, 200, 255, alpha), controller.trail[i], controller.trail[i + 1], thickness)
                screen.blit(s, (0, 0))
        
        # Draw cursor
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