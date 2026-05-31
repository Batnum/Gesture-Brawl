import cv2
import numpy as np
import tensorflow as tf
import pickle
import pygame
import sys
import threading
import os
import random

# --- STANDALONE PATH ROUTING MANAGEMENT ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "new_hagrid_model.keras")
PKL_PATH = os.path.join(BASE_DIR, "hagrid_classes.pkl")

# Sprite folder configuration
SPRITES_DIR = os.path.join(BASE_DIR, "animations")
ASSET_DIR = os.path.join(BASE_DIR, "assets")
SOUNDS_DIR = os.path.join(BASE_DIR, "sounds")   

# --- 1. INITIALIZATION & CONFIG ---
sizemult = 1.2  # Global scaling factor for screen
pygame.init()
WIDTH, HEIGHT = int(1000 * sizemult), int(600 * sizemult)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Brawl")
pygame.display.set_icon(pygame.image.load(os.path.join(ASSET_DIR, "icon.png")))
effect = pygame.image.load(os.path.join(ASSET_DIR, "effect.png"))
effect = pygame.transform.scale(effect, (int(100*sizemult), int(100*sizemult)))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)
large_font = pygame.font.SysFont("Arial", 60)

#sound effects initialization
pygame.mixer.init()
backgroundmusic = pygame.mixer.music.load(os.path.join(SOUNDS_DIR, "backgroundsound.mp3"))
pygame.mixer.music.set_volume(0.2)  # Set volume to 10%

punch_sound = pygame.mixer.Sound(os.path.join(SOUNDS_DIR, "punch.wav"))
punch_sound.set_volume(0.3)  # Set volume to 30%
whoosh_sound = pygame.mixer.Sound(os.path.join(SOUNDS_DIR, "whoosh.wav"))
whoosh_sound.set_volume(0.3)  # Set volume to 30%
victory_sound = pygame.mixer.Sound(os.path.join(SOUNDS_DIR, "yay.wav"))
victory_sound.set_volume(0.5)  # Set volume to 50%
defeat_sound = pygame.mixer.Sound(os.path.join(SOUNDS_DIR, "womp.wav"))
defeat_sound.set_volume(0.5)  # Set volume to 50%

#booleans
game_ended = False  # Track if game has ended
is_punching = False  # Track if player is currently punching

# --- 2. MULTITHREADED GESTURE ENGINE ---
class GestureRecognizer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.models.load_model(MODEL_PATH)
        with open(PKL_PATH, "rb") as f:
            self.class_names = pickle.load(f)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        self.current_gesture = "Uncertain"
        self.confidence = 0.0
        self.running = True
        self.latest_frame = None

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.latest_frame = np.rot90(frame_rgb)
            
            final_img = cv2.resize(frame_rgb, (224,224))
            img_tensor = np.expand_dims(final_img, axis=0).astype(np.float32)
            
            predictions = self.model.predict(img_tensor, verbose=0)
            class_id = np.argmax(predictions)
            pred_confidence = predictions[0][class_id]
            
            if pred_confidence > 0.60:
                self.current_gesture = self.class_names[class_id]
                self.confidence = pred_confidence * 100
            else:
                self.current_gesture = "Uncertain"
                self.confidence = 0.0

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()

gesture_thread = GestureRecognizer()
gesture_thread.start()

# --- 3. GAME ENTITIES ---
class Fighter:
    def __init__(self, x, color, is_player=True):
        self.x = x
        self.base_y = 450*sizemult  # Ground level alignment
        self.y = self.base_y
        self.color = color
        self.is_player = is_player
        
        # Stats
        self.hp = 100
        
        # State Management
        self.state = "idle"  # idle, blocking, punching, crouching, running
        self.is_jumping = False
        self.jump_velocity = 0
        self.state_timer = 0
        
        # Directional facing toggle (Player defaults to right, enemy to left)
        self.facing_left = False if is_player else True
        self.animation_tick = 0
        
        # Sprite System Initialization
        self.sprites = {}
        self.load_sprites()

    def load_sprites(self):
        # Maps game states
        sprite_map = {
            "idle": ["idle.png"],
            "jump": ["jump.png"],
            "crouching": ["crouch.png"],
            "blocking": ["block.png"],
            "running": ["run1.png", "run2.png"],
            "punching": ["punch1.png", "punch2.png"]
        }
        
        for state, filenames in sprite_map.items():
            self.sprites[state] = []
            for file in filenames:
                path = os.path.join(SPRITES_DIR, file)
                # Retains native 256x256 dimensions
                img = pygame.image.load(path).convert_alpha()
                img = pygame.transform.scale(img, (int(256*sizemult), int(256*sizemult)))
                # color sprites
                img.fill(self.color, special_flags=pygame.BLEND_RGBA_MULT)
                self.sprites[state].append(img)

    def handle_input(self, gesture):
        if not self.is_player:
            return

        moved = False
        if gesture == "two_up_inverted": 
            self.move(-5*sizemult)
            self.facing_left = True
            moved = True
        elif gesture == "two_up": 
            self.move(5*sizemult)
            self.facing_left = False
            moved = True
        
        if gesture == "stop":
            self.state = "blocking"
        elif gesture == "fist":
            self.state = "punching"
        elif gesture == "dislike":
            self.state = "crouching"
        elif gesture == "like" and not self.is_jumping:
            self.is_jumping = True
            self.jump_velocity = -18
        else:
            self.state = "running" if moved else "idle"

    def move(self, dx):
        if self.state != "crouching":
            self.x = max(50, min(WIDTH - 50, self.x + dx))

    def update(self):
        self.animation_tick += 1
        if self.is_jumping:
            self.y += self.jump_velocity
            self.jump_velocity += 1  
            if self.y >= self.base_y:
                self.y = self.base_y
                self.is_jumping = False
                self.jump_velocity = 0


    def draw(self, surface):
        # Resolve active animation state 
        current_state = "jump" if self.is_jumping else self.state
        frames = self.sprites.get(current_state, self.sprites["idle"])
        
        # Alternates animation frames every 8 game ticks for run/punch sequences
        frame_idx = (self.animation_tick // 8) % len(frames) if len(frames) > 1 else 0
        current_frame = frames[frame_idx]
        
        # Mirror flip transformation engine
        if not self.facing_left:
            current_frame = pygame.transform.flip(current_frame, True, False)
            
        # Draw frame centered overhead using original base_y alignment metrics
        # (X position is centered; bottom of 256px frame sits flush with character foot coordinates)
        surface.blit(current_frame, (int(self.x - 128), int(self.y - 200)))


# --- 4. UPGRADED ENEMY AI LOGIC ---
def update_enemy_ai(enemy_char, player_char):

    distance = abs(player_char.x - enemy_char.x)
    
    # AI dynamically tracks orientation towards player position
    enemy_char.facing_left = player_char.x < enemy_char.x

    chance = random.random()
    # Attack Logic (When close enough)
    if distance < 65:
        if chance < 0.35:     # 35% chance to punch
            enemy_char.state = "punching"
        elif chance < 0.50:   # 15% chance to block
            enemy_char.state = "blocking"
    # Movement Tracking Logic
    else:
        if chance < 0.02 and not enemy_char.is_jumping:  # 2% chance to jump
            enemy_char.is_jumping = True
            enemy_char.jump_velocity = -18
        elif enemy_char.x < player_char.x and not enemy_char.is_jumping:
            enemy_char.move(3*sizemult)   
            enemy_char.state = "running"
        elif enemy_char.x > player_char.x and not enemy_char.is_jumping:
            enemy_char.move(-3*sizemult)  
            enemy_char.state = "running"


# Initialize State
def reset_game():
    global player, enemy, game_state, game_ended
    player = Fighter(250/sizemult, (150, 200, 255), is_player=True)
    enemy = Fighter(750*sizemult, (255, 150, 150), is_player=False)
    game_state = "PLAYING" # PLAYING, WON, LOST
    pygame.mixer.music.play(-1)  # Resume background music
    game_ended = False  # Reset game ended state

reset_game()

game_running = True
show_camera = False  # Track camera display state
pygame.mixer.music.play(-1)  # Loop background music indefinitely

def draw_hit_effect(surface, x, y):
    # Simple hit effect that flashes and fades out
    effect = pygame.image.load(os.path.join(ASSET_DIR, "effect.png"))
    effect = pygame.transform.scale(effect, (int(100*sizemult), int(100*sizemult)))
    for alpha in range(255, 0, -15):
        effect.set_alpha(alpha)
        surface.blit(effect, (int(x - 50*sizemult), int(y - 50*sizemult)))

# --- 5. MAIN GAME LOOP ---
while game_running:
    clock.tick(60)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and game_state != "PLAYING":
                reset_game()
            if event.key == pygame.K_c:
                show_camera = not show_camera  # Toggle camera view

    active_gesture = gesture_thread.current_gesture
    cv_confidence = gesture_thread.confidence

    # --- DRAW BACKGROUND AND HUD ---
    screen.fill((50,100,200))
    pygame.draw.line(screen, (40, 150, 40), (0, 500*sizemult), (WIDTH, 500*sizemult), int(250*sizemult)) # Ground line

    # Render Players
    enemy.draw(screen)
    player.draw(screen)

    if game_state == "PLAYING":
        # Process Actions
        player.handle_input(active_gesture)
        update_enemy_ai(enemy, player)

        # Process Physics Updates
        player.update()
        enemy.update()

        # Combat Collision System Logic
        
        # --- PLAYER ATTACK LOGIC ---
        if player.state == "punching":
            # Only check and play audio once every 8 game ticks
            if player.animation_tick % 8 == 0:
                if abs(player.x - enemy.x) < 65 and not player.is_jumping:
                    punch_sound.play()
                    if enemy.state != "blocking" and enemy.state != "crouching":
                        draw_hit_effect(screen, enemy.x, enemy.y - 100)  # Visual hit effect on enemy
                        enemy.hp -= 3
                        if enemy.hp <= 0:
                            game_state = "WON"
                else:
                    whoosh_sound.play() # play whoosh sound effect for missed punch
                    
        # --- ENEMY ATTACK LOGIC ---            
        if enemy.state == "punching":
            # Only check and play audio once every 8 game ticks
            if enemy.animation_tick % 8 == 0:
                if abs(enemy.x - player.x) < 65 and not enemy.is_jumping:
                    punch_sound.play()
                    if player.state != "blocking" and player.state != "crouching":
                        player.hp -= 2
                        draw_hit_effect(screen, player.x, player.y - 100)  # Visual hit effect on player
                        if player.hp <= 0:
                            game_state = "LOST"
                else:
                    whoosh_sound.play() # play whoosh sound effect for missed punch

    # Health Bars Visual Rendering
    pygame.draw.rect(screen, (255, 0, 0), (50, 30, 300, 20))
    pygame.draw.rect(screen, (0, 255, 0), (50, 30, player.hp * 3, 20))
    pygame.draw.rect(screen, (255, 0, 0), (WIDTH - 350, 30, 300, 20))
    pygame.draw.rect(screen, (0, 255, 0), (WIDTH - 350, 30, enemy.hp * 3, 20))

    # Show Camera Overlays
    if show_camera and gesture_thread.latest_frame is not None:
        cam_surf = pygame.surfarray.make_surface(gesture_thread.latest_frame)
        cam_surf = pygame.transform.scale(cam_surf, (160, 120))
        screen.blit(cam_surf, (WIDTH // 2 - 80, 10))

    # Text UI Displays
    gesture_txt = font.render(f"Gesture: {active_gesture} ({cv_confidence:.1f}%)", True, (255, 255, 255))
    camera_toggle_txt = font.render("Press 'C' to Toggle Camera view", True, (120, 135, 155))
    restart_txt = font.render("Press 'R' to Retry", True, (120, 135, 155))

    screen.blit(gesture_txt, (50, 60))
    screen.blit(camera_toggle_txt, (50, 90))
    screen.blit(restart_txt, (50, 120))

    if game_state == "WON":
        win_txt = large_font.render("YOU WIN!", True, (0, 255, 0))
        screen.blit(win_txt, (WIDTH // 2 - 130, HEIGHT // 2 - 50))
        #play victory sound effect once
        if not game_ended:  # Ensure victory sound plays only once
            pygame.mixer.music.stop()  # Stop background music
            pygame.mixer.stop()  # Stop all sound effects to prevent overlap
            victory_sound.play()
            game_ended = True
    elif game_state == "LOST":
        lost_txt = large_font.render("GAME OVER", True, (255, 0, 0))
        screen.blit(lost_txt, (WIDTH // 2 - 160, HEIGHT // 2 - 50))
        #play defeat sound effect
        if not game_ended:  # Ensure defeat sound plays only once
            pygame.mixer.music.stop()  # Stop background music
            pygame.mixer.stop()  # Stop all sound effects to prevent overlap
            defeat_sound.play()
            game_ended = True
    pygame.display.flip()



gesture_thread.stop()
pygame.quit()
sys.exit()
