from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
CAR_SPRITE_PATH = ASSETS_DIR / "car_red_1.png"
CAR_HUE_SHIFT = -25           # degrees: shifts the sprite from orange to red (used in Part 2G)
GRASS_TILE_PATH = ASSETS_DIR / "land_grass11.png"
TREE_PATH = ASSETS_DIR / "tree_small.png"
TIRES_PATH = ASSETS_DIR / "tires_red.png"

# --- Window ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60                      # frames per second when a human is playing
STEPS_PER_SECOND = 60         # physics steps per "game second" (for lap times)


# FROZEN for training. Human record: 5.68 s (40 checkpoints)
# --- Car ---
CAR_LENGTH = 32               # pixels (matches the sprite's shape)
CAR_WIDTH = 17
CAR_ACCELERATION = 0.15       # speed added per step while holding throttle
CAR_BRAKE = 0.30              # speed removed per step while braking
CAR_FRICTION = 0.05           # speed lost per step when coasting
CAR_MAX_SPEED = 6.0           # pixels per step (must stay below CAR_WIDTH)
CAR_TURN_RATE = 4.0           # degrees per step at full speed
CAR_REVERSE_ACCEL = 0.10      # speed added backwards per step while holding brake at a standstill
CAR_MAX_REVERSE_SPEED = 2.0   # reverse is slow on purpose


# FROZEN for training. Human record: 5.68 s (40 checkpoints)
# --- Track ---
TRACK_CENTER = (500, 350)
TRACK_RADIUS_X = 380          # half-width of the oval's centre line
TRACK_RADIUS_Y = 250          # half-height of the oval's centre line
ROAD_WIDTH = 110
NUM_CHECKPOINTS = 40

# --- Race ---
LAP_TIME_LIMIT = 60           # seconds
WALL_MODE = "solid"           # "solid" = walls block the car, "reset" = crash sends you back to start
ALLOW_REVERSE = True          # humans can reverse; the AI will train without it


# --- Colours (R, G, B) ---
COLOR_GRASS = (76, 140, 60)
COLOR_ROAD = (70, 70, 75)
COLOR_WALL = (235, 235, 235)
COLOR_CHECKPOINT = (255, 215, 0)
COLOR_NEXT_GATE = (0, 255, 120)
COLOR_CAR = (210, 40, 40)
COLOR_HITBOX = (0, 255, 255)
COLOR_HUD_TEXT = (255, 255, 255)
COLOR_MESSAGE = (255, 230, 80)
COLOR_CRASH = (255, 60, 60)
COLOR_KERB_A = (210, 40, 40)
COLOR_KERB_B = (245, 245, 245)