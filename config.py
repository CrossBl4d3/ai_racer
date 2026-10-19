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
NUM_CHECKPOINTS = 40          # gates on the oval (other tracks space gates by GATE_SPACING)
GATE_SPACING = 30             # pixels between gates on spline tracks (smaller = smoother walls)
DEFAULT_TRACK = "oval"
# --- Random tracks (8F) ---
RANDOM_TRACK_RADII = (390, 250)      # size of the base ellipse (x, y) in pixels
RANDOM_TRACK_POINTS = (9, 16)         # how many control points (fewest, most)
RANDOM_TRACK_WOBBLE = (0.1, 0.55)    # how far each point may be pushed in/out (share of the radius)
RANDOM_TRACK_DENT_CHANCE = 0.5       # chance of one deep dent (a hairpin-like bend)
RANDOM_TRACK_DENT_DEPTH = (0.4, 0.7)   # a dented point keeps this share of its radius
RANDOM_TRACK_WIDTH = (75, 110)       # road width range in pixels
RANDOM_TRACK_MAX_ATTEMPTS = 200      # shapes tried per seed before giving up

# --- Race ---
LAP_TIME_LIMIT = 60           # seconds
WALL_MODE = "solid"           # "solid" = walls block the car, "reset" = crash sends you back to start
ALLOW_REVERSE = True          # humans can reverse; the AI will train without it

# --- Neural network ---
HIDDEN_SIZES = [128, 128]     # neurons in each hidden layer

# --- DQN learning ---
AGENT_BACKEND = "torch"       # "torch" or "numpy": which of your two networks to train
GAMMA = 0.99                  # how much future rewards count
LEARNING_RATE = 5e-4
LEARNING_RATE_END = 5e-5       # learning rate shrinks to this by the end of training
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
WARMUP_STEPS = 5_000          # experiences collected before learning starts
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 100_000
TARGET_UPDATE_EVERY = 1_000   # decisions between target-network refreshes
DOUBLE_DQN = True
GRAD_CLIP = 10.0

# --- Watching and racing ---
HUMAN_RECORD = 5.68           # seconds: your best lap on Track 1
HUMAN_RECORDS = {             # your best lap per track (fill these in as you set them)
    "oval": 5.68,
    "bean": 16.21,
    "hairpin": 11.25,
    "wiggle": 25.45,
}
AI_CAR_HUE_SHIFT = 195        # turns the Kenney car blue for the AI in race mode
CAR_COLOURS = {               # name -> hue shift applied to the orange Kenney car
    "red": -25, "blue": 195, "green": 95, "purple": 255,
    "gold": 22, "pink": 305, "cyan": 160,
}
TORCH_CAR_COLOUR = "red"      # duel colour for PyTorch models (.pt)
NUMPY_CAR_COLOUR = "gold"     # duel colour for YOUR from-scratch models (.npz)

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
COLOR_Q_BAR = (90, 140, 255)
COLOR_Q_CHOSEN = (255, 215, 0)

# --- AI environment ---
RAY_ANGLES = [-90, -60, -30, 0, 30, 60, 90]   # degrees, relative to where the car faces
RAY_LENGTH = 300              # pixels: how far each sensor can see
FRAME_SKIP = 4                # physics steps per AI decision
EPISODE_LAPS = 3              # laps needed to "finish" an episode
NO_PROGRESS_LIMIT = 3         # seconds without a new checkpoint before the episode is cut off

REWARD_CHECKPOINT = 1.0
REWARD_LAP = 10.0
REWARD_CRASH = -10.0
REWARD_STEP = -0.01           # per decision: a small push to hurry

COLOR_RAY = (0, 220, 255)
COLOR_RAY_HIT = (255, 80, 80)

# --- Training run ---
TRAIN_STEPS = 150_000         # total decisions per training run
EVAL_EVERY = 10_000           # decisions between greedy test drives
PRINT_EVERY = 5_000           # decisions between progress lines
RANDOM_START = True           # training episodes start at a random gate
START_JITTER_POS = 8          # pixels: random nudge to the start position
START_JITTER_ANGLE = 10       # degrees: random nudge to the start angle
EVAL_DRIVES = 5               # test drives per evaluation, from evenly spaced gates


# --- Multi-track training (Phase 8) ---
TRAIN_TRACKS = ["oval", "bean", "hairpin", "wiggle",
                "oval_mirror", "bean_mirror", "hairpin_mirror", "wiggle_mirror"]
TEST_TRACK = "technical"      # NEVER trained on: the final exam
MULTI_TRAIN_STEPS = 300_000   # more tracks to learn = more practice needed
EVAL_DRIVES_PER_TRACK = 2     # multi-track test drives: this many per training track
FINE_TUNE_EPSILON = 0.3       # exploration starts here when continuing from a trained model
# --- Corner fixes (Phase 8) ---
LOOKAHEAD_GATES = [4, 8]      # extra inputs: direction to the gate this many gates ahead ([] = old 9 inputs)
MIRROR_AUGMENT = True         # also learn from the mirror image of every experience
HARD_START_FRACTION = 0.5     # share of training starts placed just before recent crash spots
HARD_START_BACK = (3, 10)     # start between this many gates before the crash spot
CRASH_MEMORY = 0.995          # older crashes fade: each new crash shrinks all old counts a little
FAIR_START_PROBE_SECONDS = 1.5   # fair-start dry run: how long each steering test lasts
FAIR_START_GATES = 2          # ...and how many gates it must get through to count as winnable
PICK_FINISH_TOLERANCE = 1     # pick_best.py: snapshots this many finishes short of the best still count