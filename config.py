from pathlib import Path

# Base project paths (resolved automatically relative to this file)
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

RAW_LEVELS_DIR = DATA_DIR / "raw"
EXTRACTED_ROOMS_DIR = DATA_DIR / "extracted_rooms"
UNIQUE_ROOMS_DIR = DATA_DIR / "unique_rooms"
SELECTED_ROOMS_DIR = DATA_DIR / "selected_rooms"
AUGMENTED_ROOMS_DIR = DATA_DIR / "augmented_rooms"

MODELS_DIR = ROOT_DIR / "saved_models"
OUTPUT_DIR = ROOT_DIR / "outputs"
ROOM_POOL_DIR = OUTPUT_DIR / "generated_room_pool"
DUNGEONS_DIR = OUTPUT_DIR / "generated_dungeons"

# Ensure critical directories exist
for directory in [RAW_LEVELS_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Room geometry and tile vocabulary
ROOM_HEIGHT = 16
ROOM_WIDTH = 11
VOID_CHAR = "-"

TILE_TO_IDX = {
    "F": 0,  # Floor
    "B": 1,  # Block / inner wall
    "M": 2,  # Monster
    "P": 3,  # Player / puzzle element
    "O": 4,  # Obstacle
    "I": 5,  # Item
    "D": 6,  # Door
    "S": 7,  # Secret / special / exit
    "W": 8,  # Outer wall
    "-": 9,  # Void / padding
}
IDX_TO_TILE = {v: k for k, v in TILE_TO_IDX.items()}
NUM_TILES = len(TILE_TO_IDX)

# Conditional features: [top, bottom, left, right]
NUM_DOOR_FEATURES = 4

# Dataset preparation settings
DESIRED_FIXED_SAMPLES = 100

# Model & training hyperparameters
LATENT_DIM = 32
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS_BASE = 200
EPOCHS_RULES = 500
RULE_LOSS_WEIGHT = 10.0

# Model artifact filenames
BASE_MODEL_PATH = MODELS_DIR / "best_cvae_model_no_rules.keras"
RULES_MODEL_PATH = MODELS_DIR / "best_cvae_model_with_rules.keras"

# Dungeon generation settings
DUNGEON_ROWS = 6
DUNGEON_COLS = 6
MIN_DUNGEON_ROOMS = 8
MAX_DUNGEON_ROOMS = 13
ROOM_CONNECTION_PROBABILITY = 0.5
NUM_ROOMS_PER_CONFIG = 50
