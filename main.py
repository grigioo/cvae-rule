import argparse
import shutil
import keras
import tensorflow as tf

import config
from src.data.processor import DataProcessor
from src.data.filtration import filter_unique_rooms
from src.data.selection import select_random_samples
from src.data.augmentation import generate_and_save_augmented_rooms
from src.data.encoding import load_full_dataset

from src.models.layers import build_encoder, build_decoder, Sampling
from src.models.losses import CVAELoss
from src.models.cvae_base import CVAEBase
from src.models.cvae_rules import CVAERules

from src.generation.room_gen import generate_single_room, visualize_room_text, visualize_room_color
from src.generation.level_gen import generate_full_dungeon


def run_prepare_data():
    print("=== Step 1: Splitting full levels into 16x11 rooms ===")
    processor = DataProcessor(
        void_char=config.VOID_CHAR,
        default_room_dims=(config.ROOM_HEIGHT, config.ROOM_WIDTH),
    )
    raw_files = list(config.RAW_LEVELS_DIR.glob("*.txt"))
    if not raw_files:
        raise FileNotFoundError(f"No .txt level files found in {config.RAW_LEVELS_DIR}")

    for level_path in raw_files:
        processor.process_text_level_to_rooms(str(level_path), str(config.EXTRACTED_ROOMS_DIR))

    print("\n=== Step 2: Filtering duplicate rooms (MD5) ===")
    filter_unique_rooms(str(config.EXTRACTED_ROOMS_DIR), str(config.UNIQUE_ROOMS_DIR))

    print("\n=== Step 3: Selecting fixed subset of rooms ===")
    select_random_samples(
        str(config.UNIQUE_ROOMS_DIR),
        str(config.SELECTED_ROOMS_DIR),
        desired_samples=config.DESIRED_FIXED_SAMPLES,
    )

    print("\n=== Step 4: Augmenting dataset (flips & 180-rotation) ===")
    generate_and_save_augmented_rooms(
        str(config.SELECTED_ROOMS_DIR),
        str(config.AUGMENTED_ROOMS_DIR),
        overwrite_existing=True,
    )
    # Copy original selected rooms into augmented folder so they are included in training
    for file_path in config.SELECTED_ROOMS_DIR.glob("*.txt"):
        shutil.copy(file_path, config.AUGMENTED_ROOMS_DIR / file_path.name)

    print(f"\nData preparation completed. Final dataset saved to: {config.AUGMENTED_ROOMS_DIR}")


def run_train(mode: str):
    print(f"=== Loading dataset from {config.AUGMENTED_ROOMS_DIR} ===")
    X_data, Y_data = load_full_dataset(str(config.AUGMENTED_ROOMS_DIR))
    if len(X_data) == 0:
        raise RuntimeError("Dataset is empty. Run `python main.py --step prepare_data` first.")

    input_shape_x = (config.ROOM_HEIGHT, config.ROOM_WIDTH, config.NUM_TILES)
    input_shape_y = (config.NUM_DOOR_FEATURES,)

    encoder = build_encoder(input_shape_x, input_shape_y, config.LATENT_DIM)
    decoder = build_decoder(config.LATENT_DIM, input_shape_y, input_shape_x)

    if mode == "base":
        print("=== Training Base CVAE (without rule loss) ===")
        model = CVAEBase(encoder=encoder, decoder=decoder, cvae_loss_layer=CVAELoss())
        epochs = config.EPOCHS_BASE
        save_path = config.BASE_MODEL_PATH
    elif mode == "rules":
        print("=== Training Rule-Guided CVAE (with rule loss) ===")
        model = CVAERules(encoder=encoder, decoder=decoder)
        model.set_rule_flags(
            void=True,
            wall_integrity=True,
            object_placement=True,
            door_environment=True,
        )
        epochs = config.EPOCHS_RULES
        save_path = config.RULES_MODEL_PATH
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))
    model.fit(
        [X_data, Y_data],
        X_data,
        batch_size=config.BATCH_SIZE,
        epochs=epochs,
        validation_split=0.1,
    )

    model.save(str(save_path))
    print(f"Model saved successfully to: {save_path}")


def load_trained_decoder(mode: str):
    model_path = config.RULES_MODEL_PATH if mode == "rules" else config.BASE_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")

    loaded_model = keras.models.load_model(
        str(model_path),
        custom_objects={
            "Sampling": Sampling,
            "CVAELoss": CVAELoss,
            "CVAEBase": CVAEBase,
            "CVAERules": CVAERules,
        },
        safe_mode=False,
    )
    return loaded_model.decoder


def run_generate_room(mode: str, doors: list[int]):
    decoder = load_trained_decoder(mode)
    room_indices = generate_single_room(decoder, config.LATENT_DIM, doors)
    visualize_room_text(room_indices, config.IDX_TO_TILE)
    visualize_room_color(room_indices, title=f"Generated Room ({mode}) | Doors: {doors}")


def run_generate_level(mode: str):
    decoder = load_trained_decoder(mode)
    generate_full_dungeon(
        decoder_model=decoder,
        rooms_pool_save_dir=str(config.ROOM_POOL_DIR),
        dungeon_output_dir=str(config.DUNGEONS_DIR),
        dungeon_rows=config.DUNGEON_ROWS,
        dungeon_cols=config.DUNGEON_COLS,
        num_rooms_to_generate_per_config=config.NUM_ROOMS_PER_CONFIG,
        min_dungeon_rooms=config.MIN_DUNGEON_ROOMS,
        max_dungeon_rooms=config.MAX_DUNGEON_ROOMS,
        room_connection_probability=config.ROOM_CONNECTION_PROBABILITY,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CVAE Procedural Level Generator for 2D Roguelikes")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["prepare_data", "train", "gen_room", "gen_level"],
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rules",
        choices=["base", "rules"],
        help="Model variant: 'base' (no rules) or 'rules' (with rule-based loss).",
    )
    parser.add_argument(
        "--doors",
        type=int,
        nargs=4,
        default=[1, 0, 0, 1],
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        help="4 binary flags for door placement when generating a single room (e.g. --doors 1 0 1 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.step == "prepare_data":
        run_prepare_data()
    elif args.step == "train":
        run_train(args.mode)
    elif args.step == "gen_room":
        run_generate_room(args.mode, args.doors)
    elif args.step == "gen_level":
        run_generate_level(args.mode)
