import numpy as np
import matplotlib.pyplot as plt
import config


# Sample a random latent vector z and decode a single 16x11 room for the given door condition
def generate_single_room(decoder_model, latent_dim: int, room_condition: list[int]) -> np.ndarray:
    z_sample = np.random.normal(size=(1, latent_dim)).astype(np.float32)
    condition_input = np.array([room_condition], dtype=np.float32)

    predicted_room_probabilities = decoder_model.predict([z_sample, condition_input], verbose=0)
    generated_room_indices = np.argmax(predicted_room_probabilities, axis=-1)
    return generated_room_indices[0]


# Print generated room tile grid as ASCII characters in the console
def visualize_room_text(room_indices: np.ndarray, int_to_char_map: dict[int, str] = config.IDX_TO_TILE):
    print("\nСгенерированная комната (текст):")
    for row in room_indices:
        print("".join(int_to_char_map[int(idx)] for idx in row))
    print("-" * (config.ROOM_WIDTH + 2))


# Render a color-coded 2D grid visualization of a single generated room
def visualize_room_color(room_indices: np.ndarray, title: str = "Generated Room"):
    colors = [
        [0.7, 0.7, 0.7],  # F: Floor
        [0.4, 0.4, 0.4],  # B: Block
        [0.8, 0.4, 0.0],  # M: Monster
        [0.5, 0.0, 0.5],  # P: Player / element
        [0.2, 0.6, 0.2],  # O: Obstacle
        [0.1, 0.3, 0.9],  # I: Item
        [0.6, 0.3, 0.0],  # D: Door
        [0.8, 0.0, 0.0],  # S: Secret / exit
        [0.2, 0.2, 0.2],  # W: Outer wall
        [0.0, 0.0, 0.0],  # -: Void
    ]

    colored_room = np.zeros((config.ROOM_HEIGHT, config.ROOM_WIDTH, 3))
    for r in range(config.ROOM_HEIGHT):
        for c in range(config.ROOM_WIDTH):
            tile_idx = int(room_indices[r, c])
            colored_room[r, c] = colors[tile_idx]

    plt.figure(figsize=(6, 6))
    plt.imshow(colored_room, interpolation="nearest")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.show()
