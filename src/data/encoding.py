import os
import numpy as np
import config

# Convert a single ASCII tile character into a 10-dimensional one-hot vector
def one_hot_encode(tile_char: str) -> np.ndarray:
    one_hot_vector = np.zeros(config.NUM_TILES, dtype=np.float32)
    idx = config.TILE_TO_IDX.get(tile_char)
    if idx is not None:
        one_hot_vector[idx] = 1.0
    else:
        print(f"Предупреждение: Неизвестный символ тайла '{tile_char}'. Возвращен нулевой вектор.")
    return one_hot_vector


# Load a 16x11 room text file and encode it into a (16, 11, 10) float32 tensor
def load_and_preprocess_room(filepath: str) -> np.ndarray:
    room_grid_chars = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                room_grid_chars.append(list(stripped))

    current_height = len(room_grid_chars)
    current_width = len(room_grid_chars[0]) if current_height > 0 else 0

    if current_height != config.ROOM_HEIGHT or current_width != config.ROOM_WIDTH:
        raise ValueError(
            f"Размер комнаты в файле {filepath} ({current_height}x{current_width}) "
            f"не соответствует ожидаемому ({config.ROOM_HEIGHT}x{config.ROOM_WIDTH})."
        )

    encoded_room = np.zeros((config.ROOM_HEIGHT, config.ROOM_WIDTH, config.NUM_TILES), dtype=np.float32)
    for r in range(config.ROOM_HEIGHT):
        for c in range(config.ROOM_WIDTH):
            encoded_room[r, c] = one_hot_encode(room_grid_chars[r][c])

    return encoded_room


# Extract a 4-element binary vector [top, bottom, left, right] indicating door presence
# Checks two outer layers on each side because Zelda doors are inset by one wall tile
def extract_door_info(room_filepath: str) -> np.ndarray:
    has_door_top = 0
    has_door_bottom = 0
    has_door_left = 0
    has_door_right = 0

    with open(room_filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if "D" in lines[0] or "D" in lines[1]:
        has_door_top = 1

    if config.ROOM_HEIGHT >= 2 and ("D" in lines[config.ROOM_HEIGHT - 1] or "D" in lines[config.ROOM_HEIGHT - 2]):
        has_door_bottom = 1

    for r in range(config.ROOM_HEIGHT):
        if config.ROOM_WIDTH >= 2:
            if lines[r][0] == "D" or lines[r][1] == "D":
                has_door_left = 1
            if lines[r][config.ROOM_WIDTH - 1] == "D" or lines[r][config.ROOM_WIDTH - 2] == "D":
                has_door_right = 1
        else:
            if lines[r][0] == "D":
                has_door_left = 1
            if lines[r][config.ROOM_WIDTH - 1] == "D":
                has_door_right = 1

    return np.array([has_door_top, has_door_bottom, has_door_left, has_door_right], dtype=np.float32)


# Walk a directory of room files and build training arrays X_data (rooms) and Y_data (doors)
def load_full_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    all_encoded_rooms = []
    all_door_info = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".txt"):
                filepath = os.path.join(root, file)
                try:
                    encoded_room = load_and_preprocess_room(filepath)
                    door_info = extract_door_info(filepath)
                    all_encoded_rooms.append(encoded_room)
                    all_door_info.append(door_info)
                except ValueError as e:
                    print(f"Ошибка при обработке файла {filepath}: {e}")
                except Exception as e:
                    print(f"Непредвиденная ошибка при обработке файла {filepath}: {e}")

    if all_encoded_rooms:
        x_data = np.array(all_encoded_rooms, dtype=np.float32)
        y_data = np.array(all_door_info, dtype=np.float32)
        print(f"Загружено {len(all_encoded_rooms)} комнат.")
        print(f"Размер X_data (комнаты): {x_data.shape}")
        print(f"Размер Y_data (информация о дверях): {y_data.shape}")
    else:
        print("Не найдено комнат для загрузки.")
        x_data = np.array([], dtype=np.float32)
        y_data = np.array([], dtype=np.float32)

    return x_data, y_data


# Decode a single one-hot vector back into its ASCII tile character
def one_hot_decode(one_hot_vector: np.ndarray) -> str:
    idx = int(np.argmax(one_hot_vector))
    return config.IDX_TO_TILE.get(idx, "?")


# Convert a 2D index grid or 3D one-hot room tensor back into a multiline ASCII string
def room_array_to_text(room_array: np.ndarray) -> str:
    if room_array.ndim == 3 and room_array.shape[2] == config.NUM_TILES:
        height, width, _ = room_array.shape
        text_grid = [
            "".join(one_hot_decode(room_array[r, c]) for c in range(width))
            for r in range(height)
        ]
    elif room_array.ndim == 2:
        height, width = room_array.shape
        text_grid = [
            "".join(config.IDX_TO_TILE.get(int(room_array[r, c]), "?") for c in range(width))
            for r in range(height)
        ]
    else:
        raise ValueError(
            f"Неподдерживаемый формат room_array для room_array_to_text. "
            f"Ожидается 2D или 3D массив, получен {room_array.ndim}D с формой {room_array.shape}"
        )
    return "\n".join(text_grid)
