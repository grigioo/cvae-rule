import os
import config


# Slice full ASCII dungeon maps into non-empty 16x11 room text files
class DataProcessor:
    def __init__(self, void_char=config.VOID_CHAR, default_room_dims=(config.ROOM_HEIGHT, config.ROOM_WIDTH)):
        self.void_char = void_char
        self.default_room_rows = default_room_dims[0]
        self.default_room_cols = default_room_dims[1]

    # Read a full level ASCII map into a 2D character grid
    def _read_level_text(self, filepath: str) -> list[list[str]]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Level text file not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [list(line.strip()) for line in f if line.strip()]
        return lines

    # Pad grid dimensions with void characters so they divide evenly by 16x11 chunks
    def _pad_level_grid(self, level_grid: list[list[str]], target_rows: int, target_cols: int) -> list[list[str]]:
        current_rows = len(level_grid)
        if current_rows == 0:
            return []
        current_cols = max(len(row) for row in level_grid)

        padded_rows = ((current_rows + target_rows - 1) // target_rows) * target_rows
        padded_cols = ((current_cols + target_cols - 1) // target_cols) * target_cols

        padded_grid = []
        for r in range(padded_rows):
            row = []
            for c in range(padded_cols):
                if r < current_rows and c < len(level_grid[r]):
                    row.append(level_grid[r][c])
                else:
                    row.append(self.void_char)
            padded_grid.append(row)
        return padded_grid

    # Extract all non-void 16x11 rooms from a single level file and save them separately
    def process_text_level_to_rooms(self, full_level_text_path: str, output_base_dir: str):
        print(f"\n--- Starting processing for level: {os.path.basename(full_level_text_path)} ---")
        try:
            level_grid = self._read_level_text(full_level_text_path)
        except FileNotFoundError as e:
            print(e)
            return

        if not level_grid:
            print(f"Level grid from {os.path.basename(full_level_text_path)} is empty. Skipping.")
            return

        standard_room_rows, standard_room_cols = self.default_room_rows, self.default_room_cols
        padded_level_grid = self._pad_level_grid(level_grid, standard_room_rows, standard_room_cols)

        level_name = os.path.splitext(os.path.basename(full_level_text_path))[0]
        output_level_dir = os.path.join(output_base_dir, level_name)
        os.makedirs(output_level_dir, exist_ok=True)
        print(f"Saving extracted rooms to: {output_level_dir}")

        room_count = 0
        total_rows = len(padded_level_grid)
        total_cols = len(padded_level_grid[0])

        for r_grid_idx in range(total_rows // standard_room_rows):
            r_start = r_grid_idx * standard_room_rows
            for c_grid_idx in range(total_cols // standard_room_cols):
                c_start = c_grid_idx * standard_room_cols

                room_data = []
                is_room_empty = True
                for r_offset in range(standard_room_rows):
                    row_content = []
                    for c_offset in range(standard_room_cols):
                        char = padded_level_grid[r_start + r_offset][c_start + c_offset]
                        row_content.append(char)
                        if char != self.void_char:
                            is_room_empty = False
                    room_data.append("".join(row_content))

                if not is_room_empty:
                    room_file_name = f"room_{room_count:04d}_{r_grid_idx}_{c_grid_idx}.txt"
                    with open(os.path.join(output_level_dir, room_file_name), "w", encoding="utf-8") as out_f:
                        for line in room_data:
                            out_f.write(line + "\n")
                    room_count += 1

        print(f"Extracted {room_count} rooms from {os.path.basename(full_level_text_path)}.")
