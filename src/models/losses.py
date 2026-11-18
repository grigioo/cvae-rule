import tensorflow as tf
import keras
import config

# Pre-defined tensor constants for domain-specific rule checks
VOID_ID = tf.constant(config.TILE_TO_IDX["-"], dtype=tf.int32)
WALL_ID = tf.constant(config.TILE_TO_IDX["W"], dtype=tf.int32)
DOOR_ID = tf.constant(config.TILE_TO_IDX["D"], dtype=tf.int32)
FLOOR_ID = tf.constant(config.TILE_TO_IDX["F"], dtype=tf.int32)

OBJECT_TILES_IDS = tf.constant(
    [
        config.TILE_TO_IDX["M"],
        config.TILE_TO_IDX["P"],
        config.TILE_TO_IDX["O"],
        config.TILE_TO_IDX["I"],
        config.TILE_TO_IDX["S"],
    ],
    dtype=tf.int32,
)


# Evaluate structural validity penalties on generated rooms
def calculate_rule_loss(
    generated_outputs_logits,
    target_door_info_flat,
    room_height: int = config.ROOM_HEIGHT,
    room_width: int = config.ROOM_WIDTH,
    enable_void_rule: bool = True,
    enable_wall_integrity_rule: bool = True,
    enable_object_placement_rule: bool = True,
    enable_door_environment_rule: bool = True,
):
    generated_token_ids = tf.argmax(generated_outputs_logits, axis=-1, output_type=tf.int32)
    batch_size = tf.shape(generated_token_ids)[0]
    rule_penalties = tf.zeros(batch_size, dtype=tf.float32)

    # Rule 1: Penalize void ('-') tiles appearing inside a room
    if enable_void_rule:
        penalty_void = tf.reduce_sum(tf.cast(generated_token_ids == VOID_ID, tf.float32), axis=[1, 2])
        rule_penalties += penalty_void

    # Rule 2: Enforce perimeter boundary integrity (outer border must contain only walls or doors)
    if enable_wall_integrity_rule:
        top_row = generated_token_ids[:, 0, :]
        rule_penalties += tf.reduce_sum(
            tf.cast(tf.logical_and(top_row != WALL_ID, top_row != DOOR_ID), tf.float32), axis=1
        )

        bottom_row = generated_token_ids[:, room_height - 1, :]
        rule_penalties += tf.reduce_sum(
            tf.cast(tf.logical_and(bottom_row != WALL_ID, bottom_row != DOOR_ID), tf.float32), axis=1
        )

        left_col = generated_token_ids[:, 1 : room_height - 1, 0]
        rule_penalties += tf.reduce_sum(
            tf.cast(tf.logical_and(left_col != WALL_ID, left_col != DOOR_ID), tf.float32), axis=1
        )

        right_col = generated_token_ids[:, 1 : room_height - 1, room_width - 1]
        rule_penalties += tf.reduce_sum(
            tf.cast(tf.logical_and(right_col != WALL_ID, right_col != DOOR_ID), tf.float32), axis=1
        )

    # Rule 3: Prevent gameplay objects from spawning inside outer walls or void spaces
    if enable_object_placement_rule:
        expanded_ids = tf.expand_dims(generated_token_ids, axis=-1)
        is_object_tile = tf.reduce_any(tf.equal(expanded_ids, OBJECT_TILES_IDS), axis=-1)
        is_invalid_placement = tf.logical_or(generated_token_ids == WALL_ID, generated_token_ids == VOID_ID)
        penalty_object_placement = tf.reduce_sum(
            tf.cast(tf.logical_and(is_object_tile, is_invalid_placement), tf.float32), axis=[1, 2]
        )
        rule_penalties += penalty_object_placement

    # Rule 4: Verify expected door placements and ensure unobstructed floor tiles directly in front of them
    if enable_door_environment_rule:
        horizontal_door_cols = tf.constant([room_width // 2 - 1, room_width // 2, room_width // 2 + 1], dtype=tf.int32)
        vertical_door_rows = tf.constant([room_height // 2 - 1, room_height // 2], dtype=tf.int32)

        # Top door check
        top_door_expected = tf.cast(target_door_info_flat[:, 0], tf.bool)
        actual_top_door_tiles = tf.gather(generated_token_ids[:, 1, :], horizontal_door_cols, axis=1)
        is_actual_top_door = tf.reduce_all(tf.equal(actual_top_door_tiles, DOOR_ID), axis=1)
        rule_penalties += tf.cast(tf.logical_and(top_door_expected, tf.logical_not(is_actual_top_door)), tf.float32) * 5.0

        floor_below_top = tf.gather(generated_token_ids[:, 2, :], horizontal_door_cols, axis=1)
        is_floor_below_top = tf.reduce_all(tf.equal(floor_below_top, FLOOR_ID), axis=1)
        rule_penalties += tf.cast(tf.logical_and(is_actual_top_door, tf.logical_not(is_floor_below_top)), tf.float32) * 2.0

        # Bottom door check
        bottom_door_expected = tf.cast(target_door_info_flat[:, 1], tf.bool)
        actual_bottom_door_tiles = tf.gather(generated_token_ids[:, room_height - 2, :], horizontal_door_cols, axis=1)
        is_actual_bottom_door = tf.reduce_all(tf.equal(actual_bottom_door_tiles, DOOR_ID), axis=1)
        rule_penalties += (
            tf.cast(tf.logical_and(bottom_door_expected, tf.logical_not(is_actual_bottom_door)), tf.float32) * 5.0
        )

        floor_above_bottom = tf.gather(generated_token_ids[:, room_height - 3, :], horizontal_door_cols, axis=1)
        is_floor_above_bottom = tf.reduce_all(tf.equal(floor_above_bottom, FLOOR_ID), axis=1)
        rule_penalties += (
            tf.cast(tf.logical_and(is_actual_bottom_door, tf.logical_not(is_floor_above_bottom)), tf.float32) * 2.0
        )

        # Left door check
        left_door_expected = tf.cast(target_door_info_flat[:, 2], tf.bool)
        actual_left_door_tiles = tf.gather(generated_token_ids[:, :, 1], vertical_door_rows, axis=1)
        is_actual_left_door = tf.reduce_all(tf.equal(actual_left_door_tiles, DOOR_ID), axis=1)
        rule_penalties += (
            tf.cast(tf.logical_and(left_door_expected, tf.logical_not(is_actual_left_door)), tf.float32) * 5.0
        )

        floor_right_of_left = tf.gather(generated_token_ids[:, :, 2], vertical_door_rows, axis=1)
        is_floor_right_of_left = tf.reduce_all(tf.equal(floor_right_of_left, FLOOR_ID), axis=1)
        rule_penalties += (
            tf.cast(tf.logical_and(is_actual_left_door, tf.logical_not(is_floor_right_of_left)), tf.float32) * 2.0
        )

        # Right door check
        right_door_expected = tf.cast(target_door_info_flat[:, 3], tf.bool)
        actual_right_door_tiles = tf.gather(generated_token_ids[:, :, room_width - 2], vertical_door_rows, axis=1)
        is_actual_right_door = tf.reduce_all(tf.equal(actual_right_door_tiles, DOOR_ID), axis=1)
        rule_penalties += (
            tf.cast(tf.logical_and(right_door_expected, tf.logical_not(is_actual_right_door)), tf.float32) * 5.0
        )

        floor_left_of_right = tf.gather(generated_token_ids[:, :, room_width - 3], vertical_door_rows, axis=1)
        is_floor_left_of_right = tf.reduce_all(tf.equal(floor_left_of_right, FLOOR_ID), axis=1)
        rule_penalties += (
            tf.cast(tf.logical_and(is_actual_right_door, tf.logical_not(is_floor_left_of_right)), tf.float32) * 2.0
        )

    return tf.reduce_mean(rule_penalties)


# Auxiliary serialized loss layer kept for backward compatibility with saved checkpoints
@keras.utils.register_keras_serializable()
class CVAELoss(keras.layers.Layer):
    def call(self, inputs):
        x_true, x_reconstructed, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(tf.cast(x_true, dtype=tf.int32), x_reconstructed)
        ) * tf.cast(tf.size(x_true), dtype=tf.float32)

        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)
        return total_loss

    def get_config(self):
        return super().get_config()
