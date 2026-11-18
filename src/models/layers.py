import tensorflow as tf
import keras
from keras import layers
import config


# Build convolutional encoder conditioned on tiled 4-bit door flags
def build_encoder(input_shape_x: tuple, input_shape_y: tuple, latent_dim: int) -> keras.Model:
    encoder_input_x = keras.Input(shape=input_shape_x, name="room_input")
    encoder_input_y = keras.Input(shape=input_shape_y, name="condition_input_encoder")

    # Broadcast door vector to (16, 11, 4) and concatenate along channel axis
    y_expanded = layers.Lambda(
        lambda y: tf.expand_dims(tf.expand_dims(y, 1), 1),
        name="expand_y",
        output_shape=(1, 1, input_shape_y[-1]),
    )(encoder_input_y)

    y_tiled = layers.Lambda(
        lambda y_e: tf.tile(y_e, [1, config.ROOM_HEIGHT, config.ROOM_WIDTH, 1]),
        name="tile_y",
        output_shape=(input_shape_x[0], input_shape_x[1], input_shape_y[-1]),
    )(y_expanded)

    x_and_y = layers.Concatenate(axis=-1)([encoder_input_x, y_tiled])

    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x_and_y)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same")(x)

    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    return keras.Model([encoder_input_x, encoder_input_y], [z_mean, z_log_var], name="encoder")


# Build transposed-convolutional decoder reconstructing a (16, 11, 10) probability grid
def build_decoder(latent_dim: int, input_shape_y: tuple, output_shape_x: tuple) -> keras.Model:
    decoder_input_z = keras.Input(shape=(latent_dim,), name="z_sample")
    decoder_input_y = keras.Input(shape=input_shape_y, name="condition_input_decoder")

    z_and_y = layers.Concatenate(axis=-1)([decoder_input_z, decoder_input_y])

    start_h = 4
    start_w = 3

    x = layers.Dense(start_h * start_w * 128, activation="relu")(z_and_y)
    x = layers.Reshape((start_h, start_w, 128))(x)

    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    # Upsampling (8, 6) with stride 2 yields (16, 12); crop 1 column on the right to match width 11
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Cropping2D(cropping=((0, 0), (0, 1)))(x)

    decoder_output = layers.Conv2DTranspose(config.NUM_TILES, 3, activation="softmax", padding="same")(x)

    return keras.Model([decoder_input_z, decoder_input_y], decoder_output, name="decoder")


# Reparameterization trick layer: z = z_mean + exp(0.5 * z_log_var) * epsilon
@keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
