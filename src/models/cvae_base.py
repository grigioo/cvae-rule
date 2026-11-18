import tensorflow as tf
import keras
import config


# Standard Conditional Variational Autoencoder trained with reconstruction + KL divergence loss
@keras.utils.register_keras_serializable()
class CVAEBase(keras.Model):
    def __init__(self, encoder, decoder, cvae_loss_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.cvae_loss_layer = cvae_loss_layer

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        x_batch, y_batch_cond = inputs
        z_mean, z_log_var = self.encoder([x_batch, y_batch_cond])

        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, config.LATENT_DIM))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        return self.decoder([z, y_batch_cond])

    def _compute_losses(self, x_batch, y_batch_cond, x_target_batch):
        z_mean, z_log_var = self.encoder([x_batch, y_batch_cond])

        batch_size = tf.shape(z_mean)[0]
        epsilon = tf.random.normal(shape=(batch_size, config.LATENT_DIM))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

        reconstruction = self.decoder([z, y_batch_cond])

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(keras.losses.categorical_crossentropy(x_target_batch, reconstruction), axis=(1, 2))
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        (x_batch, y_batch_cond), x_target_batch = data

        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self._compute_losses(x_batch, y_batch_cond, x_target_batch)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        (x_batch, y_batch_cond), x_target_batch = data
        total_loss, reconstruction_loss, kl_loss = self._compute_losses(x_batch, y_batch_cond, x_target_batch)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update(
            {
                "encoder_config": keras.utils.serialize_keras_object(self.encoder),
                "decoder_config": keras.utils.serialize_keras_object(self.decoder),
                "cvae_loss_layer_config": keras.utils.serialize_keras_object(self.cvae_loss_layer)
                if self.cvae_loss_layer is not None
                else None,
            }
        )
        return config_dict

    @classmethod
    def from_config(cls, config_dict):
        encoder = keras.utils.deserialize_keras_object(config_dict.pop("encoder_config"), safe_mode=False)
        decoder = keras.utils.deserialize_keras_object(config_dict.pop("decoder_config"), safe_mode=False)
        loss_layer_cfg = config_dict.pop("cvae_loss_layer_config", None)
        cvae_loss_layer = (
            keras.utils.deserialize_keras_object(loss_layer_cfg, safe_mode=False)
            if loss_layer_cfg is not None
            else None
        )
        return cls(encoder=encoder, decoder=decoder, cvae_loss_layer=cvae_loss_layer, **config_dict)
