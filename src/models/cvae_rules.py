import tensorflow as tf
import keras
import config
from src.models.losses import calculate_rule_loss


# Conditional Variational Autoencoder extended with domain-specific rule penalties
@keras.utils.register_keras_serializable()
class CVAERules(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.rule_loss_tracker = keras.metrics.Mean(name="rule_loss")

        self._enable_void_rule = True
        self._enable_wall_integrity_rule = True
        self._enable_object_placement_rule = True
        self._enable_door_environment_rule = True

    def set_rule_flags(self, void=True, wall_integrity=True, object_placement=True, door_environment=True):
        self._enable_void_rule = void
        self._enable_wall_integrity_rule = wall_integrity
        self._enable_object_placement_rule = object_placement
        self._enable_door_environment_rule = door_environment

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
        rule_loss = calculate_rule_loss(
            reconstruction,
            y_batch_cond,
            room_height=config.ROOM_HEIGHT,
            room_width=config.ROOM_WIDTH,
            enable_void_rule=self._enable_void_rule,
            enable_wall_integrity_rule=self._enable_wall_integrity_rule,
            enable_object_placement_rule=self._enable_object_placement_rule,
            enable_door_environment_rule=self._enable_door_environment_rule,
        )

        total_loss = reconstruction_loss + kl_loss + config.RULE_LOSS_WEIGHT * rule_loss
        return total_loss, reconstruction_loss, kl_loss, rule_loss

    def train_step(self, data):
        (x_batch, y_batch_cond), x_target_batch = data

        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss, rule_loss = self._compute_losses(
                x_batch, y_batch_cond, x_target_batch
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.rule_loss_tracker.update_state(rule_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "rule_loss": self.rule_loss_tracker.result(),
        }

    def test_step(self, data):
        (x_batch, y_batch_cond), x_target_batch = data
        total_loss, reconstruction_loss, kl_loss, rule_loss = self._compute_losses(
            x_batch, y_batch_cond, x_target_batch
        )

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.rule_loss_tracker.update_state(rule_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "rule_loss": self.rule_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.rule_loss_tracker,
        ]

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update(
            {
                "encoder_config": keras.utils.serialize_keras_object(self.encoder),
                "decoder_config": keras.utils.serialize_keras_object(self.decoder),
                "enable_void_rule": self._enable_void_rule,
                "enable_wall_integrity_rule": self._enable_wall_integrity_rule,
                "enable_object_placement_rule": self._enable_object_placement_rule,
                "enable_door_environment_rule": self._enable_door_environment_rule,
            }
        )
        return config_dict

    @classmethod
    def from_config(cls, config_dict):
        encoder = keras.utils.deserialize_keras_object(config_dict.pop("encoder_config"), safe_mode=False)
        decoder = keras.utils.deserialize_keras_object(config_dict.pop("decoder_config"), safe_mode=False)

        void_rule = config_dict.pop("enable_void_rule", True)
        wall_integrity_rule = config_dict.pop("enable_wall_integrity_rule", True)
        object_placement_rule = config_dict.pop("enable_object_placement_rule", True)
        door_environment_rule = config_dict.pop("enable_door_environment_rule", True)

        instance = cls(encoder=encoder, decoder=decoder, **config_dict)
        instance.set_rule_flags(
            void=void_rule,
            wall_integrity=wall_integrity_rule,
            object_placement=object_placement_rule,
            door_environment=door_environment_rule,
        )
        return instance
