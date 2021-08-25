import logging

import tensorflow as tf
from .resnet_decoder import ResNetDecoder, SequentialDecoder

logger = logging.getLogger(__name__)


class Sampling(tf.keras.layers.Layer):
    """Sampling head for VAE."""
    def __init__(self, kl=1.0, **kwargs):
        super().__init__(**kwargs)
        self.kl = kl

    def call(self, inputs, training=False):
        z_mean, z_log_sigma = inputs

        # Compute KL divergence to normal distribution.
        loss = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean) -
                                     tf.exp(z_log_sigma))
        self.add_loss(self.kl * loss)
        self.add_metric(loss, name='kl_divergence')

        if training:
            # Sample when training.
            epsilon = tf.random.normal(shape=tf.shape(z_log_sigma))
            z_mean += tf.exp(0.5 * z_log_sigma) * epsilon
        return z_mean


class VariationalHead(tf.keras.layers.Layer):
    """Encoder head for VAE."""
    def __init__(self, latent_dim, dense_options=None, **kwargs):
        super().__init__(**kwargs)
        self.z_mean = tf.keras.layers.Dense(
            latent_dim,
            name='z_mean',
            **(dense_options or {}),
        )
        self.z_log_sigma = tf.keras.layers.Dense(
            latent_dim,
            name='z_log_sigma',
            **(dense_options or {}),
        )

    def call(self, inputs):
        z_mean = self.z_mean(inputs)
        z_log_sigma = self.z_log_sigma(inputs)
        return z_mean, z_log_sigma


class Encoder(tf.keras.layers.Layer):
    '''
    Encoder implementation.
    '''
    def __init__(
        self,
        arch,
        input_shape=None,
        latent_dim=128,
        dense_options=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cnn = getattr(tf.keras.applications, arch)(
            input_shape=input_shape,
            include_top=False,
            pooling='avg',
            weights=None,
        )
        self.head = VariationalHead(
            latent_dim,
            dense_options=dense_options,
        )

    def call(self, inputs, training=False):
        x = tf.image.convert_image_dtype(inputs, tf.float32)
        x = self.cnn(x, training=training)
        return self.head(x)


class Decoder(tf.keras.layers.Layer):
    '''
    Decoder implementation.
    '''
    def __init__(self, output_shape, arch=None, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(units=8 * 8 * 32, activation='relu')
        self.reshape0 = tf.keras.layers.Reshape(target_shape=(8, 8, 32))
        if arch == 'resnet':
            self.cnn = ResNetDecoder(
                output_channels=output_shape[2] * output_shape[3],
                filters=[64, 64, 32, 16],
                blocks=[2, 2, 2, 2],
            )
        else:
            self.cnn = SequentialDecoder(output_shape[2] * output_shape[3])
        self.reshape1 = tf.keras.layers.Reshape(target_shape=output_shape)

    def call(self, z, training=False):
        x = self.dense(z)
        x = self.reshape0(x)
        x = self.cnn(x, training=training)
        x = self.reshape1(x)
        return x


class PixelVAE(tf.keras.Model):
    """
    VAE for pixels.
    """
    def __init__(
        self,
        arch='MobileNetV2',
        input_shape=None,
        latent_dim=256,
        kl=None,
        l2=None,
        quantize_factor=4,
        name='pixelvae',
        **kwargs,
    ):
        super().__init__()
        input_shape = input_shape or (256, 256, 4)

        self.quantize_factor = quantize_factor

        dense_options = None
        if l2 is not None:
            dense_options = dict(
                kernel_regularizer=tf.keras.regularizers.L2(l2=l2),
                bias_regularizer=tf.keras.regularizers.L2(l2=l2),
            )
        self.encoder = Encoder(
            arch,
            input_shape=input_shape,
            latent_dim=latent_dim,
            dense_options=dense_options,
            **kwargs,
        )
        self.sampling = Sampling(kl=kl)

        self.decoder = Decoder(
            output_shape=input_shape + (256 >> quantize_factor, ),
            **kwargs,
        )

        self._make(input_shape)

    def call(self, inputs, training=False):
        Z = self.encoder(inputs, training=training)
        z = self.sampling(Z, training=training)
        outputs = self.decoder(z, training=training)

        inputs_quantized = tf.bitwise.right_shift(inputs, self.quantize_factor)
        loss = reconstruction_loss_fn(inputs_quantized, outputs)
        self.add_loss(loss)
        self.add_metric(loss, name='reconstruction')

        return outputs

    def _make(self, input_shape):
        """Pass through the symbolic input to build the model."""
        inputs = tf.keras.Input(
            input_shape,
            dtype=tf.uint8,
            name='image',
        )
        return self(inputs)


@tf.function
def reconstruction_loss_fn(y_true, y_pred):
    loss_fn = tf.keras.losses.sparse_categorical_crossentropy

    rgb_true = y_true[:, :, :, :-1]
    rgb_pred = y_pred[:, :, :, :-1, :]
    alpha_true = y_true[:, :, :, -1:]
    alpha_pred = y_pred[:, :, :, -1:, :]
    mask = tf.cast(tf.math.greater(alpha_true, 0), tf.float32)

    rgb_loss = loss_fn(rgb_true, rgb_pred, from_logits=True) * mask
    alpha_loss = loss_fn(alpha_true, alpha_pred, from_logits=True)
    loss = tf.reduce_sum(rgb_loss + alpha_loss, axis=1)
    return tf.reduce_mean(loss)
