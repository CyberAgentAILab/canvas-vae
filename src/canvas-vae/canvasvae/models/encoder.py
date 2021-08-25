import logging

import tensorflow as tf

from .mask import get_mask, Unmask
from .transformer import PositionEmbedding, get_sequence_block
from .utils import make_dense_options, make_emb_options

logger = logging.getLogger(__name__)


class VariationalHead(tf.keras.layers.Layer):
    """Encoder head for VAE."""
    def __init__(self, output_dim, kl=None, l2=None, **kwargs):
        super().__init__(**kwargs)
        self.kl = 1.0 if kl is None else kl

        self.z_mean = tf.keras.layers.Dense(
            output_dim,
            name='z_mean',
            **make_dense_options(l2),
        )
        self.z_log_sigma = tf.keras.layers.Dense(
            output_dim,
            name='z_log_sigma',
            **make_dense_options(l2),
        )

    def call(self, inputs, training=False):
        z_mean = self.z_mean(inputs)
        z_log_sigma = self.z_log_sigma(inputs)

        # Compute KL divergence to normal distribution.
        kl_div = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean) -
                                       tf.exp(z_log_sigma))
        self.add_loss(self.kl * kl_div)
        self.add_metric(kl_div, name='kl_divergence')

        if training:
            # Sample when training.
            epsilon = tf.random.normal(shape=tf.shape(z_log_sigma))
            z_mean += tf.exp(0.5 * z_log_sigma) * epsilon
        return z_mean


class Encoder(tf.keras.layers.Layer):
    '''
    Encoder implementation.
    '''
    def __init__(
        self,
        input_columns,
        latent_dim=128,
        num_blocks=1,
        block_type='deepsvg',
        dropout=0.1,
        kl=None,
        l2=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_columns = input_columns
        self.kl = 1.0 if kl is None else kl
        self.latent_dim = latent_dim

        self.input_layer = {}
        self.input_layer['const'] = PositionEmbedding(
            latent_dim,
            self.input_columns['length']['input_dim'],
            dropout=dropout,
            emb_options=make_emb_options(l2),
            name='input_const',
        )
        for key, column in self.input_columns.items():
            if column['type'] == 'categorical':
                self.input_layer[key] = tf.keras.layers.Embedding(
                    input_dim=column['input_dim'],
                    output_dim=latent_dim,
                    name='input_%s' % key,
                    **make_emb_options(l2),
                )
            elif column['type'] == 'numerical':
                self.input_layer[key] = tf.keras.layers.Dense(
                    units=latent_dim,
                    name='input_%s' % key,
                    **make_dense_options(l2),
                )
            else:
                raise ValueError('Invalid column: %s' % column)

        self.seq2seq = {}
        layer_fn = get_sequence_block(block_type)
        for i in range(num_blocks):
            self.seq2seq['seq2seq_%d' % i] = layer_fn(
                latent_dim,
                dropout=dropout,
                conditional=True,
                pooling=(i == num_blocks - 1),
                dense_options=make_dense_options(l2),
                name='seq2seq_%d' % i,
            )

        self.norm = tf.keras.layers.BatchNormalization()
        self.unmask = Unmask()
        self.head = VariationalHead(latent_dim, kl=kl, l2=l2, name='z_head')

    def call(self, inputs, training=False, sampling=False):
        batch = tf.shape(inputs['length'])[0]

        # Context inputs.
        context = tf.zeros((batch, 1, self.latent_dim), dtype=tf.float32)
        for key, column in self.input_columns.items():
            if not column['is_sequence']:
                x = self.input_layer[key](inputs[key])
                context += x
        context = context[:, 0, :]  # Squeeze timesteps.
        tf.debugging.assert_rank(context, 2)

        # Sequence inputs.
        mask = get_mask(inputs['length'])
        sequence = self.input_layer['const'](mask, training=training)
        for key, column in self.input_columns.items():
            if column['is_sequence']:
                x = self.input_layer[key](inputs[key])
                if column['type'] == 'categorical':
                    x = tf.reduce_sum(x, axis=2)  # Vector categorical
                sequence += x
        tf.debugging.assert_rank(sequence, 3)

        # Sequence transform.
        for layer in self.seq2seq.values():
            sequence = layer((sequence, context), training=training, mask=mask)

        # Last layer already applies temporal pooling, shape=(N, D).
        pooled = self.norm(sequence, training=training)
        pooled = self.unmask(pooled)

        return self.head(pooled, training=training or sampling)
