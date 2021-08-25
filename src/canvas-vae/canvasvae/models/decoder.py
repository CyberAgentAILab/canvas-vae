import logging

import tensorflow as tf

from .mask import get_mask
from .transformer import PositionEmbedding, get_sequence_block
from .utils import make_dense_options, make_emb_options

logger = logging.getLogger(__name__)


class DecoderHead(tf.keras.layers.Layer):
    """Multi-way head for decoders."""
    def __init__(self, input_columns, l2=None, **kwargs):
        super().__init__(**kwargs)
        self.input_columns = input_columns

        self.decoders = {}
        for key, column in self.input_columns.items():
            if column['type'] == 'categorical':
                units = column['shape'][-1] * column['input_dim']
            else:
                units = column['shape'][-1]

            self.decoders[key] = tf.keras.layers.Dense(
                units,
                name='decoder_%s' % key,
                **make_dense_options(l2),
            )

    # def compute_mask(self, z, mask=None):
    #     """Compute mask according to Keras specification."""
    #     if isinstance(z, tuple):
    #         _, z = z
    #         mask = get_mask(inputs['length'])
    #     else:
    #         mask = self.predict_mask(z)
    #     tf.debugging.assert_rank(mask, 2)

    #     outputs = {}
    #     for key, column in self.input_columns.items():
    #         if column['is_sequence']:
    #             outputs[key] = mask
    #         else:
    #             outputs[key] = None
    #     return outputs

    def predict_mask(self, z):
        length_logit = self.decoders['length'](z)
        return get_mask(length_logit, from_logits=True)

    def call(self, inputs):
        """Take a sequence of transformed embeddings and compute outputs."""
        sequence, z = inputs
        batch = tf.shape(z)[0]

        # Predict output for each head.
        outputs = {}
        for key, column in self.input_columns.items():
            if column['type'] == 'categorical':
                shape = (column['shape'][-1], column['input_dim'])
            else:
                shape = (column['shape'][-1], )

            if column['is_sequence']:
                outputs[key] = tf.reshape(self.decoders[key](sequence),
                                          (batch, -1) + shape)
                tf.debugging.assert_rank_at_least(outputs[key], 3)
            else:
                outputs[key] = tf.reshape(self.decoders[key](z),
                                          (batch, ) + shape)
                tf.debugging.assert_rank(outputs[key], 3)
        return outputs


class Decoder(tf.keras.layers.Layer):
    '''
    Decoder implementation.

    Decoder takes z or (z, minlen). minlen is optional length specifier.
    '''
    def __init__(self,
                 input_columns,
                 latent_dim=128,
                 num_blocks=1,
                 block_type='deepsvg',
                 dropout=0.1,
                 l2=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_columns = input_columns
        self.latent_dim = latent_dim

        self.embedding_const = PositionEmbedding(
            latent_dim,
            self.input_columns['length']['input_dim'],
            dropout=dropout,
            emb_options=make_emb_options(l2),
            name='embedding_const',
        )

        self.seq2seq = {}
        layer_fn = get_sequence_block(block_type)
        for i in range(num_blocks):
            self.seq2seq['seq2seq_%d' % i] = layer_fn(
                latent_dim,
                dropout=dropout,
                conditional=True,
                dense_options=make_dense_options(l2),
                name='seq2seq_%d' % i,
            )

        self.head = DecoderHead(self.input_columns, l2=l2)

    # def compute_mask(self, z, mask=None):
    #     return self.head.compute_mask(z, mask=mask)

    def call(self, z, training=False):
        if training:
            # At training, use the supplied GT mask.
            z, inputs = z
            mask = get_mask(inputs['length'])
        else:
            mask = self.head.predict_mask(z)
        sequence = self.embedding_const(mask, training=training)
        for layer in self.seq2seq.values():
            sequence = layer((sequence, z), training=training, mask=mask)
        tf.debugging.assert_equal(tf.shape(sequence)[:2], tf.shape(mask))
        return self.head((sequence, z))


class AutoregressiveDecoder(tf.keras.layers.Layer):
    '''
    Autoregressive decoder implementation.
    '''
    def __init__(
        self,
        input_columns,
        latent_dim=256,
        dropout=0.1,
        block_type='lstm',
        num_blocks=1,
        l2=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_columns = input_columns
        self.latent_dim = latent_dim

        self.input_layer = {}
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

        # BOS initializer.
        initializer = tf.random_normal_initializer()
        self.bos = tf.Variable(initial_value=initializer(
            shape=(1, 1, self.latent_dim), dtype=tf.float32),
                               trainable=True)

        self.seq2seq = {}
        layer_fn = get_sequence_block(block_type)
        for i in range(num_blocks):
            self.seq2seq['seq2seq_%d' % i] = layer_fn(
                latent_dim,
                dropout=dropout,
                conditional=True,
                dense_options=make_dense_options(l2),
                lookahead=False,
                name='seq2seq_%d' % i,
            )
        self.head = DecoderHead(self.input_columns, l2=l2)

    # def compute_mask(self, z, mask=None):
    #     return self.head.compute_mask(z, mask=mask)

    def call(self, inputs, training=False):
        if training:
            # At training, use the GT mask.
            z, inputs = inputs
            mask = get_mask(inputs['length'])
            sequence = self._forward_train(inputs)
        else:
            z = inputs
            mask = self.head.predict_mask(z)
            sequence = self._forward_inference(z, mask=mask)

        for layer in self.seq2seq.values():
            sequence = layer((sequence, z), training=training, mask=mask)
        return self.head((sequence, z))

    def _forward_train(self, inputs):
        """
        Training is teacher-forcing.

        Inputs contain ground-truth sequence.
        """
        batch = tf.shape(inputs['length'])[0]

        # Project teacher sequence into embeddings.
        sequence = tf.zeros((batch, 1, self.latent_dim), dtype=tf.float32)
        for key, column in self.input_columns.items():
            if column['is_sequence']:
                x = self.input_layer[key](inputs[key], training=True)
                if column['type'] == 'categorical':
                    x = tf.reduce_sum(x, axis=2)  # Vector categorical
                sequence += x
        tf.debugging.assert_rank(sequence, 3)

        # Prepend the beginning-of-seq embedding, and drop the last.
        bos = tf.tile(self.bos, [batch, 1, 1])
        return tf.concat([bos, sequence[:, 0:-1, :]], axis=1)

    def _forward_inference(self, z, mask=None):
        """
        Test-time inference takes only latent code z and iteratively generates.
        """
        batch = tf.shape(z)[0]
        seq_length = tf.shape(mask)[1]

        # Given the initial embedding, loop to build a full embedding sequence.
        sequence = tf.tile(self.bos, [batch, 1, 1])
        for t in range(seq_length - 1):
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                (sequence, tf.TensorShape([None, None, self.latent_dim])),
            ])
            next_elem = self._compute_next((sequence, z), mask=mask[:, :t + 1])
            sequence = tf.concat([sequence, next_elem], axis=1)

        tf.debugging.assert_rank(sequence, 3)
        tf.debugging.assert_equal(tf.shape(sequence)[1], seq_length)
        return sequence

    def _compute_next(self, sequence_z, mask=None):
        if mask is not None:
            is_valid_mask = tf.math.reduce_all(tf.math.reduce_any(mask,
                                                                  axis=1))
            tf.debugging.assert_equal(is_valid_mask,
                                      True,
                                      message='mask=%s' % mask)
        sequence, z = sequence_z
        batch = tf.shape(z)[0]

        # Transform sequence and get the last element.
        for layer in self.seq2seq.values():
            sequence = layer((sequence, z), training=False, mask=mask)
        y_t = sequence[:, -1:]

        # Get output (=next input) at step t.
        next_elem = tf.zeros((batch, 1, self.latent_dim))
        for key, column in self.input_columns.items():
            if column['is_sequence']:
                # Compute output. TODO: Maybe sample output here?
                if column['type'] == 'categorical':
                    shape = (batch, -1, column['shape'][-1],
                             column['input_dim'])
                    logit = tf.reshape(self.head.decoders[key](y_t), shape)
                    output = tf.argmax(logit, axis=-1)
                else:
                    shape = (batch, -1, column['shape'][-1])
                    output = tf.reshape(self.head.decoders[key](y_t), shape)

                # Compute input embedding for the next step.
                x = self.input_layer[key](output, training=False)
                if column['type'] == 'categorical':
                    x = tf.reduce_sum(x, axis=2)  # Vector categorical
                next_elem += x

        tf.debugging.assert_rank(next_elem, 3)
        return next_elem
