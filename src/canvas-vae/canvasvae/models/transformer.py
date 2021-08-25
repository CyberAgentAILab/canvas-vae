import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class PositionEmbedding(tf.keras.layers.Layer):
    """Returns positional const embeddings."""
    def __init__(
        self,
        output_dim,
        maxlen,
        dropout=0.1,
        emb_options=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embeddings = tf.keras.layers.Embedding(
            maxlen,
            output_dim,
            **(emb_options or {}),
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        batch = tf.shape(inputs)[0]
        positions = tf.range(tf.shape(inputs)[1])
        embeddings = self.embeddings(positions[tf.newaxis, :])
        embeddings = tf.tile(embeddings, [batch, 1, 1])
        embeddings = self.dropout(embeddings, training=training)
        return embeddings


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """
    Taken from
    https://keras.io/examples/nlp/text_classification_with_transformer/

    :param emb_size: Size of the embedding.
    :param num_heads: Number of heads.
    :param lookahead: Allow attention to future tokens.
    """
    def __init__(self, emb_size, num_heads=8, lookahead=True, **dense_options):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.lookahead = lookahead
        if emb_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {emb_size} should be divisible by "
                f"number of heads = {num_heads}.")
        self.projection_dim = emb_size // num_heads
        self.dense_query = tf.keras.layers.Dense(emb_size, **dense_options)
        self.dense_key = tf.keras.layers.Dense(emb_size, **dense_options)
        self.dense_value = tf.keras.layers.Dense(emb_size, **dense_options)
        self.combine_heads = tf.keras.layers.Dense(emb_size, **dense_options)
        self.supports_masking = True

    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key,
                          transpose_b=True)  # (B, H, S, projection_dim)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)  # (B, H, S, S)
        scaled_score = score / tf.math.sqrt(dim_key)  # (B, H, S, S)
        if mask is not None:
            # padding mask (B, 1, 1, S)
            mask = tf.cast(mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]
            if not self.lookahead:
                size = tf.shape(mask)[-1]
                mask *= tf.linalg.band_part(tf.ones((size, size)), -1,
                                            0)[tf.newaxis, tf.newaxis, :, :]
            # Force large negative for masks: (B, H, S, S).
            scaled_score += -1e+9 * (1. - mask)
        weights = tf.nn.softmax(scaled_score, axis=-1)  # (B, H, S, S)
        output = tf.matmul(weights, value)  # (B, H, S, projection_dim)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x,
                       (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        # inputs.shape = [B, S, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.dense_query(inputs)  # (B, S, emb_size)
        query = self.separate_heads(query,
                                    batch_size)  # (B, H, S, projection_dim)
        key = self.dense_key(inputs)  # (B, S, emb_size)
        key = self.separate_heads(key, batch_size)  # (B, H, S, projection_dim)
        value = self.dense_value(inputs)  # (B, S, emb_size)
        value = self.separate_heads(value,
                                    batch_size)  # (B, H, S, projection_dim)
        attention, _ = self.attention(query, key, value, mask)
        attention = tf.transpose(attention,
                                 perm=[0, 2, 1,
                                       3])  # (B, S, H, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.emb_size))  # (B, S, emb_size)
        output = self.combine_heads(concat_attention)  # (B, S, emb_size)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with optional global conditional."""
    def __init__(
        self,
        emb_size=64,
        num_heads=8,
        ff_dim=None,
        dropout=0.1,
        conditional=None,
        pooling=None,
        dense_options=None,
        lookahead=True,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__(**kwargs)
        dense_options = dense_options or {}
        self.attn = MultiHeadSelfAttention(emb_size,
                                           num_heads,
                                           lookahead=lookahead,
                                           **dense_options)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(
                ff_dim or (2 * emb_size),
                activation="relu",
                **dense_options,
            ),
            tf.keras.layers.Dense(emb_size, **dense_options),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True
        self.conditional = None
        if conditional:
            self.norm3 = tf.keras.layers.LayerNormalization()
            self.conditional = tf.keras.layers.Dense(emb_size, **dense_options)

        self.pooling = None
        if pooling:
            self.relu = tf.keras.layers.Activation('relu')
            self.pooling = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, inputs, training=False, mask=None):
        if self.conditional is not None:
            x, z = inputs
        else:
            x = inputs
        y = self.attn(x, mask=mask)
        y = self.dropout1(y, training=training)
        x = self.norm1(x + y, training=training)
        if self.conditional is not None:
            z = tf.expand_dims(self.conditional(z), 1)
            x = self.norm3(x + z, training=training)
        y = self.mlp(x)
        y = self.dropout2(y, training=training)
        x = self.norm2(x + y, training=training)
        if self.pooling is not None:
            x = self.relu(x)
            return self.pooling(x, mask=mask)
        return x


class DeepSVGBlock(TransformerBlock):
    """DeepSVG-style transformer block."""
    def call(self, inputs, training=False, mask=None):
        if self.conditional is not None:
            x, z = inputs
        else:
            x = inputs
        y = self.norm1(x, training=training)
        y = self.attn(y, mask=mask)
        y = self.dropout1(y, training=training)
        x += y
        if self.conditional is not None:
            x += tf.expand_dims(self.conditional(z), 1)
        y = self.norm2(x, training=training)
        y = self.mlp(y)
        y = self.dropout2(y, training=training)
        x = x + y
        if self.pooling is not None:
            x = self.relu(x)
            return self.pooling(x, mask=mask)
        return x


class LSTMBlock(tf.keras.layers.Layer):
    """LSTM-based block"""
    def __init__(
        self,
        *args,
        conditional=None,
        pooling=None,
        dense_options=None,
        dropout=0.1,
        lookahead=True,
        **kwargs,
    ):
        super(LSTMBlock, self).__init__(**kwargs)
        self.lookahead = lookahead
        if self.lookahead:
            self.lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    *args,
                    dropout=dropout,
                    return_sequences=False if pooling else True,
                    **(dense_options or {}),
                ),
                merge_mode='sum',
            )
        else:
            self.lstm = tf.keras.layers.LSTM(
                *args,
                dropout=dropout,
                return_sequences=False if pooling else True,
                **(dense_options or {}),
            )
        self.conditional = conditional
        self.pooling = pooling

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if self.conditional:
            x, z = inputs
            if self.lookahead:
                initial_state = [z, z, z, z]
            else:
                initial_state = [z, z]
        else:
            x = inputs
            initial_state = None

        return self.lstm(
            x,
            mask=mask,
            training=training,
            initial_state=initial_state,
        )


def get_sequence_block(layer_type):
    return {
        'transformer': TransformerBlock,
        'lstm': LSTMBlock,
        'deepsvg': DeepSVGBlock,
    }[layer_type]
