import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def make_dense_options(l2):
    if l2 is None:
        return {}
    return dict(
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2),
    )


def make_emb_options(l2):
    if l2 is None:
        return {}
    return dict(embeddings_regularizer=tf.keras.regularizers.l2(l2), )
