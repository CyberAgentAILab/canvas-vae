import logging

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

logger = logging.getLogger(__name__)


class SequenceDiscretizer(tf.keras.layers.Layer):
    """
    Discretization wrapper for stable operation under variable shapes.
    """
    def __init__(self, bin_boundaries, **kwargs):
        super().__init__(**kwargs)
        self.discretizer = preprocessing.Discretization(bin_boundaries)

    def call(self, inputs):
        if tf.__version__.startswith('2.3'):
            # TF 2.3 behavior
            return self.discretizer(inputs)
        else:
            # TF 2.4 or later has unstable discretization behavior.
            inputs = tf.cast(inputs, tf.float32)
            shape = tf.shape(inputs)
            reshaped = tf.reshape(inputs, (-1, 1))
            outputs = self.discretizer(reshaped)
            return tf.reshape(outputs, shape)

    @property
    def bin_boundaries(self):
        if tf.__version__.split('.')[1] in ('3', '4'):
            return self.discretizer.bins
        else:
            return self.discretizer.bin_boundaries
