import logging

import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder, AutoregressiveDecoder
from .metrics import LossLayer, VectorMetricLayer, LayoutMetricLayer

logger = logging.getLogger(__name__)

_DECODERS = {
    'oneshot': Decoder,
    'autoregressive': AutoregressiveDecoder,
}


class VAE(tf.keras.Model):
    """
    VAE trainer.
    """
    def __init__(
        self,
        input_columns,
        decoder_type='oneshot',
        kl=None,
        l2=None,
        name='vae',
        **kwargs,
    ):
        super(VAE, self).__init__(name=name)
        self.input_columns = input_columns
        self.options = kwargs
        self.encoder = Encoder(self.input_columns, kl=kl, l2=l2, **kwargs)

        self.decoder = _DECODERS[decoder_type](
            self.input_columns,
            l2=l2,
            **kwargs,
        )
        self.loss_layer = LossLayer(self.input_columns)
        self.vector_metric = VectorMetricLayer(
            self.input_columns,
            from_logits=True,
        )
        self.layout_metric = LayoutMetricLayer(
            self.input_columns,
            from_logits=True,
        )

        self._make()

    def call(self, inputs, training=False, sampling=False):
        z = self.encoder(
            inputs,
            training=training,
            sampling=training or sampling,
        )

        if training:
            z = (z, inputs)  # Teacher-forcing inputs.
        outputs = self.decoder(z, training=training)

        self.vector_metric((inputs, outputs), training=training)
        if training:
            self.loss_layer((inputs, outputs))
        else:
            # Layout metric is expensive, only evaluate on validation.
            self.layout_metric((inputs, outputs))
        return outputs

    def _make(self):
        """Pass through the symbolic input to build the model."""
        inputs = {}
        for key, column in self.input_columns.items():
            shape = column['shape']
            if column['is_sequence']:
                shape = (None, ) + shape
            dtype = tf.int32
            if column['type'] == 'numerical':
                dtype = tf.float32
            inputs[key] = tf.keras.Input(shape, dtype=dtype, name=key)
        return self(inputs)
