import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


class HyperTune(tf.keras.callbacks.Callback):
    """Callback for HyperTune on AI Platform."""
    def __init__(self, metric, tag=None, logdir=None, **kwargs):
        super().__init__(**kwargs)
        self._metric = metric
        self._tag = tag or 'training/hptuning/metric'
        self._logdir = logdir or '/tmp/hypertune/output.metrics'
        self._writer = tf.summary.create_file_writer(self._logdir)

    def on_epoch_end(self, epoch, logs=None):
        if logs and self._metric in logs:
            with self._writer.as_default():
                tf.summary.scalar(self._tag, logs[self._metric], step=epoch)
