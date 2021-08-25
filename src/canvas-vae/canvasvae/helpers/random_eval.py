import logging

import tensorflow as tf

from canvasvae.models.mask import get_mask

logger = logging.getLogger(__name__)


def compute_categorical_histogram(values, num_categories, normalize=False):
    values = tf.cast(values, tf.int64)
    batch_size = tf.shape(values)[0]
    feature_dim = tf.shape(values)[1]

    feature_index = tf.tile(
        tf.range(feature_dim, dtype=tf.int64)[:, tf.newaxis],
        [batch_size, 1],
    )
    indices = tf.stack([
        tf.reshape(values, (-1, )),
        tf.reshape(feature_index, (-1, )),
    ],
                       axis=1)
    updates = tf.ones((batch_size * feature_dim, ), dtype=tf.int64)
    h = tf.scatter_nd(indices, updates, (num_categories, feature_dim))
    if normalize:
        h = tf.cast(h, tf.float32)
        return h / tf.reduce_sum(h, axis=0, keepdims=True)
    return h


def compute_mean_covariance(x):
    mu = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mu), mu)
    vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float32)
    sigma = vx - mx
    return mu, sigma


def get_conditional_mask(column, inputs, mask, from_logits=False):
    if 'loss_condition' in column:
        cond = column['loss_condition']
        control = inputs[cond['key']]
        if from_logits:
            control = tf.argmax(control, axis=-1)
        mask = tf.math.logical_and(
            mask,
            tf.gather(cond['mask'], control)[:, :, 0],
        )
    return mask


def _logits_to_label(input_columns, inputs, from_logits):
    if from_logits:
        outputs = {}
        for key, column in input_columns.items():
            if column['type'] == 'categorical':
                outputs[key] = tf.argmax(inputs[key], axis=-1)
            else:
                outputs[key] = inputs[key]
        return outputs
    else:
        return inputs


def compute_batch_statistics(input_columns, inputs, from_logits=False):
    """Compute batch statistics."""
    inputs = _logits_to_label(input_columns, inputs, from_logits)
    maxlen = tf.reduce_max([
        tf.shape(inputs[key])[1] for key, column in input_columns.items()
        if column['type'] == 'categorical' and column['is_sequence']
    ])
    mask = get_mask(inputs['length'], maxlen=maxlen)

    stats = {}
    for key in inputs:
        column = input_columns[key]
        if not column['is_sequence']:
            stats[key] = compute_categorical_histogram(
                inputs[key],
                column['input_dim'],
                normalize=True,
            )
        else:
            mask_k = get_conditional_mask(column, inputs, mask)
            values = tf.boolean_mask(inputs[key], mask_k)
            if column['type'] == 'categorical':
                stats[key] = compute_categorical_histogram(
                    values,
                    column['input_dim'],
                    normalize=True,
                )
            else:
                stats[key] = tf.reduce_mean(values, axis=0, keepdims=True)
    return stats


def compare_batch_statistics(input_columns, s_true, s_pred):
    """Compare batch statistics"""
    scores = {}
    for key, column in input_columns.items():
        if column['type'] == 'categorical':
            scores[key] = tf.reduce_sum(tf.minimum(s_true[key], s_pred[key]),
                                        axis=0)
        else:
            scores[key] = .5 + .5 * tf.keras.losses.cosine_similarity(
                s_true[key], s_pred[key])
    scores['total'] = tf.reduce_mean(tf.concat(list(scores.values()), axis=0),
                                     keepdims=True)
    return {k: tf.reduce_mean(v) for k, v in scores.items()}


def evaluate_random_generation(
    model,
    dataspec,
    split,
    sample_size=None,
    prefix='random_',
):
    logger.info('Computing %s statistics' % split)
    data_size = dataspec.size(split)
    dataset = dataspec.make_dataset(split, batch_size=data_size)
    s_true = compute_batch_statistics(
        model.input_columns,
        next(iter(dataset)),
        from_logits=False,
    )
    z = tf.random.normal(
        shape=(sample_size or data_size, model.decoder.latent_dim),
        dtype=tf.float32,
    )
    s_pred = compute_batch_statistics(
        model.input_columns,
        model.decoder(z),
        from_logits=True,
    )
    scores = compare_batch_statistics(model.input_columns, s_true, s_pred)
    logger.info('Random generation - ' +
                ' - '.join('%s: %.4f' % (k, v) for k, v in scores.items()))
    return {prefix + key: float(scores[key].numpy()) for key in scores}


class RandomEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 dataspec,
                 split='val',
                 sample_size=None,
                 prefix='random_',
                 logdir=None,
                 **kwargs):
        super().__init__(**kwargs)
        self._split = split
        self._sample_size = sample_size or dataspec.size(self._split)
        self._prefix = prefix
        self._input_columns = dataspec.make_input_columns()
        self._reference_stats = self._compute_reference_stats(dataspec)

        # For HyperTune.
        self._logdir = logdir or '/tmp/hypertune/output.metrics'
        self._writer = tf.summary.create_file_writer(self._logdir)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or not any(k.startswith('val') for k in logs):
            return

        z = tf.random.normal(shape=(self._sample_size,
                                    self.model.decoder.latent_dim))
        prediction = self.model.decoder(z)
        prediction_stats = compute_batch_statistics(
            self._input_columns,
            prediction,
            from_logits=True,
        )
        scores = compare_batch_statistics(self._input_columns,
                                          self._reference_stats,
                                          prediction_stats)
        logger.info('Random generation - ' +
                    ' - '.join('%s: %.4f' % (k, v) for k, v in scores.items()))

        # Report (val_total_score + random_total) / 2 to HyperTune.
        if 'val_total_score' in logs:
            objective = 0.5 * (float(scores['total'].numpy()) +
                               float(logs['val_total_score']))
            with self._writer.as_default():
                tf.summary.scalar(
                    'training/hptuning/metric',
                    objective,
                    step=epoch,
                )

    def _compute_reference_stats(self, dataspec):
        logger.info('Computing %s statistics' % self._split)
        data_size = dataspec.size(self._split)
        dataset = dataspec.make_dataset(self._split, batch_size=data_size)
        return compute_batch_statistics(
            self._input_columns,
            next(iter(dataset)),
            from_logits=False,
        )
