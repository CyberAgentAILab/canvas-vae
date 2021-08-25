import logging

import tensorflow as tf

from .mask import get_mask

logger = logging.getLogger(__name__)


def _get_mask(x):
    if isinstance(x, tuple):
        x, mask = x
    elif hasattr(x, '_keras_mask'):
        mask = x._keras_mask
    else:
        # No mask found, assume all valid tokens.
        mask = tf.ones(tf.shape(x)[0:2], dtype=tf.bool)
    return x, mask


class LossLayer(tf.keras.layers.Layer):
    """Teacher-forced reconstruction loss layer."""
    def __init__(self, input_columns, name='loss_layer', **kwargs):
        super().__init__(name=name, **kwargs)
        self._input_columns = input_columns

    def call(self, inputs):
        y_true, y_pred = inputs
        mask = get_mask(y_true['length'])

        loss_total = 0
        for key, column in self._input_columns.items():
            prediction = y_pred[key]
            if column['is_sequence']:
                # Cut extra elements in prediction.
                prediction = prediction[:, :tf.shape(mask)[1]]

            if column['type'] == 'categorical':
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true[key], prediction, from_logits=True)
            else:
                loss = tf.expand_dims(
                    tf.keras.losses.mse(y_true[key], prediction),
                    -1) * tf.cast(column['shape'][-1], tf.float32)

            if 'loss_condition' in column:
                cond = column['loss_condition']
                weight = tf.gather(cond['mask'], y_true[cond['key']])
                loss *= tf.cast(weight, tf.float32)

            if column['is_sequence']:
                weight = tf.cast(mask[:, :, tf.newaxis], tf.float32)
                loss = tf.reduce_sum(loss * weight, axis=1)  # sum timesteps

            loss = tf.reduce_sum(loss, axis=1)  # sum features
            tf.debugging.assert_rank(loss, 1)

            loss = tf.reduce_mean(loss)  # average batch
            loss_total += loss

            self.add_metric(loss, name=key + '_loss')

        self.add_loss(loss_total)

        return inputs


@tf.function(experimental_relax_shapes=True)
def _compute_bow(x, mask, depth):
    """Compute bag-of-words for the batch."""
    s = tf.shape(x)
    mask = tf.cast(mask, dtype=tf.int32)
    batch_index = tf.tile(
        tf.range(s[0], dtype=tf.int32)[:, tf.newaxis, tf.newaxis],
        [1, s[1], s[2]],
    )
    vector_index = tf.tile(
        tf.range(s[2], dtype=tf.int32)[tf.newaxis, tf.newaxis, :],
        [s[0], s[1], 1],
    )
    indices = tf.stack(
        [batch_index, tf.cast(x, tf.int32), vector_index], axis=-1)
    updates = tf.tile(mask[:, :, tf.newaxis], [1, 1, s[2]])
    return tf.scatter_nd(indices, updates, (s[0], depth, s[2]))


@tf.function(experimental_relax_shapes=True)
def bleu1(y_true, y_pred):
    """Compute unigram BLEU score for each example."""
    y_true, mask_true = _get_mask(y_true)
    y_pred, mask_pred = _get_mask(y_pred)
    depth = tf.shape(y_pred)[-1]
    y_pred = tf.argmax(y_pred, axis=-1)

    tf.debugging.assert_equal(tf.shape(y_pred)[:2], tf.shape(mask_pred))

    len_true = tf.reduce_sum(tf.cast(mask_true, tf.float32), axis=1) + 1e-9
    len_pred = tf.reduce_sum(tf.cast(mask_pred, tf.float32), axis=1) + 1e-9

    bow_true = _compute_bow(y_true, mask_true, depth)
    bow_pred = _compute_bow(y_pred, mask_pred, depth)

    match = tf.reduce_sum(tf.minimum(bow_true, bow_pred), axis=1)
    prec = tf.cast(match, tf.float32) / len_pred[:, tf.newaxis]
    len_scale = tf.exp(tf.minimum(0., 1. - len_true / len_pred))
    score = len_scale[:, tf.newaxis] * tf.sqrt(prec)
    return tf.clip_by_value(score, 0.0, 1.0)


@tf.function(experimental_relax_shapes=True)
def scaled_mean_cosine_similarity(y_true, y_pred):
    """Compute cosine similarity of sequence mean, scaled by length factor."""
    y_true, mask_true = _get_mask(y_true)
    y_pred, mask_pred = _get_mask(y_pred)
    len_true = tf.reduce_sum(tf.cast(mask_true, tf.float32), axis=1) + 1e-9
    len_pred = tf.reduce_sum(tf.cast(mask_pred, tf.float32), axis=1) + 1e-9
    mask_true = tf.cast(mask_true, tf.float32)[:, :, tf.newaxis]
    mask_pred = tf.cast(mask_pred, tf.float32)[:, :, tf.newaxis]
    a_true = tf.reduce_sum(y_true * mask_true, axis=1) / len_true[:,
                                                                  tf.newaxis]
    a_pred = tf.reduce_sum(y_pred * mask_pred, axis=1) / len_pred[:,
                                                                  tf.newaxis]
    # Normalize to [0, 1] range.
    similarity = -.5 * tf.keras.losses.cosine_similarity(a_true, a_pred) + .5
    score = tf.exp(tf.minimum(0., 1. - len_true / len_pred)) * similarity
    return tf.clip_by_value(score[:, tf.newaxis], 0.0, 1.0)


class VectorMetricLayer(tf.keras.layers.Layer):
    """Reconstruction metric."""
    def __init__(
        self,
        input_columns,
        from_logits=True,
        name='metric_layer',
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._input_columns = input_columns
        self._from_logits = from_logits

    def call(self, inputs, training=False):
        y_true, y_pred = inputs
        mask_true, mask_pred = self._get_masks(y_true, y_pred, training)

        metrics, weights = {}, {}
        for key, column in self._input_columns.items():
            if key == 'length':
                continue
            pred = y_pred[key]
            if column['type'] == 'categorical' and not self._from_logits:
                pred = tf.one_hot(y_pred[key], column['input_dim'])

            if not column['is_sequence']:
                metric = tf.keras.metrics.sparse_categorical_accuracy(
                    y_true[key],
                    pred,
                )
                weights[key] = tf.ones_like(metric)
            else:
                m_true, m_pred = self._conditional_mask(
                    column,
                    y_true,
                    y_pred,
                    mask_true,
                    mask_pred,
                )
                if column['type'] == 'categorical':
                    metric = bleu1((y_true[key], m_true), (pred, m_pred))
                else:
                    metric = scaled_mean_cosine_similarity(
                        (y_true[key], m_true),
                        (pred, m_pred),
                    )
                # When there is no element for this key, apply zero-weight.
                weight = tf.cast(
                    tf.math.logical_and(
                        tf.reduce_any(m_true, axis=1, keepdims=True),
                        tf.reduce_any(m_pred, axis=1, keepdims=True),
                    ),
                    tf.float32,
                )
                weights[key] = tf.tile(weight, [1, tf.shape(metric)[1]])

            metrics[key] = metric
            self.add_metric(metric, name=key + '_score')

        # Average over valid entries.
        total = tf.reduce_sum(tf.concat(list(metrics.values()), axis=1),
                              axis=1,
                              keepdims=True)
        weights = tf.reduce_sum(tf.concat(list(weights.values()), axis=1),
                                axis=1,
                                keepdims=True)
        mean = total / weights
        metrics['total'] = mean
        self.add_metric(mean, name='total_score')
        return metrics

    def _get_masks(self, y_true, y_pred, training):
        maxlen = tf.reduce_max([
            tf.shape(y_true[key])[1]
            for key, column in self._input_columns.items()
            if column['is_sequence']
        ])
        mask_true = get_mask(y_true['length'], maxlen=maxlen)
        if training:
            mask_pred = mask_true
        else:
            maxlen = tf.reduce_max([
                tf.shape(y_pred[key])[1]
                for key, column in self._input_columns.items()
                if column['is_sequence']
            ])
            mask_pred = get_mask(
                y_pred['length'],
                from_logits=self._from_logits,
                maxlen=maxlen,
            )
        tf.debugging.assert_rank(mask_true, 2)
        tf.debugging.assert_rank(mask_pred, 2)
        return mask_true, mask_pred

    def _conditional_mask(self, column, y_true, y_pred, mask_true, mask_pred):
        m_true = mask_true
        m_pred = mask_pred

        if 'loss_condition' in column:
            cond = column['loss_condition']
            m_true = tf.math.logical_and(
                m_true,
                tf.gather(cond['mask'], y_true[cond['key']])[:, :, 0],
            )
            cond_pred = y_pred[cond['key']]
            if self._from_logits:
                cond_pred = tf.argmax(cond_pred, axis=-1)

            m_pred = tf.math.logical_and(
                m_pred,
                tf.gather(cond['mask'], cond_pred)[:, :, 0],
            )
        return m_true, m_pred


class LayoutMetricLayer(tf.keras.layers.Layer):
    """Compute Accuracy and mean IoU of the layout map."""
    def __init__(self, input_columns, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self._xsize = tf.cast(input_columns['left']['input_dim'], tf.int32)
        self._ysize = tf.cast(input_columns['top']['input_dim'], tf.int32)
        self._label_name = next(key for key, c in input_columns.items()
                                if c['primary_label'] is not None)
        self._default_label = tf.cast(
            input_columns[self._label_name]['primary_label'], tf.int32)
        self._label_size = tf.cast(
            input_columns[self._label_name]['input_dim'], tf.int32)
        self._from_logits = from_logits
        assert input_columns['left']['input_dim'] == input_columns['width'][
            'input_dim']
        assert input_columns['top']['input_dim'] == input_columns['height'][
            'input_dim']

    def call(self, inputs, training=False):
        y_true, y_pred = inputs
        mask_true, mask_pred = self._get_masks(y_true, y_pred, training)
        map_true = _compute_gridmaps(
            y_true,
            mask_true,
            from_logits=False,
            label_name=self._label_name,
            xsize=self._xsize,
            ysize=self._ysize,
            default_label=self._default_label,
        )
        map_pred = _compute_gridmaps(
            y_pred,
            mask_pred,
            from_logits=self._from_logits,
            label_name=self._label_name,
            xsize=self._xsize,
            ysize=self._ysize,
            default_label=self._default_label,
        )
        acc, miou = _compute_acc_miou(map_true, map_pred, self._label_size)
        self.add_metric(acc, name='layout_acc')
        self.add_metric(miou, name='layout_miou')
        return {'layout_acc': acc, 'layout_miou': miou}

    def _get_masks(self, y_true, y_pred, training):
        maxlen = tf.shape(y_true[self._label_name])[1]
        mask_true = get_mask(y_true['length'], maxlen=maxlen)
        if training:
            mask_pred = mask_true
        else:
            maxlen = tf.shape(y_pred[self._label_name])[1]
            mask_pred = get_mask(
                y_pred['length'],
                from_logits=self._from_logits,
                maxlen=maxlen,
            )
        tf.debugging.assert_rank(mask_true, 2)
        tf.debugging.assert_rank(mask_pred, 2)
        return mask_true, mask_pred


@tf.function(experimental_relax_shapes=True)
def _compute_gridmaps(
    example,
    mask,
    from_logits,
    label_name,
    xsize,
    ysize,
    default_label,
):
    if from_logits:
        # Assume all categorical here.
        example = {
            key: tf.cast(
                tf.argmax(tf.stop_gradient(example[key]), axis=-1),
                tf.int32,
            )
            for key in ('left', 'top', 'width', 'height', label_name)
        }
    else:
        example = {
            key: tf.cast(tf.stop_gradient(example[key]), tf.int32)
            for key in ('left', 'top', 'width', 'height', label_name)
        }

    batch_size = tf.shape(mask)[0]
    gridmaps = tf.TensorArray(tf.int32, size=batch_size)
    for i in tf.range(batch_size):
        left = tf.reshape(example['left'][i][mask[i]], (-1, ))
        top = tf.reshape(example['top'][i][mask[i]], (-1, ))
        width = tf.reshape(example['width'][i][mask[i]], (-1, ))
        height = tf.reshape(example['height'][i][mask[i]], (-1, ))

        label = tf.cast(
            tf.reshape(example[label_name][i][mask[i]], (-1, )),
            tf.int32,
        )
        tf.assert_rank(left, 1)

        right = tf.minimum(xsize - 1, left + width)
        bottom = tf.minimum(ysize - 1, top + height)

        gridmap = _make_gridmap(
            left,
            top,
            right,
            bottom,
            label,
            ysize,
            xsize,
            default_label,
        )
        gridmaps = gridmaps.write(i, gridmap)
    return gridmaps.stack()


@tf.function(experimental_relax_shapes=True)
def _make_gridmap(left, top, right, bottom, label, ysize, xsize,
                  default_label):
    # Fill bbox region with the specified label.
    canvas = tf.fill((ysize, xsize), default_label)
    for j in tf.range(tf.shape(label)[0]):
        if top[j] >= bottom[j] or left[j] >= right[j]:
            continue
        y, x = tf.meshgrid(
            tf.range(top[j], bottom[j] + 1),
            tf.range(left[j], right[j] + 1),
        )
        indices = tf.stack(
            [tf.reshape(y,
                        (-1, )), tf.reshape(x, (-1, ))], axis=1)
        updates = tf.fill((tf.shape(indices)[0], ), label[j])
        canvas = tf.tensor_scatter_nd_update(canvas, indices, updates)
    return canvas


@tf.function
def _compute_acc_miou(map_true, map_pred, label_size):
    batch_size = tf.shape(map_pred)[0]
    batch_index = tf.reshape(
        tf.tile(
            tf.range(batch_size)[:, tf.newaxis], [1, tf.size(map_pred[0])]),
        (-1, ))
    indices = tf.stack([
        tf.cast(batch_index, tf.int32),
        tf.reshape(map_pred, (-1, )),
        tf.reshape(map_true, (-1, )),
    ],
                       axis=1)
    updates = tf.ones((tf.shape(indices)[0], ), dtype=tf.int32)
    confusion = tf.cast(
        tf.scatter_nd(indices, updates, (batch_size, label_size, label_size)),
        tf.float32)

    inter = tf.linalg.diag_part(confusion)
    union = tf.reduce_sum(confusion, axis=1) + tf.reduce_sum(confusion,
                                                             axis=2) - inter

    # Compute accuracy
    acc = tf.math.truediv(tf.reduce_sum(inter, axis=1),
                          tf.reduce_sum(confusion, axis=(1, 2)))

    # Compute nanmean of iou.
    weight = tf.cast(union > 0, tf.float32)
    iou = inter / (union + 1e-9)
    miou = tf.reduce_sum(weight * iou, axis=1) / tf.reduce_sum(weight, axis=1)
    return acc, miou
