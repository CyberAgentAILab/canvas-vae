import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def make_feature(value, encoding='ascii', dtype=None):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if isinstance(value, str):
        value = value.encode(encoding)

    if not isinstance(value, (list, tuple)):
        value = [value]

    if dtype:
        value = [dtype(x) for x in value]

    if dtype is int or (len(value) and isinstance(value[0], int)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    elif dtype is float or (len(value) and isinstance(value[0], float)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    elif dtype in (bytes, str) or (len(value) and isinstance(value[0], bytes)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    raise ValueError('Unsupported type: %s' % value)


def make_feature_list(value, field, default=None, **kwargs):
    return tf.train.FeatureList(
        feature=[make_feature(e.get(field, default), **kwargs) for e in value])
