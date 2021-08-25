import logging
import json
import os

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DatasetBuilder(object):
    """
    Dataset utility class.

    Usage::
        builder = DatasetBuilder('gs://bucket/to/data')
        dataset = builder.build('train')
    """
    def __init__(self, path, batch_size=64):
        self.path = path
        self.batch_size = batch_size
        self.splits = self._load_json('count.json')

    def build(
        self,
        split,
        batch_size=None,
        shuffle=None,
        repeat=False,
        prefetch=tf.data.experimental.AUTOTUNE,
        cache=None,
    ):
        assert split in self.splits, ('split must be one of (%s)' %
                                      ', '.join(self.splits.keys()))
        if shuffle is True:
            shuffle = self.size(split)

        parallel = (tf.data.experimental.AUTOTUNE if shuffle else None)
        file_pattern = os.path.join(self.path, split + '-*.tfrecord')
        logger.info('TFRecord from %s' % file_pattern)
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
        dataset = tf.data.TFRecordDataset(
            dataset,
            num_parallel_reads=parallel,
        )
        if cache:
            dataset = dataset.cache()
        if shuffle:
            dataset = dataset.shuffle(shuffle)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.map(
            self.make_parse_fn(),
            num_parallel_calls=parallel,
        )
        dataset = dataset.batch(batch_size or self.batch_size)
        if prefetch:
            dataset = dataset.prefetch(prefetch)
        return dataset

    def make_parse_fn(self):
        def _parse(serialized):
            example = tf.io.parse_single_example(
                serialized,
                {'image': tf.io.FixedLenFeature((), tf.string)},
            )
            image = tf.io.decode_png(example['image'], channels=4)
            return image

        return _parse

    def size(self, split):
        """Length of the records for the split."""
        return self.splits.get(split)

    def steps_per_epoch(self, split, batch_size=None):
        """Steps per epoch."""
        return int(np.ceil(self.size(split) / (batch_size or self.batch_size)))

    def _load_json(self, file_name):
        """Load json resource file."""
        path = os.path.join(self.path, file_name)
        logger.info('Reading %s' % path)
        with tf.io.gfile.GFile(path) as f:
            return json.load(f)
