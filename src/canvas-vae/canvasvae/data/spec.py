import logging
import json
import os

import numpy as np
import yaml
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from ..models.discretizer import SequenceDiscretizer

logger = logging.getLogger(__name__)


class DataSpec(object):
    """
    Utility class to handle data schema.

    We assume the following directory structure::

        root/
        root/count.json
        root/vocabulary.json
        root/train-*-of-*.tfrecord
        root/val-*-of-*.tfrecord
        root/test-*-of-*.tfrecord

    Additionally, there must be a spec file in YAML format (loaded via name)::

        name: rico
        columns:
          column1:
            shape: []
            dtype: int64
          column2:
            is_sequence: true
            dtype: string
            lookup:
              num_oov_indices: 1
              mask_token: null

    Usage::

        dataspec = DataSpec('crello', '/path/to/tfrecords', batch_size=256)

        train_dataset = dataspec.make_dataset('train', shuffle=True, cache=True)
        batch = next(iter(train_dataset))

        for item in dataspec.unbatch(batch):
            print(item)
    """
    def __init__(
        self,
        name,
        path,
        batch_size=8,
    ):
        self._path = path
        self._batch_size = batch_size

        spec_path = os.path.join(os.path.dirname(__file__), name + '-spec.yml')
        if os.path.exists(spec_path):
            name = spec_path
        self._spec = self._load_resource(spec_path, rel_path=False)
        self._splits = self._load_resource('count.json')
        self._init_preprocessor()

    @property
    def columns(self):
        return self._spec.get('columns', {})

    @property
    def preprocessor(self):
        return self._preprocessor

    def _init_preprocessor(self):
        # Initialize preprocessing functions.
        vocabulary = self._load_resource('vocabulary.json')

        self._preprocessor = {}
        for name, column in self.columns.items():
            if 'lookup' in column:
                self._preprocessor[name] = self._create_lookup(
                    name, column, vocabulary)
            elif 'discretize' in column:
                spec = column['discretize']
                boundaries = list(
                    np.linspace(spec['min'], spec['max'], spec['bins']))[1:]
                self._preprocessor[name] = SequenceDiscretizer(boundaries)
                logger.info('Discretizer for %s: bins=%s' %
                            (name, len(boundaries) + 1))

    def _create_lookup(self, name, column, vocabulary):
        assert name in vocabulary or 'vocabulary' in column['lookup']
        layer_fn = {
            'string': preprocessing.StringLookup,
            'int64': preprocessing.IntegerLookup,
        }[column['dtype']]

        if name in vocabulary:
            vocab = vocabulary[name]
        else:
            # Integer [min, max] vocabulary.
            min_value = column['lookup']['vocabulary']['min']
            max_value = column['lookup']['vocabulary']['max']
            vocab = list(range(min_value, max_value + 1))
        if isinstance(vocab, dict):
            vocab = [
                int(key) if column['dtype'] == 'int64' else key
                for key, value in vocab.items()
                if value >= column.get('min_freq', 1)
            ]
        options = {} if column['lookup'] is True else {
            k: v
            for k, v in column['lookup'].items() if k != 'vocabulary'
        }
        logger.info('Lookup for %s: vocabulary_size=%s, options=%s' %
                    (name, len(vocab), options))
        return layer_fn(vocabulary=vocab, **options)

    def size(self, split):
        """Length of the records for the split."""
        return self._splits[split]

    def steps_per_epoch(self, split, batch_size=None):
        """Steps per epoch."""
        return int(np.ceil(
            self.size(split) / (batch_size or self._batch_size)))

    def make_input_columns(self):
        """Returns input specification for a model."""
        inputs = {}
        for key, column in self.columns.items():
            # Inspect categorical inputs and its size.
            layer = self._preprocessor.get(key)
            if isinstance(layer, SequenceDiscretizer):
                inputs[key] = {
                    'type': 'categorical',
                    'input_dim': len(layer.bin_boundaries) + 1,
                }
            elif isinstance(layer, (
                    preprocessing.StringLookup,
                    preprocessing.IntegerLookup,
            )):
                if tf.__version__.split('.')[1] in ('3', '4'):
                    vocabulary_size = layer.vocab_size()
                else:
                    vocabulary_size = layer.vocabulary_size()
                inputs[key] = {
                    'type': 'categorical',
                    'input_dim': vocabulary_size,
                }
            elif column['dtype'] in ('int', 'int32', 'int64'):
                inputs[key] = {
                    'type': 'categorical',
                    'input_dim': column['max'] + 1,  # Include zero.
                }
            else:
                assert column['dtype'] in ('float', 'float32', 'float64')
                inputs[key] = {
                    'type': 'numerical',
                }
            inputs[key]['shape'] = tuple(column.get('shape', (1, )))
            inputs[key]['is_sequence'] = column.get('is_sequence', False)

            if 'primary_label' in column:
                inputs[key]['primary_label'] = (self._preprocessor[key](
                    [column['primary_label']['default']]))[0]
            else:
                inputs[key]['primary_label'] = None

        for key, column in self.columns.items():
            if 'loss_condition' in column:
                cond = column['loss_condition']
                logger.info('Loss condition for %s: %s included in %s' %
                            (key, cond['key'], cond['values']))
                mask = [
                    v in cond['values']
                    for v in self._preprocessor[cond['key']].get_vocabulary()
                ]
                inputs[key]['loss_condition'] = {
                    'key': cond['key'],
                    'mask': mask,
                }

        return inputs

    def make_dataset(
        self,
        split,
        batch_size=None,
        shuffle=None,
        repeat=False,
        prefetch=tf.data.experimental.AUTOTUNE,
        parallel=None,
        cache=None,
    ):
        assert split in self._splits, ('split must be one of (%s)' %
                                       ', '.join(self._splits.keys()))
        if shuffle is True:
            shuffle = self.size(split)
        if parallel is None:
            parallel = (tf.data.experimental.AUTOTUNE if shuffle else None)

        file_pattern = os.path.join(self._path, split + '-*.tfrecord')
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
        dataset = dataset.batch(batch_size or self._batch_size)
        dataset = dataset.map(self.parse_fn,
                              num_parallel_calls=parallel,
                              deterministic=(shuffle is False))
        if prefetch:
            dataset = dataset.prefetch(prefetch)
        return dataset

    def parse_fn(self, serialized):
        context, sequence, _ = tf.io.parse_sequence_example(
            serialized, {
                name: tf.io.FixedLenFeature(column.get('shape',
                                                       (1, )), column['dtype'])
                for name, column in self.columns.items()
                if not column.get('is_sequence')
            }, {
                name: tf.io.FixedLenSequenceFeature(column.get('shape', (1, )),
                                                    column['dtype'])
                for name, column in self.columns.items()
                if column.get('is_sequence')
            })
        output = context
        output.update(sequence)

        for key, preprocess_fn in self._preprocessor.items():
            output[key] = preprocess_fn(output[key])
        return output

    def logit_to_label(self, example):
        """Convert logit prediction to labels."""
        for key, column in self.columns.items():
            rank = 1 + column.get('is_sequence', 0) + len(
                column.get('shape', (1, )))
            if tf.rank(example[key]) >= rank + 1:
                example[key] = tf.cast(tf.argmax(example[key], axis=-1),
                                       tf.int32)
        return example

    def unbatch(self, example):
        """
        Convert a batch tensor example to a list of items for post-processing.

        Sequence items get stored in `elements` while others are in dict::

            items = [{key: value, 'elements': [{key: value}]}]
        """
        example = self.logit_to_label(example)
        batch_size = tf.shape(example['length'])[0]

        items = []
        for i in range(batch_size):
            # Find length.
            length = int(tf.squeeze(example['length'][i]) + 1)  # zero-based
            for name, column in self.columns.items():
                if column.get('is_sequence'):
                    length = min(length, tf.shape(example[name][i])[0])
                    break

            # Fill in items.
            item = {'elements': [{} for _ in range(length)]}
            for name, column in self.columns.items():
                x = example[name][i].numpy()

                # Un-preprocess.
                if 'lookup' in column:
                    layer = self._preprocessor.get(name)
                    table = np.array(layer.get_vocabulary())
                    x = table[x]
                elif 'discretize' in column:
                    spec = column['discretize']
                    scale = (spec['max'] - spec['min']) / (spec['bins'] - 1.)
                    x = scale * x + spec['min']

                if column.get('is_sequence'):
                    for j in range(length):
                        item['elements'][j][name] = x[
                            j, :].tolist() if x.shape[1] > 1 else x[j, 0]
                else:
                    item[name] = x[0]
            items.append(item)
        return items

    def _load_resource(self, path, format=None, rel_path=True):
        """Load resource file."""
        format = format or os.path.splitext(path)[-1]
        format = format.replace('.', '').lower()

        if rel_path:
            path = os.path.join(self._path, path)
        logger.info('Loading resource at %s' % path)
        with tf.io.gfile.GFile(path) as f:
            if format == 'json':
                return json.load(f)
            elif format in ('yml', 'yaml'):
                return yaml.safe_load(f)
            else:
                logger.warning('Unsupported format: %s' % path)
                return f.read()
