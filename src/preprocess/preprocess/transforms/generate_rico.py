import logging
import json
import zipfile
import os

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
import tensorflow as tf
import numpy as np

from preprocess.helpers.hash import stable_hash
from preprocess.helpers.feature import make_feature, make_feature_list
from preprocess.transforms.vocabulary import WriteVocabularyToFile
from preprocess.transforms.count import WriteCountToFile

logger = logging.getLogger(__name__)


class UnZipFn(beam.DoFn):
    def __init__(self, extname='.json', **kwargs):
        super().__init__(**kwargs)
        self.extname = extname

    def process(self, file_path):
        with FileSystems.open(file_path) as f:
            with zipfile.ZipFile(f) as z:
                for name in z.namelist():
                    if not name.endswith(self.extname):
                        continue
                    key = os.path.splitext(os.path.basename(name))[0]
                    content = z.open(name).read()
                    yield stable_hash(content), (key, content)


class ReadJsonFromZipArchive(beam.PTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def expand(self, pcoll):
        return (pcoll
                | 'UnZipJson' >> beam.ParDo(UnZipFn(extname='.json'))
                | 'GroupByKey' >> beam.GroupByKey())  # Deduplicate if any.


class _MakeRecordFn(beam.DoFn):
    """Make TFRecord for each template."""
    def __init__(self, max_seq_length, width=1440., height=2560.):
        self.width = width
        self.height = height
        self.max_seq_length = max_seq_length

    def process(self, hash_key_serialized):
        _, key_serialized_group = hash_key_serialized
        key, serialized = next(iter(key_serialized_group))
        split = {
            0: 'val',
            1: 'test'
        }.get((stable_hash(serialized) % 10), 'train')

        data = json.loads(serialized)
        elements = list(self.extract_elements(data))

        if len(elements) > self.max_seq_length:
            logger.warning('#elements > %d: %s has %d elements' %
                           (self.max_seq_length, key, len(elements)))
            return

        feature = {
            'id': make_feature(int(key)),
            'length': make_feature(len(elements)),
            'canvas_width': make_feature(self.width),
            'canvas_height': make_feature(self.height),
        }
        feature_list = {
            'left':
            make_feature_list(elements, 'left', dtype=float),
            'top':
            make_feature_list(elements, 'top', dtype=float),
            'width':
            make_feature_list(elements, 'width', dtype=float),
            'height':
            make_feature_list(elements, 'height', dtype=float),
            'class':
            make_feature_list(elements, 'class', encoding='utf-8'),
            'clickable':
            make_feature_list(elements, 'clickable', dtype=int),
            'component':
            make_feature_list(elements, 'component', encoding='utf-8'),
            'icon':
            make_feature_list(elements, 'icon', encoding='utf-8'),
            'text':
            make_feature_list(elements, 'text', encoding='utf-8'),
            'text_button':
            make_feature_list(elements, 'text_button', encoding='utf-8'),
        }
        yield split, tf.train.SequenceExample(
            context=tf.train.Features(feature=feature),
            feature_lists=tf.train.FeatureLists(feature_list=feature_list),
        ).SerializeToString()

    def extract_elements(self, element):
        bounds = element.get('bounds')
        yield {
            'left': bounds[0] / self.width,
            'top': bounds[1] / self.height,
            'width': (bounds[2] - bounds[0]) / self.width,
            'height': (bounds[3] - bounds[1]) / self.height,
            'class': element['class'],  # element['ancestors'],
            'clickable': element.get('clickable', False),
            'component': element.get('componentLabel', ''),
            'icon': element.get('iconClass', ''),
            'text': element.get('text', ''),
            'text_button': element.get('textButtonClass', ''),
        }
        if 'children' in element:
            for e in element['children']:
                yield from self.extract_elements(e)


class GenerateRicoDataset(beam.PTransform):
    def __init__(
        self,
        input_path,
        output_path,
        num_shards=8,
        max_seq_length=50,
        **kwargs,
    ):
        super().__init__()
        self._input_path = input_path
        self._output_path = output_path
        self._num_shards = num_shards
        self._max_seq_length = max_seq_length
        self._split_labels = ['train', 'val', 'test']

    def expand(self, pcoll):
        records = (pcoll
                   | beam.Create([self._input_path])
                   | 'ReadJsonFromZipArchive' >> ReadJsonFromZipArchive()
                   | 'MakeRecordFn' >> beam.ParDo(
                       _MakeRecordFn(max_seq_length=self._max_seq_length)))
        # Write TFRecord for each split.
        split_records = (records
                         | 'Partition' >> beam.Partition(
                             lambda kv, num: self._split_labels.index(kv[0]),
                             len(self._split_labels),
                         ))

        output = []
        for label, p in zip(self._split_labels, split_records):
            output.append(
                p
                | 'Values:%s' % label >> beam.transforms.util.Values()
                | 'WriteToTFRecords:%s' % label >> beam.io.WriteToTFRecord(
                    os.path.join(self._output_path, label),
                    file_name_suffix='.tfrecord',
                    num_shards=self._num_shards if label == 'train' else 1,
                ))
        # Write global vocabulary.
        output.append(
            records
            | WriteVocabularyToFile(
                os.path.join(self._output_path, 'vocabulary'),
                context_spec={},
                sequence_spec={
                    'class': tf.io.FixedLenSequenceFeature((), tf.string),
                    'component': tf.io.FixedLenSequenceFeature((), tf.string),
                    'icon': tf.io.FixedLenSequenceFeature((), tf.string),
                    'text_button': tf.io.FixedLenSequenceFeature(
                        (), tf.string),
                },
            ))
        # Write count per split.
        output.append(
            records
            | WriteCountToFile(os.path.join(self._output_path, 'count')))
        return tuple(output)
