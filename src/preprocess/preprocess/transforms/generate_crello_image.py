import logging
import os

import apache_beam as beam
import tensorflow as tf

from preprocess.helpers.feature import make_feature
from preprocess.transforms.count import WriteCountToFile

logger = logging.getLogger(__name__)


class _MakeRecordFn(beam.DoFn):
    """Extract image record from each element."""
    def __init__(self, element_types=None):
        self._element_types = set(element_types or (
            b'imageElement',
            b'maskElement',
            b'svgElement',
        ))

    def process(self, serialized):
        _, example = tf.io.parse_single_sequence_example(
            serialized,
            sequence_features={
                'type': tf.io.FixedLenSequenceFeature((), tf.string),
                'image_bytes': tf.io.FixedLenSequenceFeature((), tf.string),
            },
        )
        for element_type, image_bytes in zip(
                example['type'].numpy().tolist(),
                example['image_bytes'].numpy().tolist(),
        ):
            if element_type in self._element_types:
                yield tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': make_feature(image_bytes),
                    }), ).SerializeToString()


class GenerateSingleImageDataset(beam.PTransform):
    def __init__(
        self,
        input_path,
        output_path,
        **kwargs,
    ):
        super().__init__()
        self._input_path = input_path
        self._output_path = output_path

    def expand(self, pcoll):
        records = (pcoll
                   |
                   'ReadFromText' >> beam.io.ReadFromTFRecord(self._input_path)
                   | 'MakeRecordFn' >> beam.ParDo(_MakeRecordFn())
                   | 'Deduplicate' >> beam.transforms.util.Distinct())
        (records
         | 'WriteToTFRecords' >> beam.io.WriteToTFRecord(
             self._output_path, file_name_suffix='.tfrecord'))
        return records


class GenerateImageDataset(beam.PTransform):
    def __init__(
        self,
        input_path,
        output_path,
        splits=None,
        **kwargs,
    ):
        super().__init__()
        self._input_path = input_path
        self._output_path = output_path
        self._splits = splits or ['train', 'val', 'test']

    def expand(self, pcoll):
        outputs = []
        for split in self._splits:
            records = (
                pcoll
                | 'GenerateDataset:%s' % split >> GenerateSingleImageDataset(
                    os.path.join(self._input_path, split + '*'),
                    os.path.join(self._output_path, split),
                ))
            outputs.append(
                records
                | 'WithKey:%s' % split >> beam.transforms.util.WithKeys(split))

        logger.info(outputs)
        # Write counts.
        (outputs | beam.Flatten()
         | WriteCountToFile(os.path.join(self._output_path, 'count')))
        return outputs
