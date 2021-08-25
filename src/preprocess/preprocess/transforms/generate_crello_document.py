import logging
import os

import apache_beam as beam
import tensorflow as tf

from preprocess.helpers.feature import make_feature
from preprocess.transforms.count import WriteCountToFile

logger = logging.getLogger(__name__)


class _MakeRecordFn(beam.DoFn):
    """Extract image record from each element."""
    def __init__(self, encoder_path):
        self._encoder_path = encoder_path
        self._encoder = None

    def setup(self):
        if self._encoder is None:
            logger.info('Loading %s' % self._encoder_path)
            self._encoder = tf.keras.models.load_model(self._encoder_path)

    def process(self, serialized):
        example = tf.train.SequenceExample()
        example.ParseFromString(serialized)

        # Compute embedding.
        src_list = example.feature_lists.feature_list.pop('image_bytes')
        dst_list = example.feature_lists.feature_list.get_or_create(
            'image_embedding')
        for feature in src_list.feature:
            image_bytes = feature.bytes_list.value[0]
            image_embedding = self._compute_embedding(image_bytes)
            dst_list.feature.append(make_feature(image_embedding, dtype=float))

        yield example.SerializeToString()

    def _compute_embedding(self, image_bytes):
        x = tf.io.decode_png(image_bytes, channels=4)
        x = tf.expand_dims(x, axis=0)
        x = self._encoder(x)
        return x[0].numpy().tolist()


class GenerateSingleDocumentDataset(beam.PTransform):
    def __init__(
        self,
        input_path,
        output_path,
        encoder_path,
        **kwargs,
    ):
        super().__init__()
        self._input_path = input_path
        self._output_path = output_path
        self._encoder_path = encoder_path

    def expand(self, pcoll):
        records = (
            pcoll
            | beam.io.ReadFromTFRecord(self._input_path)
            | beam.transforms.Reshuffle()
            | 'MakeRecord' >> beam.ParDo(_MakeRecordFn(self._encoder_path)))
        (records
         | beam.io.WriteToTFRecord(
             self._output_path,
             file_name_suffix='.tfrecord',
         ))
        return records


class GenerateDocumentDataset(beam.PTransform):
    """Generate Crello Document dataset."""
    def __init__(
        self,
        input_path,
        output_path,
        encoder_path,
        splits=None,
        **kwargs,
    ):
        super().__init__()
        self._input_path = input_path
        self._output_path = output_path
        self._encoder_path = encoder_path
        self._splits = splits or ['train', 'val', 'test']

    def expand(self, pcoll):
        outputs = []
        for split in self._splits:
            records = (
                pcoll
                |
                'GenerateDataset:%s' % split >> GenerateSingleDocumentDataset(
                    os.path.join(self._input_path, split + '*'),
                    os.path.join(self._output_path, split),
                    self._encoder_path,
                ))
            outputs.append(
                records
                | 'WithKey:%s' % split >> beam.transforms.util.WithKeys(split))

        logger.info(outputs)
        # Write counts.
        (outputs | beam.Flatten()
         | WriteCountToFile(os.path.join(self._output_path, 'count')))
        # Copy vocabulary file.
        (pcoll
         | 'ReadVocabularyFromFile' >> beam.io.ReadFromText(
             os.path.join(self._input_path, 'vocabulary.json'))
         | 'WriteVocabularyToFile' >> beam.io.WriteToText(
             os.path.join(self._output_path, 'vocabulary'),
             file_name_suffix='.json',
             num_shards=1,
             shard_name_template='',
         ))
        return outputs
