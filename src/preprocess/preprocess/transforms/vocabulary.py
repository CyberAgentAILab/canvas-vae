from collections import Counter
import json

import apache_beam as beam
import tensorflow as tf
import numpy as np


class ExtractValuesFn(beam.DoFn):
    """Extract categorical values by key."""
    def __init__(self, context_spec, sequence_spec):
        self._context_spec = context_spec
        self._sequence_spec = sequence_spec

    def process(self, split_serialized):
        _, serialized = split_serialized
        context, sequence = tf.io.parse_single_sequence_example(
            serialized,
            self._context_spec,
            self._sequence_spec,
        )
        for key, value in context.items():
            if isinstance(value, tf.sparse.SparseTensor):
                value = tf.sparse.to_dense(value)
            value = value.numpy()
            if isinstance(value, np.ndarray):
                for v in value.tolist():
                    yield key, v
            else:
                yield key, value
        for key, value in sequence.items():
            if isinstance(value, tf.sparse.SparseTensor):
                value = tf.sparse.to_dense(value)
            for v in value.numpy().tolist():
                yield key, v


class CombineVocabularyFn(beam.CombineFn):
    """Combine distinct values per key, then encode in JSON."""
    def create_accumulator(self):
        return {}

    def add_input(self, accumulator, input):
        k, v = input
        if isinstance(v, bytes):
            v = v.decode('utf-8')
        if isinstance(v, np.int64):
            v = int(v)
        if k in accumulator:
            accumulator[k].update([v])
        else:
            accumulator[k] = Counter([v])
        return accumulator

    def merge_accumulators(self, accumulators):
        itr = iter(accumulators)
        accum = next(itr)
        for s in itr:
            for k in s:
                if k in accum:
                    accum[k].update(s[k].elements())
                else:
                    accum[k] = s[k]
        return accum

    def extract_output(self, accumulator):
        for k in accumulator:
            accumulator[k] = dict(
                sorted(
                    accumulator[k].items(),
                    key=lambda t: -t[1],
                ))
        return json.dumps(accumulator)


class WriteVocabularyToFile(beam.PTransform):
    def __init__(self, file_path_prefix, context_spec, sequence_spec,
                 **kwargs):
        super().__init__(**kwargs)
        self._file_path_prefix = file_path_prefix
        self._context_spec = context_spec
        self._sequence_spec = sequence_spec

    def expand(self, pcoll):
        return (
            pcoll
            | 'ExtractVocabulary' >> beam.ParDo(
                ExtractValuesFn(self._context_spec, self._sequence_spec))
            |
            'CombineVocabulary' >> beam.CombineGlobally(CombineVocabularyFn())
            | 'WriteVocabularyToJSON' >> beam.io.WriteToText(
                self._file_path_prefix,
                file_name_suffix='.json',
                num_shards=1,
                shard_name_template='',
            ))
