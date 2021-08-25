import logging
import json

import apache_beam as beam

logger = logging.getLogger(__name__)


class WriteCountToFile(beam.PTransform):
    def __init__(self, file_path_prefix, **kwargs):
        super().__init__(**kwargs)
        self._file_path_prefix = file_path_prefix

    def expand(self, pcoll):
        return (pcoll
                | 'CountPerKey' >> beam.transforms.combiners.Count.PerKey()
                | 'CombineToDict' >> beam.transforms.combiners.ToDict()
                | 'DumpToJson' >> beam.Map(json.dumps)
                | 'WriteCountToText' >> beam.io.WriteToText(
                    self._file_path_prefix,
                    file_name_suffix='.json',
                    num_shards=1,
                    shard_name_template='',
                ))
