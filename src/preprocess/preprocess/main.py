import logging

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

logger = logging.getLogger(__name__)


class ApplicationOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument(
            'pipeline',
            choices=[
                'crello',
                'crello-document',
                'crello-image',
                'rico',
                'magazine',
            ],
            help='Kind of pipeline to invoke.',
        )
        parser.add_argument(
            '--input-path',
            required=True,
            type=str,
            help='Input file path.',
        )
        parser.add_argument(
            '--output-path',
            required=True,
            type=str,
            help='Output prefix.',
        )
        parser.add_argument(
            '--assets-dir',
            default=None,
            type=str,
            help='Input asset, required for crello-document and crello-image.')
        parser.add_argument(
            '--encoder-path',
            default=None,
            type=str,
            help='Encoder location, required for crello-document dataset.')
        parser.add_argument(
            '--max-seq-length',
            default=50,
            type=int,
            help='Maximum sequence length.',
        )
        parser.add_argument(
            '--num-shards',
            default=8,
            type=int,
            help='Number of shards.',
        )
        parser.add_argument(
            '--loglevel',
            default='info',
            type=str,
            help='Logging level.',
        )
        parser.add_argument('--debug', action='store_true', help='Debug mode.')


def main():
    import preprocess
    from preprocess.transforms import create_transform

    options = ApplicationOptions()
    logging.basicConfig(level=getattr(logging, options.loglevel.upper()))
    if options.debug:
        logging.getLogger(preprocess.__name__).setLevel(level=logging.DEBUG)

    if options.pipeline == 'crello':
        assert options.assets_dir
    elif options.pipeline == 'crello-document':
        assert options.encoder_path

    logger.info(options.display_data())

    transform = create_transform(options.pipeline)(
        input_path=options.input_path,
        output_path=options.output_path,
        assets_dir=options.assets_dir,
        num_shards=options.num_shards,
        encoder_path=options.encoder_path,
        max_seq_length=options.max_seq_length,
    )

    with beam.Pipeline(options=options) as p:
        (p | "GenerateDataset" >> transform)
