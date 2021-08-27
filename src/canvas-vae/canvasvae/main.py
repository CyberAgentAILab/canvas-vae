import argparse
import logging
import os

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="CanvasVAE trainer")
    parser.add_argument(
        "--dataset-name",
        required=True,
        choices=[
            'crello-document',
            'rico',
        ],
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="The GCS or local path of the data location.",
    )
    parser.add_argument(
        "--job-dir",
        required=True,
        help="The GCS or local path of logs and saved models.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Path to the initial model weight.",
    )
    parser.add_argument(
        "--batch-size",
        default=1024,
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--num-epochs",
        default=500,
        type=int,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--latent-dim",
        default=256,
        type=int,
        help="Latent dimension.",
    )
    parser.add_argument(
        "--num-blocks",
        default=1,
        type=int,
        help="Number of stacked blocks in sequence encoder.",
    )
    parser.add_argument(
        "--decoder-type",
        default="oneshot",
        choices=["oneshot", "autoregressive"],
        help="Decoder type.",
    )
    parser.add_argument(
        "--block-type",
        default='deepsvg',
        choices=['lstm', 'transformer', 'deepsvg'],
        help="Stacked block type. deepsvg = deepsvg-style transformer block.",
    )
    parser.add_argument(
        "--kl",
        default=16,
        type=float,
        help="Scalar coefficient for KLD regularization.",
    )
    parser.add_argument(
        "--l2",
        default=1e-6,
        type=float,
        help="Scalar coefficient for L2 regularization.",
    )
    parser.add_argument(
        "--enable-profile",
        dest="enable_profile",
        action="store_true",
        help="Enable profiling for tensorboard.",
    )
    parser.add_argument(
        "--validation-freq",
        default=20,
        type=int,
        help="Validation frequency in terms of epochs.",
    )
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--verbose", default=2, type=int)
    return parser.parse_args(args)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.info(args)

    # Lazy loading for TF-related import
    from canvasvae.train import train_and_evaluate
    train_and_evaluate(args)
