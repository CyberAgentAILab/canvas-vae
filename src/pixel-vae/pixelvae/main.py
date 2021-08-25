import argparse
import logging
import os

from pixelvae.train import train

logger = logging.getLogger(__name__)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Pixel VAE trainer.")
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
        "--output-path",
        default=None,
        type=str,
        help="Saved model path, default to <job-dir>/encoder",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Path to the initial model weight.",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--num-epochs",
        default=250,
        type=int,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
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
        "--kl",
        default=1.0e+2,
        type=float,
        help="Scalar coefficient for KLD loss.",
    )
    parser.add_argument(
        "--l2",
        default=1.0e-6,
        type=float,
        help="L2 regularization factor.",
    )
    parser.add_argument(
        "--enable-profile",
        dest="enable_profile",
        action="store_true",
        help=
        "Enable profiling for tensorboard. (See tensorflow/tensorboard#3149)",
    )
    parser.add_argument(
        "--validation-freq",
        default=1,
        type=int,
        help="Validation frequency in terms of epochs.",
    )
    parser.add_argument("--log-level", default="INFO", type=str)
    parser.add_argument("--verbose", default=2, type=int)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(args)


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.info(args)
    train(args)


if __name__ == "__main__":
    main()
