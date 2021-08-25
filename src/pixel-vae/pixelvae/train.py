import logging
import os

import tensorflow as tf

from pixelvae.data import DatasetBuilder
from pixelvae.model import PixelVAE

logger = logging.getLogger(__name__)


def train(args):
    """Train, evaluate, and export the model."""
    strategy = tf.distribute.MirroredStrategy()
    num_workers = max(
        1, sum(['GPU' in x for x in strategy.extended.worker_devices]))
    logger.info('Found %g devices' % num_workers)

    dataset = DatasetBuilder(
        args.data_dir,
        batch_size=args.batch_size * num_workers,
    )

    train_dataset = dataset.build(
        'train',
        shuffle=True,
        repeat=True,
        cache=True,
    )
    val_dataset = dataset.build('val', repeat=True, cache=True)

    with strategy.scope():
        model = PixelVAE(latent_dim=args.latent_dim, kl=args.kl, l2=args.l2)
        if args.weights:
            logger.info('Loading %s' % args.weights)
            model.load_weights(args.weights)
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate, ), )

    checkpoint_path = os.path.join(args.job_dir, 'checkpoints', 'weights.ckpt')
    model.fit(
        train_dataset,
        steps_per_epoch=10 if args.debug else dataset.steps_per_epoch('train'),
        epochs=args.num_epochs,
        validation_data=val_dataset,
        validation_steps=10 if args.debug else dataset.steps_per_epoch('val'),
        validation_freq=min(args.validation_freq, args.num_epochs),
        callbacks=get_callbacks(args, checkpoint_path),
        verbose=args.verbose,
    )

    if tf.io.gfile.exists(checkpoint_path + '.index'):
        model.load_weights(checkpoint_path)
        logger.info('Loading %s' % checkpoint_path)
    else:
        logger.warning('No checkpoint found at %s' % checkpoint_path)

    test_dataset = dataset.build('test')
    if args.debug:
        test_dataset = test_dataset.take(10)
    model.evaluate(test_dataset, verbose=args.verbose)

    export_model(model, args)


def get_callbacks(args, checkpoint_path):
    """Callbacks for training."""
    log_dir = os.path.join(args.job_dir, 'logs')
    if tf.io.gfile.exists(log_dir):
        logger.warning('Overwriting log dir: %s' % log_dir)
        tf.io.gfile.rmtree(log_dir)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=False,
        profile_batch=2 if args.enable_profile else 0,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        monitor='val_reconstruction',
        mode='min',
        save_best_only=True,
        verbose=1,
    )
    return [tensorboard, checkpoint]


def export_model(model, args):
    """Export the encoder."""
    encoder = tf.keras.Model(model.input, model.encoder(model.input)[0])
    export_path = args.output_path or os.path.join(args.job_dir, 'encoder')
    logger.info('Saving encoder: %s' % export_path)
    encoder.save(export_path)
