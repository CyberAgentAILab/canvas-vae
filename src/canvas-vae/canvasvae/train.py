import logging
import json
import os

import tensorflow as tf

from canvasvae.data import DataSpec
from canvasvae.models.vae import VAE
from canvasvae.helpers.random_eval import (
    RandomEvaluationCallback,
    evaluate_random_generation,
)

logger = logging.getLogger(__name__)


def train_and_evaluate(args):
    """Train and evaluate CanvasVAE."""
    logger.info(f'tensorflow version {tf.__version__}')

    dataspec = DataSpec(
        args.dataset_name,
        args.data_dir,
        batch_size=args.batch_size,
    )
    model = train(args, dataspec)
    evaluate(args, dataspec, model)


def get_callbacks(args, dataspec, checkpoint_path):
    log_dir = os.path.join(args.job_dir, 'logs')
    if tf.io.gfile.exists(log_dir):
        logger.warning('Overwriting log dir: %s' % log_dir)
        tf.io.gfile.rmtree(log_dir)

    logger.info(f'checkpoint_path={checkpoint_path}')
    logger.info(f'log_dir={log_dir}')

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=False,
        profile_batch=2 if args.enable_profile else 0,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        monitor='val_total_score',
        mode='max',
        save_best_only=True,
        verbose=1,
    )
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()
    random_test = RandomEvaluationCallback(
        dataspec,
        split='val',
    )
    return [random_test, tensorboard, checkpoint, terminate_on_nan]


def train(args, dataspec, return_best_model=False):
    logger.info(f'tensorflow version {tf.__version__}')
    checkpoint_path = os.path.join(args.job_dir, 'checkpoints', 'best.ckpt')

    train_dataset = dataspec.make_dataset(
        'train',
        shuffle=True,
        repeat=True,
        cache=True,
    )
    val_dataset = dataspec.make_dataset('val', repeat=True, cache=True)

    model = VAE(
        dataspec.make_input_columns(),
        latent_dim=args.latent_dim,
        decoder_type=args.decoder_type,
        num_blocks=args.num_blocks,
        block_type=args.block_type,
        kl=args.kl,
        l2=args.l2,
    )
    if args.weights:
        logger.info('Loading %s' % args.weights)
        model.load_weights(args.weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=1.0,
    ), )

    model.fit(
        train_dataset,
        steps_per_epoch=dataspec.steps_per_epoch('train'),
        epochs=args.num_epochs,
        validation_data=val_dataset,
        validation_steps=dataspec.steps_per_epoch('val'),
        validation_freq=min(args.validation_freq, args.num_epochs),
        callbacks=get_callbacks(args, dataspec, checkpoint_path),
        verbose=args.verbose,
    )
    # Save the last model.
    model_path = os.path.join(args.job_dir, 'checkpoints', 'final.ckpt')
    logger.info('Saving %s' % model_path)
    model.save_weights(model_path)

    # Load best checkpoint.
    if return_best_model:
        logger.info('Loading %s' % checkpoint_path)
        model.load_weights(checkpoint_path)
    return model


def evaluate(args, dataspec, model, split='test'):
    evaluation = model.evaluate(
        dataspec.make_dataset(split),
        verbose=args.verbose,
        return_dict=True,
    )
    evaluation = {
        'reconst_' + k.replace('_score', ''): v
        for k, v in evaluation.items() if not k.endswith('loss')
    }
    random_scores = evaluate_random_generation(model, dataspec, split=split)

    # Save metadata and evaluation results.
    results = vars(args)
    results.update(evaluation)
    results.update(random_scores)
    results_path = os.path.join(args.job_dir, split + '_results.json')
    logger.info('Writing results to %s' % results_path)
    with tf.io.gfile.GFile(results_path, 'w') as f:
        json.dump(results, f)

    return results
