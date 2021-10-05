from base64 import b64encode
import dbm
import hashlib
import logging
import os
import sys
import tempfile

import numpy as np
import faiss
import tensorflow as tf

logger = logging.getLogger(__name__)


class ImageRetriever(object):
    """
    Image retrieval manager for PixelVAE embeddings.

    Example::

        encoder = tf.keras.models.load_model('../data/pixelvae/encoder')
        image_db = ImageRetriever(
            dataset_path='../data/crello-image/train*',
            cache_dir='../tmp/retriever/train_images',
        )
        image_db.build(encoder)

        query = np.zeros((1, image_db.dim), np.float32)
        image_url = image_db.search(query)
    """
    def __init__(self, dataset_path, cache_dir=None):
        self._dataset_path = dataset_path

        if cache_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self._cache_dir = self._temp_dir.name
        else:
            self._cache_dir = cache_dir
            if not os.path.exists(self._cache_dir):
                logger.info(f'Creating cache dir at {self._cache_dir}')
                os.makedirs(self._cache_dir)

        self._image_cache_path = os.path.join(self._cache_dir, 'image')
        self._feature_cache_path = os.path.join(self._cache_dir, 'feature')
        self._key_cache_path = os.path.join(self._cache_dir, 'keys')

        self._image_keys = None
        self._image_cache = None
        self._feature_index = None

    def build(self, encoder, batch_size=16):
        """Compute and build feature index."""

        image_cache_file = self._image_cache_path
        if sys.platform == 'linux':
            image_cache_file += '.dat'
            
        if all(
                os.path.exists(p) for p in (
                    image_cache_file,
                    self._feature_cache_path,
                    self._key_cache_path,
                )):
            logger.info(f'Loading cached retriever from {self._cache_dir}')
            with open(self._key_cache_path, 'r') as f:
                self._image_keys = [x.strip() for x in f.readlines()]
            self._feature_index = faiss.read_index(self._feature_cache_path)

        else:
            logger.info(f'No cache found at {self._cache_dir}')

            logger.info('Computing image features...')
            data_path = os.path.join(self._dataset_path)
            dataset = tf.data.Dataset.list_files(data_path)
            dataset = tf.data.TFRecordDataset(dataset)
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(
                lambda x: _compute_feature(encoder, x),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            features = []
            self._image_keys = []
            with dbm.open(self._image_cache_path, 'n') as db:
                for batch in dataset.as_numpy_iterator():
                    features.append(batch[0])

                    for x in batch[1]:
                        image_key = hashlib.md5(x).hexdigest()
                        db[image_key] = x
                        self._image_keys.append(image_key)

                features = np.concatenate(features, axis=0)

            logger.info('Building feature index')
            self._feature_index = faiss.IndexFlatL2(features.shape[1])
            self._feature_index.add(features)

            logger.info('Caching data')
            with open(self._key_cache_path, 'w') as f:
                f.writelines(x + '\n' for x in self._image_keys)
            faiss.write_index(self._feature_index, self._feature_cache_path)

        logger.info(f'Indexd {len(self._image_keys)} features of '
                    f'dim={self._feature_index.d}')

        self._image_cache = dbm.open(self._image_cache_path, 'r')

    @property
    def dim(self):
        return self._feature_index.d if self._feature_index else None

    def len(self):
        return len(self._image_keys) if self._image_keys else 0

    def search(self, query, k=1):
        assert self._feature_index is not None, 'Index is not built yet'

        if isinstance(query, list):
            query = np.array([query], dtype=np.float32)

        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
        assert query.dtype == np.float32
        assert query.shape[1] == self._feature_index.d

        def _make_data_url(i):
            key = self._image_keys[i]
            image_bytes = self._image_cache[key]
            encoded = b64encode(image_bytes).decode('ascii')
            return f'data:image/png;base64,{encoded}'

        _, index = self._feature_index.search(query, k)
        urls = [_make_data_url(i) for i in index[0].tolist()]
        if k == 1:
            return urls[0]
        return urls


@tf.function
def _compute_feature(encoder, serialized):
    image_bytes = tf.io.parse_example(
        serialized, {
            'image': tf.io.FixedLenFeature((), dtype=tf.string),
        })['image']
    images = tf.map_fn(
        lambda x: tf.io.decode_png(x, channels=4),
        image_bytes,
        fn_output_signature=tf.uint8,
    )
    features = encoder(images)
    return features, image_bytes
