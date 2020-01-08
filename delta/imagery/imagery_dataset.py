"""
Tools for loading input images into the TensorFlow Dataset class.
"""
import functools
import math
import random

import tensorflow as tf

from delta.config import config
from delta.imagery import rectangle
from delta.imagery.sources import loader, tfrecord

class ImageryDataset:
    """Create dataset with all files as described in the provided config file.
    """

    def __init__(self, images, labels, chunk_size, chunk_stride=1):
        """
        Initialize the dataset based on the specified image and label ImageSets
        """

        # Record some of the config values
        self._chunk_size = chunk_size
        self._chunk_stride = chunk_stride

        self._use_tfrecord = images.type() == 'tfrecord'

        if self._use_tfrecord and labels.type() != 'tfrecord':
            raise NotImplementedError('tfrecord images only supported with tfrecord labels.')

        assert len(images) == len(labels)
        self._images = images
        self._labels = labels

        # Load the first image to get the number of bands for the input files.
        self._num_bands = loader.load_image(images, 0).num_bands()

    def _load_tensor_imagery(self, is_labels, image_index, bbox):
        """Loads a single image as a tensor."""
        assert not self._use_tfrecord
        image = loader.load_image(self._labels if is_labels else self._images, image_index.numpy())
        w = int(bbox[2])
        h = int(bbox[3])
        rect = rectangle.Rectangle(int(bbox[0]), int(bbox[1]), w, h)
        r = image.read(rect)
        return r

    def _tile_images(self):
        max_block_bytes = config.block_size_mb() * 1024 * 1024
        def tile_generator():
            tgs = []
            for i in range(len(self._images)):
                img = loader.load_image(self._images, i)
                # TODO: account for other data types properly
                # w * h * bands * 4 * chunk * chunk = max_block_bytes
                tile_width = int(math.sqrt(max_block_bytes / img.num_bands() / 4 /
                                           (self._chunk_size ** 2) / config.tile_ratio()))
                tile_height = int(config.tile_ratio() * tile_width)
                if tile_width < self._chunk_size * 2:
                    raise ValueError('max_block_size is too low.')
                if tile_height < self._chunk_size * 2:
                    raise ValueError('max_block_size is too low.')
                tiles = img.tiles(tile_width, tile_height, overlap=self._chunk_size)
                random.Random(0).shuffle(tiles) # gives consistent random ordering so labels will match
                tgs.append((i, tiles))
            while tgs:
                cur = tgs[:config.interleave_images()]
                tgs = tgs[config.interleave_images():]
                done = False
                while not done:
                    done = True
                    for it in cur:
                        if not it[1]:
                            continue
                        t = it[1].pop(0)
                        if t:
                            done = False
                            yield (it[0], t.min_x, t.min_y, t.max_x, t.max_y)
                    if done:
                        break
        return tf.data.Dataset.from_generator(tile_generator,
                                              (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))

    def _load_images(self, is_labels, num_bands, data_type):
        """
        Loads a list of images as tensors.
        If label_list is specified, load labels instead. The corresponding image files are still required however.
        """
        if self._use_tfrecord:
            ret = tfrecord.create_dataset(list(self._labels if is_labels else self._images),
                                          num_bands, data_type, config.threads())
            # ignore images that are too small to use
            ret = ret.filter(lambda x: tf.shape(x)[0] >= self._chunk_size and tf.shape(x)[1] >= self._chunk_size)
        else:
            ds_input = self._tile_images()
            def load_tile(image_index, x1, y1, x2, y2):
                img = tf.py_function(functools.partial(self._load_tensor_imagery,
                                                       is_labels),
                                     [image_index, [x1, y1, x2, y2]], data_type)
                return img
            ret = ds_input.map(load_tile, num_parallel_calls=config.threads())

        return ret.prefetch(tf.data.experimental.AUTOTUNE)

    def _chunk_image(self, image):
        """Split up a tensor image into tensor chunks"""

        ksizes  = [1, self._chunk_size, self._chunk_size, 1] # Size of the chunks
        strides = [1, self._chunk_stride, self._chunk_stride, 1] # SPacing between chunk starts
        rates   = [1, 1, 1, 1]
        result  = tf.image.extract_patches(tf.expand_dims(image, 0), ksizes, strides, rates,
                                           padding='VALID')
        # Output is [1, M, N, chunk*chunk*bands]
        result = tf.reshape(result, [-1, self._chunk_size, self._chunk_size, self._num_bands])

        return result

    def _reshape_labels(self, labels):
        """Reshape the labels to account for the chunking process."""
        w = self._chunk_size // 2
        labels = tf.image.crop_to_bounding_box(labels, w, w, tf.shape(labels)[0] - 2 * w, tf.shape(labels)[1] - 2 * w)
        labels = labels[::self._chunk_stride, ::self._chunk_stride]
        return tf.reshape(labels, [-1, 1])

    def data(self):
        # TODO: other types?
        ret = self._load_images(False, self._num_bands, tf.float32)
        ret = ret.map(self._chunk_image, num_parallel_calls=config.threads())
        return ret.unbatch()

    def labels(self):
        label_set = self._load_images(True, 1, tf.uint8)
        label_set = label_set.map(self._reshape_labels)

        return label_set.unbatch()

    def dataset(self):
        """Return the underlying TensorFlow dataset object that this class creates"""

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((self.data(), self.labels()))

        return ds

    def num_bands(self):
        """Return the number of bands in each image of the data set"""
        return self._num_bands

    def chunk_size(self):
        return self._chunk_size

    def image_set(self):
        return self._images
    def label_set(self):
        return self._labels

class AutoencoderDataset(ImageryDataset):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""

    def __init__(self, images, chunk_size, chunk_stride=1):
        '''
        AutoencoderDataset constructor.
        @param images The images which are being used to create the autoencoder.
        @param chunk_size The size of the square chucks the autoencoder is trained on.
        @param chunk_stride The stride which are used for extracting chunks.
        '''
        super(AutoencoderDataset, self).__init__(images, None, chunk_size, chunk_stride=chunk_stride)

    def labels(self):
        return self.data()
