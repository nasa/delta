# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tools for loading input images into the TensorFlow Dataset class.
"""
import functools
import math
import random
import sys
import os
import portalocker
import numpy as np
import tensorflow as tf

from delta.config import config
from delta.imagery import rectangle
from delta.imagery.sources import loader

class ImageryDataset:
    """Create dataset with all files as described in the provided config file.
    """

    def __init__(self, images, labels, chunk_size, output_size, chunk_stride=1,
                 resume_mode=False, log_folder=None):
        """
        Initialize the dataset based on the specified image and label ImageSets
        """

        self._resume_mode = resume_mode
        self._log_folder  = log_folder
        if self._log_folder and not os.path.exists(self._log_folder):
            os.mkdir(self._log_folder)

        # Record some of the config values
        assert (chunk_size % 2) == (output_size % 2), 'Chunk size and output size must both be either even or odd.'
        self._chunk_size   = chunk_size
        self._output_size  = output_size
        self._output_dims  = 1
        self._chunk_stride = chunk_stride
        self._data_type    = tf.float32
        self._label_type   = tf.uint8

        if labels:
            assert len(images) == len(labels)
        self._images = images
        self._labels = labels

        # Load the first image to get the number of bands for the input files.
        self._num_bands = loader.load_image(images, 0).num_bands()

    def _get_image_read_log_path(self, image_path):
        """Return the path to the read log for an input image"""
        if not self._log_folder:
            return None
        image_name = os.path.basename(image_path)
        file_name  = os.path.splitext(image_name)[0] + '_read.log'
        log_path   = os.path.join(self._log_folder, file_name)
        return log_path

    def _get_image_read_count(self, image_path):
        """Return the number of ROIs we have read from an image"""
        log_path = self._get_image_read_log_path(image_path)
        if (not log_path) or not os.path.exists(log_path):
            return 0
        counter = 0
        with portalocker.Lock(log_path, 'r', timeout=300) as f:
            for line in f: #pylint: disable=W0612
                counter += 1
        return counter

    def _load_tensor_imagery(self, is_labels, image_index, bbox):
        """Loads a single image as a tensor."""
        data = self._labels if is_labels else self._images

        if not is_labels: # Record each time we write a tile
            file_path = data[image_index.numpy()]
            log_path  = self._get_image_read_log_path(file_path)
            if log_path:
                with portalocker.Lock(log_path, 'a', timeout=300) as f:
                    f.write(str(bbox) + '\n')
                    # TODO: What to write and when to clear it?

        try:
            image = loader.load_image(data, image_index.numpy())
            w = int(bbox[2])
            h = int(bbox[3])
            rect = rectangle.Rectangle(int(bbox[0]), int(bbox[1]), w, h)
            r = image.read(rect)
        except Exception as e: #pylint: disable=W0703
            print('Caught exception loading tile from image: ' + data[image_index.numpy()] + ' -> ' + str(e)
                  + '\nSkipping tile: ' + str(bbox))
            if config.general.stop_on_input_error():
                print('Aborting processing, set --bypass-input-errors to bypass this error.')
                raise
            # Else just skip this tile
            r = np.zeros(shape=(0,0,0), dtype=np.float32)
        return r

    def _tile_images(self):
        max_block_bytes = config.io.block_size_mb() * 1024 * 1024
        def tile_generator():
            tgs = []
            for i in range(len(self._images)):

                if self._resume_mode:
                    # TODO: Improve feature to work with multiple epochs
                    # Skip images which we have already read some number of tiles from
                    if self._get_image_read_count(self._images[i]) > config.io.resume_cutoff():
                        continue

                try:
                    img = loader.load_image(self._images, i)

                    if self._labels: # If we have labels make sure they are the same size as the input images
                        label = loader.load_image(self._labels, i)
                        if label.size() != img.size():
                            raise Exception('Label file ' + self._labels[i] + ' with size ' + str(label.size())
                                            + ' does not match input image size of ' + str(img.size()))
                    # w * h * bands * 4 * chunk * chunk = max_block_bytes
                    tile_width = int(math.sqrt(max_block_bytes / img.num_bands() / self._data_type.size /
                                               config.io.tile_ratio()))
                    tile_height = int(config.io.tile_ratio() * tile_width)
                    min_block_size = self._chunk_size ** 2 * config.io.tile_ratio() * img.num_bands() * 4
                    if max_block_bytes < min_block_size:
                        print('Warning: max_block_bytes=%g MB, but %g MB is recommended (minimum: %g MB)'
                              % (max_block_bytes / 1024 / 1024,
                                 min_block_size * 2 / 1024 / 1024, min_block_size / 1024/ 1024),
                              file=sys.stderr)
                    if tile_width < self._chunk_size or tile_height < self._chunk_size:
                        raise ValueError('max_block_bytes is too low.')
                    tiles = img.tiles(tile_width, tile_height, min_width=self._chunk_size, min_height=self._chunk_size,
                                      overlap=self._chunk_size - 1)
                except Exception as e: #pylint: disable=W0703
                    print('Caught exception tiling image: ' + self._images[i] + ' -> ' + str(e)
                          + '\nWill not load any tiles from this image')
                    if config.general.stop_on_input_error():
                        print('Aborting processing, set --bypass-input-errors to bypass this error.')
                        raise
                    tiles = [] # Else move past this image without loading any tiles

                random.Random(0).shuffle(tiles) # gives consistent random ordering so labels will match
                tgs.append((i, tiles))
            if not tgs:
                return
            while tgs:
                cur = tgs[:config.io.interleave_images()]
                tgs = tgs[config.io.interleave_images():]
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

    def _load_images(self, is_labels, data_type):
        """
        Loads a list of images as tensors.
        If label_list is specified, load labels instead. The corresponding image files are still required however.
        """
        ds_input = self._tile_images()
        def load_tile(image_index, x1, y1, x2, y2):
            img = tf.py_function(functools.partial(self._load_tensor_imagery,
                                                   is_labels),
                                 [image_index, [x1, y1, x2, y2]], data_type)
            return img
        ret = ds_input.map(load_tile, num_parallel_calls=tf.data.experimental.AUTOTUNE)#config.io.threads())

        # Don't let the entire session be taken down by one bad dataset input.
        # - Would be better to handle this somehow but it is not clear if TF supports that.
#        ret = ret.apply(tf.data.experimental.ignore_errors())

        return ret

    def _chunk_image(self, image):
        """Split up a tensor image into tensor chunks"""
        ksizes  = [1, self._chunk_size, self._chunk_size, 1] # Size of the chunks
        strides = [1, self._chunk_stride, self._chunk_stride, 1] # Spacing between chunk starts
        rates   = [1, 1, 1, 1]
        result  = tf.image.extract_patches(tf.expand_dims(image, 0), ksizes, strides, rates,
                                           padding='VALID')
        # Output is [1, M, N, chunk*chunk*bands]
        result = tf.reshape(result, [-1, self._chunk_size, self._chunk_size, self._num_bands])

        return result

    def _reshape_labels(self, labels):
        """Reshape the labels to account for the chunking process."""
        w = (self._chunk_size - self._output_size) // 2
        labels = tf.image.crop_to_bounding_box(labels, w, w, tf.shape(labels)[0] - 2 * w,
                                                             tf.shape(labels)[1] - 2 * w) #pylint: disable=C0330

        ksizes  = [1, self._output_size, self._output_size, 1]
        strides = [1, self._chunk_stride, self._chunk_stride, 1]
        rates   = [1, 1, 1, 1]
        labels = tf.image.extract_patches(tf.expand_dims(labels, 0), ksizes, strides, rates,
                                          padding='VALID')
        return tf.reshape(labels, [-1, self._output_size, self._output_size])

    def data(self):
        """
        Unbatched dataset of image chunks.
        """
        ret = self._load_images(False, self._data_type)
        ret = ret.map(self._chunk_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ret.unbatch()

    def labels(self):
        """
        Unbatched dataset of labels.
        """
        label_set = self._load_images(True, self._label_type)
        label_set = label_set.map(self._reshape_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE) #pylint: disable=C0301
        return label_set.unbatch()

    def dataset(self, class_weights=None):
        """
        Return the underlying TensorFlow dataset object that this class creates.

        class_weights: a list of weights to apply to the samples in each class, if specified.
        """

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((self.data(), self.labels()))
        # ignore labels with no data
        if self._labels.nodata_value():
            ds = ds.filter(lambda x, y: tf.math.not_equal(y, self._labels.nodata_value()))
        if class_weights is not None:
            lookup = tf.constant(class_weights)
            ds = ds.map(lambda x, y: (x, y, tf.gather(lookup, tf.cast(y, tf.int32))))
        return ds

    def num_bands(self):
        """
        Return the number of bands in each image of the data set.
        """
        return self._num_bands

    def chunk_size(self):
        """
        Size of chunks used for inputs.
        """
    def output_shape(self):
        """
        Output size of blocks of labels.
        """
        return (self._output_size, self._output_size, self._output_dims)

    def image_set(self):
        """
        Returns set of images.
        """
        return self._images
    def label_set(self):
        """
        Returns set of label images.
        """
        return self._labels

class AutoencoderDataset(ImageryDataset):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""

    def __init__(self, images, chunk_size, chunk_stride=1, resume_mode=False, log_folder=None):
        """
        The images are used as labels as well.
        """
        super(AutoencoderDataset, self).__init__(images, None, chunk_size, chunk_size, chunk_stride=chunk_stride,
                                                 resume_mode=resume_mode, log_folder=log_folder)
        self._labels = self._images
        self._output_dims = self.num_bands()

    def labels(self):
        return self.data()
