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
import random
import os
import portalocker
import numpy as np
import tensorflow as tf

from delta.config import config
from delta.imagery import rectangle

class ImageryDataset:
    """Create dataset with all files as described in the provided config file.
    """

    def __init__(self, images, labels, output_shape, chunk_shape, chunk_stride=(1, 1),
                 tile_shape=(256, 256)):
        """
        Initialize the dataset based on the specified image and label ImageSets
        """

        self._resume_mode = False
        self._log_folder  = None

        # Record some of the config values
        self.set_chunk_output_shapes(chunk_shape, output_shape)
        self._output_dims  = 1
        self._chunk_stride = chunk_stride
        self._data_type    = tf.float32
        self._label_type   = tf.uint8
        self._tile_shape = tile_shape

        if labels:
            assert len(images) == len(labels)
        self._images = images
        self._labels = labels

        # Load the first image to get the number of bands for the input files.
        self._num_bands = images.load(0).num_bands()

    def set_resume_mode(self, resume_mode, log_folder):
        self._resume_mode = resume_mode
        self._log_folder = log_folder
        if self._log_folder and not os.path.exists(self._log_folder):
            os.mkdir(self._log_folder)

    def _get_image_read_log_path(self, image_path):
        """Return the path to the read log for an input image"""
        if not self._log_folder:
            return None
        image_name = os.path.basename(image_path)
        file_name  = os.path.splitext(image_name)[0] + '_read.log'
        log_path   = os.path.join(self._log_folder, file_name)
        return log_path

    def _read_access_count_file(self, path): #pylint: disable=R0201
        """Reads an access count file containing a boolean and a count.
           The boolean is set to true if we need to check the count."""
        try:
            with portalocker.Lock(path, 'r', timeout=300) as f:
                line = f.readline()
                parts = line.split()
                if len(parts) == 1: # Legacy files
                    return (True, int(parts[0]))
                needToCheck = (parts[0] == '1')
                return (needToCheck, int(parts[1]))
        except OSError as e:
            if e.errno == 122: # Disk quota exceeded
                raise
            return (False, 0)
        except Exception: #pylint: disable=W0703
            # If there is a problem reading the count just treat as zero
            return (False, 0)

    def _update_access_count_file(self, path, need_check, count):  #pylint: disable=R0201
        if need_check:
            bool_str = '1 '
        else:
            bool_str = '0 '
        count_str = str(count+1)
        with portalocker.Lock(path, 'w', timeout=300) as f:
            f.write(bool_str + count_str) # No need to check again


    def reset_access_counts(self, set_need_check=False):
        """Go through all the access files and reset one of the values.
           This is needed for resume mode to work.
           Call with default value to reset the counts to zero.  Call with
           "set_need_check" to keep the count and mark that it needs to be checked.
           Call with default after each epoch, call (True) at start of training."""
        if not self._log_folder:
            return
        if config.general.verbose():
            print('Resetting access counts in folder: ' + self._log_folder)
        file_list = os.listdir(self._log_folder)
        for log_name in file_list:
            if '_read.log' in log_name:
                log_path = os.path.join(self._log_folder, log_name)
                if set_need_check:
                    _, count = self._read_access_count_file(log_path)
                    with portalocker.Lock(log_path, 'w', timeout=300) as f:
                        f.write("1 " + str(count)) # Keep the count, set the bool
                else: # Reset the access count
                    with portalocker.Lock(log_path, 'w', timeout=300) as f:
                        f.write("0 0") # No need to read if we reset the count

    def _load_tensor_imagery(self, is_labels, image_index, bbox):
        """Loads a single image as a tensor."""
        data = self._labels if is_labels else self._images

        file_path = self._images[image_index.numpy()]
        log_path  = self._get_image_read_log_path(file_path)
        if log_path:
            (need_to_check, count) = self._read_access_count_file(log_path)

            if self._resume_mode and need_to_check and (count > config.io.resume_cutoff()):
                # Read this file too many times in a previous run, skip the image file
                # and leave the access file alone so we keep skipping it.
                if config.general.verbose():
                    print('Skipping index ' + str(image_index.numpy())
                          +' with count ' + str(count) + ' -> ' + file_path)
                return np.zeros(shape=(0,0,0), dtype=np.float32)

            if not is_labels: # The count file is shared don't write to it as label
                self._update_access_count_file(log_path, need_check=False, count=count)

        try:
            image = data.load(image_index.numpy())
            rect = rectangle.Rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            r = image.read(rect)
        except Exception as e: #pylint: disable=W0703
            print('Caught exception loading tile from image: ' + data[image_index.numpy()] + ' -> ' + str(e)
                  + '\nSkipping tile: ' + str(bbox))
            if config.io.stop_on_input_error():
                print('Aborting processing, set --bypass-input-errors to bypass this error.')
                raise
            # Else just skip this tile
            r = np.zeros(shape=(0,0,0), dtype=np.float32)
        return r

    def _tile_images(self):
        """Return a Dataset generator object which will cycle through all tiles in all input images"""

        # Define a local generator function to be passed into a TF dataset function.
        def tile_generator():

            # Get a list of input image indices in a random order
            num_images = len(self._images)
            indices    = list(range(num_images))
            random.Random(0).shuffle(indices) # Use consistent random ordering

            # Local function to get the tile list for a single image index
            def get_image_tile_list(i):
                try:
                    # If we need to skip this file because of the read count, no need to look up tiles.
                    if self._resume_mode:
                        file_path = self._images[i]
                        log_path  = self._get_image_read_log_path(file_path)
                        if config.general.verbose():
                            print('get_image_tile_list for index ' + str(i) + ' -> ' + file_path)
                        if log_path:
                            (need_to_check, count) = self._read_access_count_file(log_path)
                            if need_to_check and (count > config.io.resume_cutoff()): #pylint: disable=R1705
                                if config.general.verbose():
                                    print('Skipping index ' + str(i) + ' tile gen with count '
                                          + str(count) + ' -> ' + file_path)
                                return (i, [])
                            else:
                                if config.general.verbose():
                                    print('Computing tile list for index ' + str(i) + ' with count '
                                          + str(count) + ' -> ' + file_path)
                        else:
                            if config.general.verbose():
                                print('No read log file for index ' + str(i))

                    img = self._images.load(i)

                    if self._labels: # If we have labels make sure they are the same size as the input images
                        label = self._labels.load(i)
                        if label.size() != img.size():
                            raise Exception('Label file ' + self._labels[i] + ' with size ' + str(label.size())
                                            + ' does not match input image size of ' + str(img.size()))
                    tile_shape = self._tile_shape
                    if self._chunk_shape:
                        assert tile_shape[0] >= self._chunk_shape[0] and \
                               tile_shape[1] >= self._chunk_shape[1], 'Tile too small.'
                        tiles = img.tiles((tile_shape[0], tile_shape[1]), min_shape=self._chunk_shape,
                                          overlap_shape=(self._chunk_shape[0] - 1, self._chunk_shape[1] - 1))
                    else:
                        # TODO: make overlap configurable for FCN
                        tiles = img.tiles((tile_shape[0], tile_shape[1]), partials=False, partials_overlap=True)
                except Exception as e: #pylint: disable=W0703
                    print('Caught exception tiling image: ' + self._images[i] + ' -> ' + str(e)
                          + '\nWill not load any tiles from this image')
                    if config.io.stop_on_input_error():
                        print('Aborting processing, set --bypass-input-errors to bypass this error.')
                        raise
                    tiles = [] # Else move past this image without loading any tiles

                random.Random(0).shuffle(tiles) # Gives consistent random ordering so labels will match
                return (i, tiles)

            while indices: # Loop until all input images have been processed

                # Split off a set of input files from the list
                set_size    = config.io.interleave_images()
                current_set = indices[:set_size]
                indices     = indices[set_size:]

                # Convert from indicies into tile lists for this set
                if config.general.verbose():
                    print('Loading tile lists for set of ' + str(set_size) + ' images.')
                current_tiles = [get_image_tile_list(i) for i in current_set]
                if config.general.verbose():
                    print('Done loading set of tile lists, '+str(len(indices))+' indices remaining.')

                empty_tiles = 0
                for it in current_tiles:
                    if not it[1]:
                        empty_tiles += 1
                if config.general.verbose():
                    print('In this set, ' + str(empty_tiles) + ' empty groups.')

                done = False
                tile_count = 0
                while not done:  # Loop through this set of input files and yield interleaved tiles
                    done = True
                    for it in current_tiles:
                        if not it[1]: # No tiles loaded for this input image
                            continue
                        roi = it[1].pop(0) # Get the next tile for this input image
                        if roi:
                            done = False
                            tile_count += 1
                            yield (it[0], roi.min_x, roi.min_y, roi.max_x, roi.max_y)
                    if done:
                        if config.general.verbose():
                            print('Set done with tile count = ' + str(tile_count))
                        break

        # Pass the local function into the dataset generator function
        return tf.data.Dataset.from_generator(tile_generator,
                                              (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32))

    def _load_images(self, is_labels, data_type):
        """
        Loads a list of images as tensors.
        If label_list is specified, load labels instead. The corresponding image files are still required however.
        """
        ds_input = self._tile_images()
        tile_shape = self._tile_shape
        def load_tile(image_index, x1, y1, x2, y2):
            img = tf.py_function(functools.partial(self._load_tensor_imagery,
                                                   is_labels),
                                 [image_index, [x1, y1, x2, y2]], data_type)
            if not self._chunk_shape:
                img.set_shape([tile_shape[0], tile_shape[1]] + ([1] if is_labels else [self._num_bands]))
            return img
        ret = ds_input.map(load_tile, num_parallel_calls=
                           tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)

        # Skip past empty inputs
        # - When we skip an image as part of resume it shows up as empty
        ret = ret.filter(lambda n: tf.math.greater(tf.size(n), 0))

        return ret

    def _chunk_image(self, image):
        """Split up a tensor image into tensor chunks"""

        ksizes  = [1, self._chunk_shape[0], self._chunk_shape[1], 1] # Size of the chunks
        strides = [1, self._chunk_stride[0], self._chunk_stride[1], 1] # Spacing between chunk starts
        rates   = [1, 1, 1, 1]
        result  = tf.image.extract_patches(tf.expand_dims(image, 0), ksizes, strides, rates,
                                           padding='VALID')
        # Output is [1, M, N, chunk*chunk*bands]
        result = tf.reshape(result, [-1, self._chunk_shape[0], self._chunk_shape[1], self._num_bands])

        return result

    def _reshape_labels(self, labels):
        """Reshape the labels to account for the chunking process."""
        if self._chunk_shape:
            w = (self._chunk_shape[0] - self._output_shape[0]) // 2
            h = (self._chunk_shape[1] - self._output_shape[1]) // 2
        else:
            w = (tf.shape(labels)[0] - self._output_shape[0]) // 2
            h = (tf.shape(labels)[1] - self._output_shape[1]) // 2
        labels = tf.image.crop_to_bounding_box(labels, w, h, tf.shape(labels)[0] - 2 * w,
                                               tf.shape(labels)[1] - 2 * h)
        if not self._chunk_shape:
            return labels

        ksizes  = [1, self._output_shape[0], self._output_shape[1], 1]
        strides = [1, self._chunk_stride[0], self._chunk_stride[1], 1]
        rates   = [1, 1, 1, 1]
        labels = tf.image.extract_patches(tf.expand_dims(labels, 0), ksizes, strides, rates,
                                          padding='VALID')
        result = tf.reshape(labels, [-1, self._output_shape[0], self._output_shape[1]])
        return result

    def data(self):
        """
        Unbatched dataset of image chunks.
        """
        ret = self._load_images(False, self._data_type)
        if self._chunk_shape:
            ret = ret.map(self._chunk_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return ret.unbatch()
        return ret

    def labels(self):
        """
        Unbatched dataset of labels.
        """
        label_set = self._load_images(True, self._label_type)
        if self._chunk_shape or self._output_shape:
            label_set = label_set.map(self._reshape_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE) #pylint: disable=C0301
            if self._chunk_shape:
                return label_set.unbatch()
        return label_set

    def dataset(self, class_weights=None):
        """
        Return the underlying TensorFlow dataset object that this class creates.

        class_weights: list of weights in the classes.
        If class_weights is specified, returns a dataset of (data, labels, weights) instead.
        """

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((self.data(), self.labels()))
        # ignore chunks which are all nodata (nodata is re-indexed to be after the classes)
        if self._labels.nodata_value() is not None:
            ds = ds.filter(lambda x, y: tf.math.reduce_any(tf.math.not_equal(y, self._labels.nodata_value())))
        if class_weights is not None:
            class_weights.append(0.0)
            lookup = tf.constant(class_weights)
            ds = ds.map(lambda x, y: (x, y, tf.gather(lookup, tf.cast(y, tf.int32), axis=None)),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    def num_bands(self):
        """Return the number of bands in each image of the data set"""
        return self._num_bands

    def set_chunk_output_shapes(self, chunk_shape, output_shape):
        if chunk_shape:
            assert len(chunk_shape) == 2, 'Chunk must be two dimensional.'
            assert (chunk_shape[0] % 2) == (chunk_shape[1] % 2) == \
                   (output_shape[0] % 2) == (output_shape[1] % 2), 'Chunk and output shapes must both be even or odd.'
        if output_shape:
            assert len(output_shape) == 2, 'Output must be two dimensional.'
        self._chunk_shape = chunk_shape
        self._output_shape = output_shape

    def chunk_shape(self):
        """
        Size of chunks used for inputs.
        """
        return self._chunk_shape
    def input_shape(self):
        """Input size for the network."""
        if self._chunk_shape:
            return (self._chunk_shape[0], self._chunk_shape[1], self._num_bands)
        return (None, None, self._num_bands)
    def output_shape(self):
        """Output size of blocks of labels"""
        if self._output_shape:
            return (self._output_shape[0], self._output_shape[1], self._output_dims)
        return (None, None, self._output_dims)

    def image_set(self):
        """Returns set of images"""
        return self._images
    def label_set(self):
        """Returns set of label images"""
        return self._labels

    def set_tile_shape(self, tile_shape):
        """Set the tile size."""
        self._tile_shape = tile_shape

    def tile_shape(self):
        """Returns tile size."""
        return self._tile_shape

class AutoencoderDataset(ImageryDataset):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""

    def __init__(self, images, chunk_shape, chunk_stride=(1, 1), tile_shape=(256, 256)):
        """
        The images are used as labels as well.
        """
        super().__init__(images, None, chunk_shape, chunk_shape, tile_shape=tile_shape,
                         chunk_stride=chunk_stride)
        self._labels = self._images
        self._output_dims = self.num_bands()

    def labels(self):
        return self.data()

    def dataset(self, class_weights=None):
        return self.data().map(lambda x: (x, x))
