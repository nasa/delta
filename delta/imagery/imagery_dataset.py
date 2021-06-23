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
from concurrent.futures import ThreadPoolExecutor
import functools
import random
import os
import portalocker
import numpy as np
import tensorflow as tf

from delta.config import config

class ImageryDataset: # pylint: disable=too-many-instance-attributes
    """
    A dataset for tiling very large imagery for training with tensorflow.
    """

    def __init__(self, images, labels, output_shape, chunk_shape, stride=None,
                 tile_shape=(256, 256), tile_overlap=None):
        """
        Parameters
        ----------
        images: ImageSet
            Images to train on
        labels: ImageSet
            Corresponding labels to train on
        output_shape: (int, int)
            Shape of the corresponding labels for a given chunk or tile size.
        chunk_shape: (int, int)
            If specified, divide tiles into individual chunks of this shape.
        stride: (int, int)
            Skip this stride between chunks. Only valid with chunk_shape.
        tile_shape: (int, int)
            Size of tiles to load from the images at a time.
        tile_overlap: (int, int)
            If specified, overlap tiles by this amount.
        """

        self._resume_mode = False
        self._log_folder  = None
        self._iopool = ThreadPoolExecutor(1)

        # Record some of the config values
        self.set_chunk_output_shapes(chunk_shape, output_shape)
        self._output_dims  = 1
        # one for imagery, one for labels
        if stride is None:
            stride = (1, 1)
        self._stride = stride
        self._data_type    = tf.float32
        self._label_type   = tf.uint8
        self._tile_shape = tile_shape
        if tile_overlap is None:
            tile_overlap = (0, 0)
        self._tile_overlap = tile_overlap

        if labels:
            assert len(images) == len(labels)
        self._images = images
        self._labels = labels
        self._access_counts = [np.zeros(0, np.uint8), np.zeros(0, np.uint8)]

        # Load the first image to get the number of bands for the input files.
        self._num_bands = images.load(0).num_bands()

    def set_resume_mode(self, resume_mode, log_folder):
        """
        Enable / disable resume mode and configure it.

        Parameters
        ----------
        resume_mode: bool
            If true, log and check access counts for if imagery can be skipped
            this epoch.
        log_folder: str
            Folder to log access counts to
        """
        self._resume_mode = resume_mode
        self._log_folder = log_folder
        if self._log_folder and not os.path.exists(self._log_folder):
            os.mkdir(self._log_folder)

    def _resume_log_path(self, image_id):
        """
        Parameters
        ----------
        image_id: int

        Returns
        -------
        str:
            the path to the read log for an input image
        """
        if not self._log_folder:
            return None
        image_path = self._images[image_id]
        image_name = os.path.basename(image_path)
        file_name  = os.path.splitext(image_name)[0] + '_read.log'
        log_path   = os.path.join(self._log_folder, file_name)
        return log_path

    def resume_log_read(self, image_id): #pylint: disable=R0201
        """
        Reads an access count file containing a boolean and a count.

        Parameters
        ----------
        image_id: int
            Image id to check logs for

        Returns
        -------
        (bool, int):
            need_to_check, access count
            The boolean is set to true if we need to check the count.
        """
        path = self._resume_log_path(image_id)
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

    def resume_log_update(self, image_id, count=None, need_check=False):  #pylint: disable=R0201
        """
        Update logs of when images are read. Should only be needed internally.

        Parameters
        ----------
        image_id: int
            The image to update
        count: int
            Number of tiles that have been read
        need_check: bool
            Set flag for if a check is needed
        """
        log_path  = self._resume_log_path(image_id)
        if not log_path:
            return
        if count is None:
            (_, count) = self.resume_log_read(image_id)
            count += 1
        with portalocker.Lock(log_path, 'w', timeout=300) as f:
            f.write('%d %d' % (int(need_check), count))

    def reset_access_counts(self, set_need_check=False):
        """
        Go through all the access files and reset the counts. Should be done at the end of each epoch.

        Parameters
        ----------
        set_need_check: bool
            if true, keep the count and mark that it needs to be checked. (should be
            set at the start of training)
        """
        if not self._log_folder:
            return
        if config.general.verbose():
            print('Resetting access counts in folder: ' + self._log_folder)
        for i in range(len(self._images)):
            self.resume_log_update(i, count=0, need_check=set_need_check)

    def _list_tiles(self, i): # pragma: no cover
        """
        Parameters
        ----------
        i: int
            Image to list tiles for.

        Returns
        -------
        List[Rectangle]:
            List of tiles to read from the given image
        """
        # If we need to skip this file because of the read count, no need to look up tiles.
        if self._resume_mode:
            file_path = self._images[i]
            log_path  = self._resume_log_path(i)
            if log_path:
                if config.general.verbose():
                    print('get_image_tile_list for index ' + str(i) + ' -> ' + file_path)
                (need_to_check, count) = self.resume_log_read(i)
                if need_to_check and (count > config.train.resume_cutoff()):
                    if config.general.verbose():
                        print('Skipping index ' + str(i) + ' tile gen with count '
                              + str(count) + ' -> ' + file_path)
                    return []
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
                raise AssertionError('Label file ' + self._labels[i] + ' with size ' + str(label.size())
                                     + ' does not match input image size of ' + str(img.size()))
        tile_shape = self._tile_shape
        if self._chunk_shape:
            assert tile_shape[0] >= self._chunk_shape[0] and \
                   tile_shape[1] >= self._chunk_shape[1], 'Tile too small.'
            return img.tiles((tile_shape[0], tile_shape[1]), min_shape=self._chunk_shape,
                             overlap_shape=(self._chunk_shape[0] - 1, self._chunk_shape[1] - 1),
                             by_block=True)
        return img.tiles((tile_shape[0], tile_shape[1]), partials=False, partials_overlap=True,
                         overlap_shape=self._tile_overlap, by_block=True)

    def _tile_generator(self, i, is_labels): # pragma: no cover
        """
        A generator that yields image tiles from the given image.

        Parameters
        ----------
        i: int
            Image id
        is_labels: bool
            Load the label if true, image if false

        Returns
        -------
        Iterator[numpy.ndarray]:
            Iterator over iamge tiles.
        """
        i = int(i)
        tiles = self._list_tiles(i)
        # track epoch (must be same for label and non-label)
        epoch = self._access_counts[1 if is_labels else 0][i]
        self._access_counts[1 if is_labels else 0][i] += 1
        if not tiles:
            return

        # different order each epoch
        random.Random(epoch * i * 11617).shuffle(tiles)

        image = (self._labels if is_labels else self._images).load(i)
        preprocess = image.get_preprocess()
        image.set_preprocess(None) # parallelize the preprocessing, not in disk i/o threadpool
        bands = range(image.num_bands())

        # read one row ahead of what we process now
        next_buf = self._iopool.submit(lambda: image.read(tiles[0][0]))
        for (c, (rect, sub_tiles)) in enumerate(tiles):
            cur_buf = next_buf
            if c + 1 < len(tiles):
                # extra lambda to bind c in closure
                next_buf = self._iopool.submit((lambda x: (lambda: image.read(tiles[x + 1][0])))(c))
            if cur_buf is None:
                continue
            if preprocess:
                buf = preprocess(cur_buf.result(), rect, bands)
            else:
                buf = cur_buf.result()
            (rect, sub_tiles) = tiles[c]
            for s in sub_tiles:
                yield buf[s.min_x:s.max_x, s.min_y:s.max_y, :]

            if not is_labels: # update access count per row
                self.resume_log_update(i, need_check=False)

    def _load_images(self, is_labels, data_type):
        """
        Loads a list of images as tensors.

        Parameters
        ----------
        is_labels: bool
            Load labels if true, images if not
        data_type: numpy.dtype
            Data type that will be returned.

        Returns
        -------
        Dataset:
            Dataset of image tiles
        """
        r = tf.data.Dataset.range(len(self._images))
        r = r.shuffle(1000, seed=0, reshuffle_each_iteration=True) # shuffle same way for labels and non-labels
        self._access_counts[1 if is_labels else 0] = np.zeros(len(self._images), np.uint8) # count epochs for random
        # different seed for each image, use ge
        gen_func = lambda x: tf.data.Dataset.from_generator(functools.partial(self._tile_generator,
                                                                              is_labels=is_labels),
                                                            output_types=data_type,
                                                            output_shapes=tf.TensorShape((None, None, None)), args=(x,))
        return r.interleave(gen_func, cycle_length=config.io.interleave_images(),
                            num_parallel_calls=config.io.threads())

    def _chunk_image(self, image): # pragma: no cover
        """Split up a tensor image into tensor chunks"""

        ksizes  = [1, self._chunk_shape[0], self._chunk_shape[1], 1] # Size of the chunks
        strides = [1, self._stride[0], self._stride[1], 1] # Spacing between chunk starts
        rates   = [1, 1, 1, 1]
        result  = tf.image.extract_patches(tf.expand_dims(image, 0), ksizes, strides, rates,
                                           padding='VALID')
        # Output is [1, M, N, chunk*chunk*bands]
        result = tf.reshape(result, [-1, self._chunk_shape[0], self._chunk_shape[1], self._num_bands])

        return result

    def _reshape_labels(self, labels): # pragma: no cover
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
        strides = [1, self._stride[0], self._stride[1], 1]
        rates   = [1, 1, 1, 1]
        labels = tf.image.extract_patches(tf.expand_dims(labels, 0), ksizes, strides, rates,
                                          padding='VALID')
        result = tf.reshape(labels, [-1, self._output_shape[0], self._output_shape[1]])
        return result

    def data(self):
        """
        Returns
        -------
        Dataset:
            image chunks / tiles.
        """
        ret = self._load_images(False, self._data_type)
        if self._chunk_shape:
            ret = ret.map(self._chunk_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            return ret.unbatch()
        return ret

    def labels(self):
        """
        Returns
        -------
        Dataset:
            Unbatched dataset of labels corresponding to `data()`.
        """
        label_set = self._load_images(True, self._label_type)
        if self._chunk_shape or self._output_shape:
            label_set = label_set.map(self._reshape_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE) #pylint: disable=C0301
            if self._chunk_shape:
                return label_set.unbatch()
        return label_set

    def dataset(self, class_weights=None, augment_function=None):
        """
        Returns a tensorflow dataset as configured by the class.

        Parameters
        ----------
        class_weights: list
            list of weights for the classes.
        augment_function: Callable[[Tensor, Tensor], (Tensor, Tensor)]
            Function to be applied to the image and label before use.

        Returns
        -------
        tensorflow Dataset:
            With (data, labels, optionally weights)
        """

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((self.data(), self.labels()))
        # ignore chunks which are all nodata (nodata is re-indexed to be after the classes)
        if self._labels.nodata_value() is not None:
            ds = ds.filter(lambda x, y: tf.math.reduce_any(tf.math.not_equal(y, self._labels.nodata_value())))
        if augment_function is not None:
            ds = ds.map(augment_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if class_weights is not None:
            class_weights.append(0.0)
            lookup = tf.constant(class_weights)
            ds = ds.map(lambda x, y: (x, y, tf.gather(lookup, tf.cast(y, tf.int32), axis=None)),
                        num_parallel_calls=config.io.threads())
        return ds

    def num_bands(self):
        """
        Returns
        -------
        int:
            number of bands in each image
        """
        return self._num_bands

    def set_chunk_output_shapes(self, chunk_shape, output_shape):
        """
        Parameters
        ----------
        chunk_shape: (int, int)
            Size of chunks to read at a time. Set to None to
            use on a per tile basis (i.e., for FCNs).
        output_shape: (int, int)
            Shape output by the network. May differ from the input size
            (dervied from chunk_shape or tile_shape)
        """
        if chunk_shape:
            assert len(chunk_shape) == 2, 'Chunk must be two dimensional.'
            assert (chunk_shape[0] % 2) == (chunk_shape[1] % 2) == \
                   (output_shape[0] % 2) == (output_shape[1] % 2), 'Chunk and output shapes must both be even or odd.'
        if output_shape:
            assert len(output_shape) == 2 or len(output_shape) == 3, 'Output must be two or three dimensional.'
            if len(output_shape) == 3:
                output_shape = output_shape[0:2]
        self._chunk_shape = chunk_shape
        self._output_shape = output_shape

    def chunk_shape(self):
        """
        Returns
        -------
        (int, int):
            Size of chunks used for inputs.
        """
        return self._chunk_shape

    def input_shape(self):
        """
        Returns
        -------
        Tuple[int, ...]:
            Input size for the network.
        """
        if self._chunk_shape:
            return (self._chunk_shape[0], self._chunk_shape[1], self._num_bands)
        return (None, None, self._num_bands)

    def output_shape(self):
        """
        Returns
        -------
        Tuple[int, ...]:
            Output size, size of blocks of labels
        """
        if self._output_shape:
            return (self._output_shape[0], self._output_shape[1], self._output_dims)
        return (None, None, self._output_dims)

    def image_set(self):
        """
        Returns
        -------
        ImageSet:
            set of images
        """
        return self._images
    def label_set(self):
        """
        Returns
        -------
        ImageSet:
            set of labels
        """
        return self._labels

    def set_tile_shape(self, tile_shape):
        """
        Set the tile size.

        Parameters
        ----------
        tile_shape: (int, int)
            New tile shape"""
        self._tile_shape = tile_shape

    def tile_shape(self):
        """
        Returns
        -------
        Tuple[int, ...]:
            tile shape to load at a time
        """
        return self._tile_shape

    def tile_overlap(self):
        """
        Returns
        -------
        Tuple[int, ...]:
            the amount tiles overlap
        """
        return self._tile_overlap

    def stride(self):
        """
        Returns
        -------
        Tuple[int, ...]:
            Stride between chunks (only when chunk_shape is set).
        """
        return self._stride

class AutoencoderDataset(ImageryDataset):
    """
    Slightly modified dataset class for the autoencoder.

    Instead of specifying labels, the inputs are used as labels.
    """

    def __init__(self, images, chunk_shape, stride=(1, 1), tile_shape=(256, 256), tile_overlap=None):
        super().__init__(images, None, chunk_shape, chunk_shape, tile_shape=tile_shape,
                         stride=stride, tile_overlap=tile_overlap)
        self._labels = self._images
        self._output_dims = self.num_bands()

    def labels(self):
        return self.data()

    def dataset(self, class_weights=None, augment_function=None):
        return self.data().map(lambda x: (x, x))
