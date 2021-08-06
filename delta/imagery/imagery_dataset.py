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
import threading
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

        self._iopool = ThreadPoolExecutor(config.io.threads())

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
        self._epoch = [0, 0] # track images and labels separately for simplicity

        # Load the first image to get the number of bands for the input files.
        self._num_bands = images.load(0).num_bands()
        self._random_seed = random.randint(0, 1 << 16)

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
        img = self._images.load(i)

        if self._labels: # If we have labels make sure they are the same size as the input images
            label = self._labels.load(i)
            if label.size() != img.size():
                raise AssertionError('Label file ' + self._labels[i] + ' with size ' + str(label.size())
                                     + ' does not match input image ' + self._images[i] + ' size of ' + str(img.size()))
        tile_shape = self._tile_shape
        if self._chunk_shape:
            assert tile_shape[0] >= self._chunk_shape[0] and \
                   tile_shape[1] >= self._chunk_shape[1], 'Tile too small.'
            return img.tiles((tile_shape[0], tile_shape[1]), min_shape=self._chunk_shape,
                             overlap_shape=(self._chunk_shape[0] - 1, self._chunk_shape[1] - 1),
                             by_block=True)
        return img.tiles((tile_shape[0], tile_shape[1]), partials=False, partials_overlap=True,
                         overlap_shape=self._tile_overlap, by_block=True)

    def _tile_generator(self, is_labels): # pragma: no cover
        """
        A generator that yields image tiles over all images.

        Parameters
        ----------
        is_labels: bool
            Load the label if true, image if false

        Returns
        -------
        Iterator[numpy.ndarray]:
            Iterator over iamge tiles.
        """
        # track epoch (must be same for label and non-label)
        epoch = self._epoch[1 if is_labels else 0]
        self._epoch[1 if is_labels else 0] += 1
        images = [(self._labels if is_labels else self._images).load(i) for i in range(len(self._images))]
        # create lock and get preprocessing function for each image
        image_locks = {}
        image_preprocesses = {}
        for img in images:
            image_locks[img] = threading.Lock()
            image_preprocesses[img] = img.get_preprocess()
            img.set_preprocess(None) # parallelize preprocessing outside lock

        # generator that creates tiles in a random order, but consistent between images and labels
        # returns generator of (img, tile_list) tuples
        def tile_gen():
            # use same seed for labels and not labels, differ by epoch times big prime number
            rand = random.Random(self._random_seed + epoch * 11617)
            image_tiles = [(images[i], self._list_tiles(i)) for i in range(len(images))]
            # shuffle tiles within each image
            for (img, tiles) in image_tiles:
                rand.shuffle(tiles)
            # create iterator
            image_tiles = [(img, iter(tiles)) for (img, tiles) in image_tiles]
            while image_tiles:
                index = rand.randrange(len(image_tiles))
                (img, it) = image_tiles[index]
                try:
                    yield (img, next(it))
                except StopIteration:
                    del image_tiles[index]

        # lock an image and read it. Necessary because gdal doesn't do multi-threading.
        def read_image(img, rect):
            lock = image_locks[img]
            preprocess = image_preprocesses[img]
            lock.acquire()
            buf = img.read(rect)
            lock.release()
            # preprocess outside of lock for concurrency
            if preprocess:
                buf = preprocess(buf, rect, None)
            return buf

        # add a buffer to read to the multiprocessing queue
        def add_to_queue(buf_queue, item):
            (img, (rect, sub_tiles)) = item
            buf = self._iopool.submit(lambda: read_image(img, rect))
            buf_queue.append((rect, sub_tiles, buf))

        gen = tile_gen()
        buf_queue = []
        for _ in range(config.io.threads() * 2): # add a bit ahead
            try:
                next_item = next(gen)
            except StopIteration:
                break
            add_to_queue(buf_queue, next_item)
        # process buffers and yield sub tiles. For efficiency, we just
        # return an entire buffer's sub tiles at once, so not fully random
        while buf_queue:
            (_, sub_tiles, buf) = buf_queue.pop(0)
            buf = buf.result()
            try:
                add_to_queue(buf_queue, next(gen))
            except StopIteration:
                pass

            for s in sub_tiles:
                yield buf[s.min_y:s.max_y, s.min_x:s.max_x, :]

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
        self._epoch[1 if is_labels else 0] = 0 # count epochs for random
        return tf.data.Dataset.from_generator(functools.partial(self._tile_generator,
                                                                is_labels=is_labels),
                                              output_types=data_type,
                                              output_shapes=tf.TensorShape((None, None, None)))

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
            h = (self._chunk_shape[0] - self._output_shape[0]) // 2
            w = (self._chunk_shape[1] - self._output_shape[1]) // 2
        else:
            h = (tf.shape(labels)[0] - self._output_shape[0]) // 2
            w = (tf.shape(labels)[1] - self._output_shape[1]) // 2
        labels = tf.image.crop_to_bounding_box(labels, h, w, tf.shape(labels)[0] - 2 * h,
                                               tf.shape(labels)[1] - 2 * w)
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
