"""
Tools for loading input images into the TensorFlow Dataset class.
"""
import functools
import math
import sys

import tensorflow as tf

from delta.imagery import rectangle
from delta.imagery.sources import landsat, worldview, tiff, tfrecord


# Map text strings to the Image wrapper classes defined above
IMAGE_CLASSES = {
        'landsat' : landsat.LandsatImage,
        'worldview' : worldview.WorldviewImage,
        'rgba' : tiff.RGBAImage,
        'tiff' : tiff.TiffImage,
        'tfrecord' : tfrecord.TFRecordImage
}

# TODO: Where is a good place for this?
# After preprocessing, these are the rough maximum values for the data.
# - This is used to scale input datasets into the 0-1 range
# - Everything over this will probably saturate the TF network but that is fine for outliers.
# - For WorldView and Landsat this means post TOA processing.
PREPROCESS_APPROX_MAX_VALUE = {'worldview': 120.0,
                               'landsat'  : 120.0, # TODO
                               'tiff'      : 255.0,
                               'rgba'      : 255.0}

class ImageDataset:
    """
    Create a dataset to load a single file.
    """
    def __init__(self, image_file, label_file, dataset_config, chunk_size, chunk_stride=1):
        # Record some of the config values
        self._chunk_size = chunk_size
        self._chunk_stride = chunk_stride

        try:
            # TODO: remove
            self._data_scale_factor  = PREPROCESS_APPROX_MAX_VALUE[dataset_config.image_type()]
        except KeyError:
            print('WARNING: No data scale factor defined for ' + dataset_config.image_type()
                  + ', defaulting to 1.0 (no scaling)')
            self._data_scale_factor  = None

        if dataset_config.file_type() not in IMAGE_CLASSES:
            raise Exception('file_type %s not recognized.' % dataset_config.file_type())

        image_class = IMAGE_CLASSES[dataset_config.file_type()]
        self._use_tfrecord = image_class is tfrecord.TFRecordImage

        if self._use_tfrecord and dataset_config.label_type() != 'tfrecord':
            raise NotImplementedError('tfrecord images only supported with tfrecord labels.')
        if dataset_config.label_type() not in IMAGE_CLASSES:
            raise Exception('label_type %s not recognized.' % dataset_config.label_type())
        label_class = IMAGE_CLASSES[dataset_config.label_type()]

        # Load the first image to get the number of bands for the input files.
        self._image_file_name = image_file
        self._label_file_name = label_file
        self._image = image_class(image_file)
        self._label = label_class(label_file)

        self._max_block_size = dataset_config.max_block_size()
        self._num_parallel_calls  = dataset_config.num_threads()
        self._shuffle_buffer_size = dataset_config.shuffle_buffer_size()

    def _load_tensor_imagery(self, is_label, scale_factor, bbox):
        """Loads a single image as a tensor."""
        assert not self._use_tfrecord
        w = int(bbox[2])
        h = int(bbox[3])
        rect = rectangle.Rectangle(int(bbox[0]), int(bbox[1]), w, h)
        if is_label:
            r = self._label.read(rect)
        else:
            r = self._image.read(rect)
        # TODO: remove this?
        if scale_factor:
            r = r / scale_factor
        return r

    def _tf_tiles(self):
        max_block_bytes = self._max_block_size * 1024 * 1024
        def tile_generator():
            # TODO: account for other data types properly
            ratio = 5 # TODO: better way to figure this out? want more height because read is contiguous that way
            # w * h * bands * 4 * chunk * chunk = max_block_bytes
            tile_width = int(math.sqrt(max_block_bytes / self._image.num_bands() / 4 / (self._chunk_size ** 2) / ratio))
            tile_height = int(ratio * tile_width)
            if tile_width < self._chunk_size * 2:
                print('Warning: max_block_size is too low. Ignoring.', file=sys.stderr)
                tile_width = self._chunk_size * 2
            if tile_height < self._chunk_size * 2:
                print('Warning: max_block_size is too low. Ignoring.', file=sys.stderr)
                tile_height = self._chunk_size * 2
            for t in self._image.tiles(tile_width, tile_height, overlap=self._chunk_size):
                yield (t.min_x, t.min_y, t.max_x, t.max_y)
        return tf.data.Dataset.from_generator(tile_generator,
                                              (tf.int32, tf.int32, tf.int32, tf.int32))

    def _load_images(self, is_label):
        """
        Loads either the image or label as a tensor.
        """
        file_name = self._image_file_name if not is_label else self._label_file_name
        num_bands = self._image.num_bands() if not is_label else 1
        # TODO: handle other types properly
        data_type = tf.float32 if not is_label else tf.uint8
        if self._use_tfrecord:
            ret = tfrecord.create_dataset([file_name], num_bands, data_type, self._num_parallel_calls)
            # ignore images that are too small to use
            ret = ret.filter(lambda x: tf.shape(x)[0] >= self._chunk_size and tf.shape(x)[1] >= self._chunk_size)
            if not is_label and self._data_scale_factor:
                # Scale data into 0-1 range
                # TODO: remove this?
                ret = ret.map(lambda x: tf.math.divide(x, self._data_scale_factor))
        else:
            tiles = self._tf_tiles()
            def load_imagery_class(x1, y1, x2, y2):
                img = tf.py_function(functools.partial(self._load_tensor_imagery, is_label,
                                                       self._data_scale_factor if not is_label else None),
                                     [[x1, y1, x2, y2]], data_type)
                return img
            ret = tiles.map(load_imagery_class, num_parallel_calls=self._num_parallel_calls)

        return ret.prefetch(tf.data.experimental.AUTOTUNE)

    def _chunk_tf_image(self, num_bands, image):
        """Split up a tensor image into tensor chunks"""

        ksizes  = [1, self._chunk_size, self._chunk_size, 1] # Size of the chunks
        strides = [1, self._chunk_stride, self._chunk_stride, 1] # SPacing between chunk starts
        rates   = [1, 1, 1, 1]
        result  = tf.image.extract_patches(tf.expand_dims(image, 0), ksizes, strides, rates,
                                           padding='VALID')
        # Output is [1, M, N, chunk*chunk*bands]
        result = tf.reshape(result, [-1, self._chunk_size, self._chunk_size, num_bands])

        return result

    def _reshape_labels(self, labels):
        """Reshape the labels to account for the chunking process."""
        w = self._chunk_size // 2
        labels = tf.image.crop_to_bounding_box(labels, w, w, tf.shape(labels)[0] - 2 * w, tf.shape(labels)[1] - 2 * w)
        labels = labels[::self._chunk_stride, ::self._chunk_stride]
        return tf.reshape(labels, [-1, 1])

    def data(self):
        ret = self._load_images(is_label=False)
        ret = ret.map(functools.partial(self._chunk_tf_image, self._image.num_bands()),
                      num_parallel_calls=self._num_parallel_calls)
        ret = ret.prefetch(tf.data.experimental.AUTOTUNE)

        return ret.unbatch()

    def labels(self):
        label_set = self._load_images(is_label=True)
        label_set = label_set.map(self._reshape_labels)

        return label_set.unbatch()

    def dataset(self, filter_zero=True, shuffle=True):
        """Return the underlying TensorFlow dataset object that this class creates"""

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((self.data(), self.labels()))

        def is_chunk_non_zero(data, label):
            """Return True if the chunk has no zeros (nodata) in the data or label"""
            return tf.math.logical_and(tf.math.equal(tf.math.zero_fraction(data ), 0),
                                       tf.math.equal(tf.math.zero_fraction(label), 0))

        ## Filter out all chunks with zero (nodata) values
        if filter_zero:
            ds = ds.filter(is_chunk_non_zero)

        if shuffle:
            ds = ds.shuffle(buffer_size=self._shuffle_buffer_size)

        return ds

class ImageryDataset:
    """
    Create dataset with all files as described in the provided config file.
    """

    def __init__(self, dataset_config, chunk_size, chunk_stride=1, image_files=None, label_files=None):
        """
        Initialize the dataset based on the specified config values.

        If image_files is None, the images and labels from the dataset config are used.
        """

        if image_files is None:
            (image_files, label_files) = dataset_config.images()

        self._images = []
        for (i, img) in enumerate(image_files):
            label = label_files[i] if label_files is not None else None
            self._images.append(ImageDataset(img, label, dataset_config, chunk_size, chunk_stride))

    def __combine_datasets(self, datasets): #pylint:disable=no-self-use
        ds = next(datasets)
        for d in datasets:
            ds = ds.concatenate(d)
        return ds

    def data(self):
        return self.__combine_datasets(map(lambda x: x.data(), self._images))
    def labels(self):
        return self.__combine_datasets(map(lambda x: x.labels(), self._images))
    def dataset(self, filter_zero=True, shuffle=True):
        return self.__combine_datasets(map(lambda x: x.dataset(filter_zero=filter_zero, shuffle=shuffle), self._images))

class AutoencoderDataset(ImageryDataset):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""
    def __init__(self, dataset_config, chunk_size, chunk_stride=1, image_files=None):
        if image_files is None:
            (image_files, _) = dataset_config.images()
        super(AutoencoderDataset, self).__init__(dataset_config, chunk_size, image_files=image_files,
                                                 label_files=image_files, chunk_stride=chunk_stride)
