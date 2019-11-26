"""
Tools for loading input images into the TensorFlow Dataset class.
"""
import functools

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

class ImageryDataset:
    """Create dataset with all files as described in the provided config file.
    """

    def __init__(self, dataset_config, chunk_size, chunk_stride=1):
        """Initialize the dataset based on the specified config values."""

        # Record some of the config values
        self._chunk_size = chunk_size
        self._chunk_stride = chunk_stride

        try:
            # TODO: remove
            self._data_scale_factor  = PREPROCESS_APPROX_MAX_VALUE[dataset_config.image_type()]
        except KeyError:
            print('WARNING: No data scale factor defined for ' + dataset_config.image_type()
                  + ', defaulting to 1.0 (no scaling)')
            self._data_scale_factor  = 1.0

        if dataset_config.file_type() not in IMAGE_CLASSES:
            raise Exception('file_type %s not recognized.' % dataset_config.file_type())
        self._image_class = IMAGE_CLASSES[dataset_config.file_type()]
        self._use_tfrecord = self._image_class is tfrecord.TFRecordImage

        if self._use_tfrecord and dataset_config.label_type() != 'tfrecord':
            raise NotImplementedError('tfrecord images only supported with tfrecord labels.')
        if dataset_config.label_type() not in IMAGE_CLASSES:
            raise Exception('label_type %s not recognized.' % dataset_config.label_type())
        self._label_class = IMAGE_CLASSES[dataset_config.label_type()]

        (self._image_files, self._label_files) = dataset_config.images()

        # Load the first image to get the number of bands for the input files.
        # TODO: remove cache manager and num_regions
        self._num_bands = self._image_class(self._image_files[0]).num_bands()

        # Tell TF to use the functions above to load our data.
        self._num_parallel_calls  = dataset_config.num_threads()
        self._shuffle_buffer_size = dataset_config.shuffle_buffer_size()

    def _load_tensor_imagery(self, image_class, filename, bbox):
        """Loads a single image as a tensor."""
        assert not self._use_tfrecord
        image = image_class(filename.numpy().decode())
        rect = rectangle.Rectangle(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        r = image.read(rect)
        return r

    def _tf_tiles(self, file_list, label_list=None):
        def tile_generator():
            for (i, f) in enumerate(file_list):
                for t in self._image_class(f).tiles():
                    yield (f if label_list is None else label_list[i], t.min_x, t.min_y, t.max_x, t.max_y)
        return tf.data.Dataset.from_generator(tile_generator,
                                              (tf.string, tf.int32, tf.int32, tf.int32, tf.int32))

    def _load_images(self, file_list, num_bands, data_type, label_list=None):
        """
        Loads a list of images as tensors.
        If label_list is specified, load labels instead. The corresponding image files are still required however.
        """
        if self._use_tfrecord:
            ret = tfrecord.create_dataset(file_list if label_list is None else label_list,
                                          num_bands, data_type, self._num_parallel_calls)
        else:
            ds_input = self._tf_tiles(file_list, label_list)
            def load_imagery_class(filename, x1, y1, x2, y2):
                y = tf.py_function(functools.partial(self._load_tensor_imagery,
                                                     self._image_class if label_list is None else self._label_class),
                                   [filename, [x1, y1, x2, y2]], [data_type])
                #y[0].set_shape((self._chunk_size, self._chunk_size, self._num_bands))
                return y
            ret = ds_input.map(load_imagery_class, num_parallel_calls=self._num_parallel_calls)
        # ignore images that are too small to use
        return ret.filter(lambda x: tf.shape(x)[0] >= self._chunk_size and tf.shape(x)[1] >= self._chunk_size)

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

    def _chunk_images(self, tf_images):
        """Chunk a tensor of images."""
        return tf_images.map(functools.partial(self._chunk_tf_image, self._num_bands),
                             num_parallel_calls=self._num_parallel_calls)

    def _reshape_labels(self, labels):
        """Reshape the labels to account for the chunking process."""
        w = self._chunk_size // 2
        labels = tf.image.crop_to_bounding_box(labels, w, w, tf.shape(labels)[0] - 2 * w, tf.shape(labels)[1] - 2 * w)
        labels = labels[0:self._chunk_stride:, 0:self._chunk_stride:]
        return tf.reshape(labels, [-1, 1])

    def data(self):
        chunk_set = self._chunk_images(self._load_images(self._image_files, self._num_bands, tf.float32))

        # Scale data into 0-1 range
        # TODO: remove this?
        chunk_set = chunk_set.map(lambda x: tf.math.divide(x, self._data_scale_factor))

        # Break up the chunk sets to individual chunks
        return chunk_set.flat_map(tf.data.Dataset.from_tensor_slices)

    def labels(self):
        label_set = self._load_images(self._image_files, 1, tf.uint8, self._label_files).map(self._reshape_labels)

        return label_set.flat_map(tf.data.Dataset.from_tensor_slices)

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

    def num_bands(self):
        """Return the number of bands in each image of the data set"""
        return self._num_bands

    def chunk_size(self):
        return self._chunk_size

    def num_images(self):
        """Return the number of images in the data set"""
        return len(self._image_files)

    def scale_factor(self):
        return self._data_scale_factor



class AutoencoderDataset(ImageryDataset):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""

    def labels(self):
        return self.data()

class ClassifyDataset(ImageryDataset):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""

    def __init__(self, dataset_config, image_file, chunk_size, chunk_stride=1): #pylint: disable=W0231
        """Initialize the dataset based on the specified config values."""

        # Record some of the config values
        self._chunk_size = chunk_size
        self._chunk_stride = chunk_stride

        try:
            # TODO: remove
            self._data_scale_factor  = PREPROCESS_APPROX_MAX_VALUE[dataset_config.image_type()]
        except KeyError:
            print('WARNING: No data scale factor defined for ' + dataset_config.image_type()
                  + ', defaulting to 1.0 (no scaling)')
            self._data_scale_factor  = 1.0

        if dataset_config.file_type() not in IMAGE_CLASSES:
            raise Exception('file_type %s not recognized.' % dataset_config.file_type())
        self._image_class = IMAGE_CLASSES[dataset_config.file_type()]
        self._use_tfrecord = self._image_class is tfrecord.TFRecordImage
        if not self._use_tfrecord:
            raise NotImplementedError('Classification only supported for TFRecord images!')

        self._image_files = [image_file]

        # Load the first image to get the number of bands for the input files.
        self._num_bands = self._image_class(self._image_files[0]).num_bands()

        # Tell TF to use the functions above to load our data.
        self._num_parallel_calls  = dataset_config.num_threads()


    def dataset(self, filter_zero=True, shuffle=True):
        """Return the underlying TensorFlow dataset object that this class creates"""
        # Just classifying so we only need the input images
        return self.data()
