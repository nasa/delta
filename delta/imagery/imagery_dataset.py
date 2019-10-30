"""
Tools for loading input images into the TensorFlow Dataset class.
"""
#pylint: disable=no-self-use,unused-argument,too-many-arguments,too-many-locals,fixme
import functools
import os
import sys #pylint: disable=W0611

import numpy as np
import tensorflow as tf

from delta.imagery import rectangle
from delta.imagery import tfrecord_utils
from delta.imagery import disk_folder_cache
from delta.imagery.sources import basic_sources
from delta.imagery.sources import landsat
from delta.imagery.sources import worldview


# Map text strings to the Image wrapper classes defined above
IMAGE_CLASSES = {
        'landsat' : landsat.LandsatImage,
        'landsat-simple' : landsat.SimpleLandsatImage,
        'worldview' : worldview.WorldviewImage,
        'rgba' : basic_sources.RGBAImage,
        'tif' : basic_sources.SimpleTiff,
        'tfrecord' : basic_sources.TFRecordImage
}

# TODO: Where is a good place for this?
# After preprocessing, these are the rough maximum values for the data.
# - This is used to scale input datasets into the 0-1 range
# - Everything over this will probably saturate the TF network but that is fine for outliers.
# - For WorldView and Landsat this means post TOA processing.
PREPROCESS_APPROX_MAX_VALUE = {'worldview': 120.0,
                               'landsat'  : 120.0, # TODO
                               'tif'      : 255.0,
                               'rgba'     : 255.0}

class ImageryDataset:
    """Create dataset with all files as described in the provided config file.
    """

    def __init__(self, config_values):
        """Initialize the dataset based on the specified config values."""

        # Record some of the config values
        self._chunk_size = config_values['ml']['chunk_size']
        self._chunk_stride = config_values['ml']['chunk_stride']
        self._cache_manager = disk_folder_cache.DiskCache(config_values['cache']['cache_dir'],
                                                          config_values['cache']['cache_limit'])

        try:
            image_type = config_values['input_dataset']['image_type']
            # TODO: remove
            self._data_scale_factor = PREPROCESS_APPROX_MAX_VALUE[image_type]
        except KeyError:
            print('WARNING: No data scale factor defined for image type: ' + image_type
                  + ', defaulting to 1.0 (no scaling)')
            self._data_scale_factor = 1.0
        try:
            file_type = config_values['input_dataset']['file_type']
            self._image_class = IMAGE_CLASSES[file_type]
            self._use_tfrecord = self._image_class is basic_sources.TFRecordImage
            self._label_class = IMAGE_CLASSES[file_type]
        except IndexError:
            raise Exception('Did not recognize input_dataset:image_type: ' + image_type)

        if self._use_tfrecord and config_values['input_dataset']['label_file_type'] != 'tfrecord':
            raise NotImplementedError('tfrecord images only supported with tfrecord labels.')
        try:
            self._label_class = IMAGE_CLASSES[config_values['input_dataset']['label_file_type']]
        except IndexError:
            raise Exception('Did not recognize input_dataset:image_type: ' + image_type)

        # Use the image_class object to get the default image extensions
        if config_values['input_dataset']['extension']:
            input_extensions = [config_values['input_dataset']['extension']]
        else:
            input_extensions = ['.tfrecord']
            print('''"input_dataset:extension" value not found in config file,
                       using default value of .tfrecord''')

        # Generate a text file list of all the input images, plus region indices.
        data_folder  = config_values['input_dataset']['data_directory']
        label_folder = config_values['input_dataset']['label_directory']
        (self._image_files, self._label_files) = self._find_images(data_folder,
                                                                   label_folder,
                                                                   input_extensions,
                                                                   config_values['input_dataset']['label_extension'])

        # Load the first image to get the number of bands for the input files.
        # TODO: remove cache manager and num_regions
        self._num_bands = self._image_class(self._image_files[0],
                                            self._cache_manager,
                                            None).get_num_bands()

        # Tell TF to use the functions above to load our data.
        self._num_parallel_calls  = config_values['input_dataset']['num_input_threads']
        self._shuffle_buffer_size = config_values['input_dataset']['shuffle_buffer_size']

    def _load_tensor_tfrecord(self, num_bands, data_type, element):
        """Loads a single tfrecord image as a tensor."""

        assert self._use_tfrecord
        return tfrecord_utils.load_tfrecord_element(element, num_bands, data_type=data_type)

    def _load_tensor_imagery(self, data_type, image_class, filename, bbox):
        """Loads a single image as a tensor."""
        assert not self._use_tfrecord
        image = image_class(filename.numpy().decode(), self._cache_manager, 1)
        rect = rectangle.Rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        r = image.read(data_type.as_numpy_dtype(), rect)
        return r

    def _tf_tiles(self, file_list, label_list=None):
        def tile_generator():
            for (i, f) in enumerate(file_list):
                for t in self._image_class(f, self._cache_manager, 1).tiles():
                    yield (f if label_list is None else label_list[i], t.min_x, t.min_y, t.max_x, t.max_y)
        return tf.data.Dataset.from_generator(tile_generator,
                                              (tf.string, tf.int32, tf.int32, tf.int32, tf.int32))

    def _load_images(self, file_list, num_bands, data_type, label_list=None):
        """
        Loads a list of images as tensors.
        If label_list is specified, load labels instead. The corresponding image files are still required however.
        """
        if self._use_tfrecord:
            ds_input = tf.data.Dataset.from_tensor_slices(file_list if label_list is None else label_list)
            ds_input = tf.data.TFRecordDataset(ds_input, compression_type=tfrecord_utils.TFRECORD_COMPRESSION_TYPE)
            return ds_input.map(functools.partial(self._load_tensor_tfrecord, num_bands, data_type),
                                num_parallel_calls=self._num_parallel_calls)
        ds_input = self._tf_tiles(file_list, label_list)
        def load_imagery_class(filename, x1, y1, x2, y2):
            y = tf.py_function(functools.partial(self._load_tensor_imagery, data_type,
                                                 self._image_class if label_list is None else self._label_class),
                               [filename, [x1, y1, x2, y2]], [data_type])
            y[0].set_shape((self._chunk_size, self._chunk_size, self._num_bands))
            return tf.stack(y)
        return ds_input.map(load_imagery_class, num_parallel_calls=self._num_parallel_calls)

    def _chunk_tf_image(self, num_bands, image):
        """Split up a tensor image into tensor chunks"""

        ksizes  = [1, self._chunk_size, self._chunk_size, 1] # Size of the chunks
        strides = [1, self._chunk_stride, self._chunk_stride, 1] # SPacing between chunk starts
        rates   = [1, 1, 1, 1]
        result  = tf.image.extract_patches(image, ksizes, strides, rates,
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
        temp = self._chunk_tf_image(1, labels)
        temp = temp[:, self._chunk_size // 2, self._chunk_size // 2]

        w = self._chunk_size // 2
        labels = tf.image.crop_to_bounding_box(labels, w, w, tf.shape(labels)[1] - 2 * w, tf.shape(labels)[2] - 2 * w)
        labels = labels[0:self._chunk_stride:][0:self._chunk_stride:]
        return tf.reshape(labels, [-1, 1])

    def data(self):
        chunk_set = self._chunk_images(self._load_images(self._image_files, self._num_bands, tf.float32))

        # Scale data into 0-1 range
        # TODO: remove this
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

    def _get_label_for_input_image(self, input_path, top_folder, label_folder, label_extension): # pylint: disable=no-self-use

        """Returns the path to the expected label for for the given input image file"""

        # Label file should have the same name but different extension in the label folder
        rel_path   = os.path.relpath(input_path, top_folder)
        label_path = os.path.join(label_folder, rel_path)
        label_path = os.path.splitext(label_path)[0] + label_extension
        # If labels are provided then we need a label file for every image in the data set!
        if not os.path.exists(label_path):
            raise Exception('Error: Expected label file to exist at path: ' + label_path)
        return label_path


    def _find_images(self, top_folder, label_folder, extensions, label_extension):
        """List all of the files in a (recursive) folder matching the provided extension.
           If a label folder is provided, look for corresponding label files which
           have the same relative path in that folder but ending with "_label.tif".
           Returns (image_files, label_files)
        """

        if label_folder:
            if not os.path.exists(label_folder):
                raise Exception('Supplied label folder does not exist: ' + label_folder)
            print('Using image labels from folder: ' + label_folder)
        else:
            print('Using fake label data!')

        image_files = []
        label_files = []

        for root, dummy_directories, filenames in os.walk(top_folder):
            for filename in filenames:
                if os.path.splitext(filename)[1] in extensions:
                    path = os.path.join(root, filename.strip())
                    image_files.append(path)

                    if label_folder:
                        label_path = self._get_label_for_input_image(path, top_folder, label_folder, label_extension)
                        label_files.append(label_path)

        image_files = np.array(image_files)
        label_files = np.array(label_files)
        indices = np.arange(len(image_files))
        np.random.shuffle(indices)

        shuffle_images = image_files[indices]
        shuffle_labels = label_files[indices]
        return (shuffle_images, shuffle_labels)

class AutoencoderDataset(ImageryDataset):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""

    def _get_label_for_input_image(self, input_path, top_folder, label_folder, label_extension): # pylint: disable=no-self-use
        # For the autoencoder, the label is the same as the input data!
        return input_path

    def labels(self):
        return self.data()
