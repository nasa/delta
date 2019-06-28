"""
Tools for loading input images into the TensorFlow Dataset class.
"""
import functools
import os
import os.path
import tempfile
import configparser

import numpy as np
import tensorflow as tf

from delta.imagery import utilities
from delta.imagery import disk_folder_cache
from delta.imagery.sources import basic_sources
from delta.imagery.sources import landsat
from delta.imagery.sources import worldview


# Map text strings to the Image wrapper classes defined above
IMAGE_CLASSES = {
        'landsat' : landsat.LandsatImage,
        'worldview' : worldview.WorldviewImage,
        'rgba' : basic_sources.RGBAImage,
        'tif' : basic_sources.SimpleTiff
}

class ImageryDataset:
    """Create dataset with all files as described in the provided config file.
    """
    def __init__(self, config_path):

        # This contains all of the input options needed for the dataset.
        self._config_values = self._load_config_file(config_path)

        image_type = self._config_values['image_type']
        self._image_class = IMAGE_CLASSES[image_type]

        # Load some optional values from the config file
        try:
            self._chunk_size = int(self._config_values['chunk_size'])
        except IndexError:
            print('"input_dataset:chunk_size" value not found in config file, using default size of 17')
            self._chunk_size = 17
        try:
            self._chunk_overlap = int(self._config_values['chunk_overlap'])
        except IndexError:
            print('"input_dataset:chunk_overlap" value not found in config file, using default value of 0')
            self._chunk_overlap = 0
        try:
            extensions = [self._config_values['extension']]
        except IndexError:
            extensions = image_class().DEFAULT_EXTENSIONS
            print('"input_dataset:extension" value not found in config file, using default value of ' + ext)


        list_path = os.path.join(self._config_values['cache_dir'], 'input_list.csv')

        self._cache_manager = self._image_class.CACHE_CLASS(self._config_values['cache_dir'],
                                                            int(self._config_values['cache_limit']))


        # Generate a text file list of all the input images, plus region indices.
        self._num_regions, self._num_images = self._make_image_list(self._config_values['cache_dir'],
                                                                    list_path, extensions)
        assert self._num_regions > 0

        # Load the first image just to figure out the number of bands
        # - This is the only way to be sure of the number in all cases.
        with open(list_path, 'r') as f:
            line  = f.readline()
            parts = line.split(',')
            image = self.image_class()(parts[0], self._cache_manager)
            self._num_bands = image.get_num_bands()

        # This dataset returns the lines from the text file as entries.
        ds = tf.data.TextLineDataset(list_path)

        # This function generates fake label info for loaded data.
        label_gen_function = functools.partial(self._load_fake_labels)

        def generate_chunks(lines):
            y = tf.py_function(self._load_data, [lines], [tf.float64])
            y[0].set_shape((0, self._num_bands, self._chunk_size, self._chunk_size))
            return y

        def generate_labels(lines):
            y = tf.py_function(label_gen_function, [lines], [tf.int32])
            y[0].set_shape((0, 1))
            return y

        # Tell TF to use the functions above to load our data.
        chunk_set = ds.map(generate_chunks, num_parallel_calls=1)
        label_set = ds.map(generate_labels, num_parallel_calls=1)

        # Break up the chunk sets to individual chunks
        # TODO: Does this improve things?
        chunk_set = chunk_set.flat_map(tf.data.Dataset.from_tensor_slices)
        label_set = label_set.flat_map(tf.data.Dataset.from_tensor_slices)

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((chunk_set, label_set))

        # Filter out all chunks with zero (nodata) values
        self._ds = ds.filter(lambda chunk, label: tf.math.equal(tf.math.zero_fraction(chunk), 0))

    def image_class(self):
        """Return the image handling class for the data type"""
        return self._image_class

    def dataset(self):
        """Return the underlying TensorFlow dataset object that this class creates"""
        return self._ds

    def num_bands(self):
        """Return the number of bands in each image of the data set"""
        return self._num_bands

    def chunk_size(self):
        return self._chunk_size

    def num_images(self):
        """Return the number of images in the data set"""
        return self._num_images

    def total_num_regions(self):
        """Return the number of image/region pairs in the data set"""
        return self._num_regions

    def _load_config_file(self, config_path):
        """Read in the config file from disk"""

        if not os.path.exists(config_path):
            raise Exception('Config file does not exist: ' + config_path)
        config_reader = configparser.ConfigParser()

        try:
            config_reader.read(config_path)
            config_values = config_reader['input_dataset']
        except IndexError:
            raise Exception('Failed to read "input_dataset" section from config file!')

        # Items with a default value are handled in the __init__ function.
        REQUIRED_ENTRIES = ['image_type', 'cache_dir', 'cache_limit']
        for entry in REQUIRED_ENTRIES:
            if entry not in config_values:
                raise Exception('Missing required value "input_dataset:%s"' % (entry))

        return config_values

    def _load_data(self, text_line):
        """Load the image chunk data corresponding ot an image/region text line"""
        text_line = text_line.numpy().decode() # Convert from TF to string type
        parts  = text_line.split(',')
        path   = parts[0].strip()

        roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
                                  int(parts[3].strip()), int(parts[4].strip()))

        # Create a new image class instance to handle this input path
        image = self.image_class()(path, self._cache_manager)
        # Load a region of the image and break it up into small image chunks
        return image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap)

    # TODO: delete this and load actual labels
    def _load_fake_labels(self, text_line):
        """Use to generate fake label data for load_image_region"""
        text_line = text_line.numpy().decode() # Convert from TF to string type
        parts  = text_line.split(',')
        path   = parts[0].strip()

        roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
                                  int(parts[3].strip()), int(parts[4].strip()))

        image = self.image_class()(path, self._cache_manager)
        chunk_data = image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap)

        # Make a fake label
        full_shape = chunk_data.shape[0]
        chunk_data = np.zeros(full_shape, dtype=np.int32)
        chunk_data[ 0:10] = 1 # Junk labels
        chunk_data[10:20] = 2
        return chunk_data

    def _make_image_list(self, top_folder, output_path, extensions):
        '''Write a file listing all of the files in a (recursive) folder
           matching the provided extension.
        '''

        num_entries = 0
        num_images = 0
        with open(output_path, 'w') as f:
            for root, dummy_directories, filenames in os.walk(top_folder):
                for filename in filenames:
                    if os.path.splitext(filename)[1] in extensions:
                        path = os.path.join(root, filename)
                        rois = self.image_class()(path, self._cache_manager).tiles()
                        for r in rois:
                            f.write('%s,%d,%d,%d,%d\n' % (path, r.min_x, r.min_y, r.max_x, r.max_y))
                            num_entries += 1
                    num_images += 1
        return num_entries, num_images
