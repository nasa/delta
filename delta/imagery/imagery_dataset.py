"""
Tools for loading input images into the TensorFlow Dataset class.
"""
import functools
import os
import os.path
import math
import psutil

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
        'landsat-simple' : landsat.SimpleLandsatImage,
        'worldview' : worldview.WorldviewImage,
        'rgba' : basic_sources.RGBAImage,
        'tif' : basic_sources.SimpleTiff
}


class ImageryDataset:
    """Create dataset with all files as described in the provided config file.
    """

    def __init__(self, config_values, no_dataset=False):
        """If no_dataset is True, just populate some dataset information and then finish
           without setting up the actual TF dataset."""

        # Record some of the config values
        self._chunk_size    = config_values['ml']['chunk_size']
        self._chunk_overlap = config_values['ml']['chunk_overlap']

        # Create an instance of the image class type
        try:
            image_type = config_values['input_dataset']['image_type']
            self._image_class = IMAGE_CLASSES[image_type]
        except IndexError:
            raise Exception('Did not recognize input_dataset:image_type: ' + image_type)

        # Use the image_class object to get the default image extensions
        if config_values['input_dataset']['extension']:
            input_extensions = [config_values['input_dataset']['extension']]
        else:
            input_extensions = self._image_class.DEFAULT_EXTENSIONS
            print('"input_dataset:extension" value not found in config file, using default value of '
                  + str(self._image_class.DEFAULT_EXTENSIONS))

        if config_values['input_dataset']['num_regions']:
            self._regions_per_image = config_values['input_dataset']['num_regions']
        else:
            self._regions_per_image = self._image_class.DEFAULT_NUM_REGIONS
            print('"input_dataset:num_regions" value not found in config file, using default value of '
                  + str(self._image_class.DEFAULT_NUM_REGIONS))

        list_path = os.path.join(config_values['cache']['cache_dir'], 'input_list.csv')

        self._cache_manager = disk_folder_cache.DiskCache(config_values['cache']['cache_dir'],
                                                          config_values['cache']['cache_limit'])

        # Load the first image just to figure out the number of bands
        # - This is the only way to be sure of the number in all cases.
        with open(list_path, 'r') as f:
            line  = f.readline()
            parts = line.split(',')
            image = self.image_class()(parts[0], self._cache_manager, self._regions_per_image)

            self._num_bands = image.get_num_bands()

            # Automatically adjust the number of image regions if memory is too low
            bytes_needed = image.estimate_memory_usage(self._chunk_size, self._chunk_overlap,
                                                       num_bands = self._num_bands)
            bytes_free   = psutil.virtual_memory().free
            if bytes_needed > bytes_free:
                MEM_EXPAND_FACTOR = 2.0
                ratio = bytes_needed / bytes_free
                new_num_regions = math.floor(ratio * self._regions_per_image * MEM_EXPAND_FACTOR)
                if not no_dataset: # Don't double print this
                    print('Estimated input image region memory usage is ', bytes_needed/utilities.BYTES_PER_GB,
                          ' GB, but only ', bytes_free/utilities.BYTES_PER_GB, ' GB is available. ',
                          'Adjusting number of regions per image from ', self._regions_per_image, ' to ',
                          new_num_regions, ' in order to stay within memory limits.')
                self._regions_per_image = new_num_regions

        # Generate a text file list of all the input images, plus region indices.
        data_folder  = config_values['input_dataset']['data_directory']
        label_folder = config_values['input_dataset']['label_directory']
        self._num_regions, self._num_images = self._make_image_list(data_folder, label_folder,
                                                                    list_path, input_extensions,
                                                                    just_one=no_dataset)
        assert self._num_regions > 0

        if no_dataset: # Quit early
            return

        # This dataset returns the lines from the text file as entries.
        ds = tf.data.TextLineDataset(list_path)

        # This function generates real or fake label info for loaded data.
        label_gen_function = functools.partial(self._load_labels)

        # TODO: Handle data types more carefully!
        def generate_chunks(lines):
            y = tf.py_func(self._load_data, [lines], [tf.float64])
            y[0].set_shape((0, self._num_bands, self._chunk_size, self._chunk_size))
            return y

        def generate_labels(lines):
            y = tf.py_func(label_gen_function, [lines], [tf.int32])
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

    def _load_data(self, text_line):
        """Load the image chunk data corresponding ot an image/region text line"""
        #text_line = text_line.numpy().decode() # Convert from TF to string type
        text_line = tf.compat.as_str_any(text_line) # Convert from TF to string type
        #print('Data: ' + text_line)
        parts  = text_line.split(',')
        path   = parts[0].strip()

        roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
                                  int(parts[3].strip()), int(parts[4].strip()))

        # Create a new image class instance to handle this input path
        image = self.image_class()(path, self._cache_manager, self._regions_per_image)
        # Load a region of the image and break it up into small image chunks
        result = image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap, data_type=np.float64)
        return result

    def _load_labels(self, text_line):
        """Use to generate real or fake label data for load_image_region"""
        #text_line = text_line.numpy().decode() # Convert from TF to string type
        text_line = tf.compat.as_str_any(text_line) # Convert from TF to string type
        #print('Label: ' + text_line)
        parts = text_line.split(',')
        path  = parts[0].strip()

        roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
                                  int(parts[3].strip()), int(parts[4].strip()))

        if len(parts) > 5: # pylint:disable=R1705
            # A label folder was provided
            label_path = parts[5].strip()
            if not os.path.exists(label_path):
                raise Exception('Missing label file: ' + label_path)

            label_image = basic_sources.SimpleTiff(label_path, self._cache_manager, self._regions_per_image)
            chunk_data = label_image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap, data_type=np.int32)
            #num_chunks = chunk_data.shape[0]
            #labels = np.zeros(num_chunks, dtype=np.int32)
            center_pixel = int(self._chunk_size/2)
            labels = chunk_data[:, 0, center_pixel, center_pixel]
            return labels
        else:
            # No labels were provided
            image = self.image_class()(path, self._cache_manager, self._regions_per_image)
            chunk_data = image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap, data_type=np.int32)

            # Make a fake label in the shape matching the input image
            num_chunks = chunk_data.shape[0]
            labels = np.zeros(num_chunks, dtype=np.int32)
            labels[ 0:10] = 1 # Junk labels
            labels[10:20] = 2
            return labels

    def _make_image_list(self, top_folder, label_folder, output_path, extensions,
                         just_one=False):
        '''Write a file listing all of the files in a (recursive) folder
           matching the provided extension.
           If a label folder is provided, look for corresponding label files which
           have the same relative path in that folder but ending with "_label.tif".
           If just_one is set, only find one file!
        '''
        LABEL_POSTFIX = '_label.tif'

        if label_folder:
            if not os.path.exists(label_folder):
                raise Exception('Supplied label folder does not exist: ' + label_folder)
            print('Using image labels from folder: ' + label_folder)
        else:
            print('Using fake label data!')

        num_entries = 0
        num_images  = 0
        with open(output_path, 'w') as f:
            for root, dummy_directories, filenames in os.walk(top_folder):
                for filename in filenames:
                    if os.path.splitext(filename)[1] in extensions:
                        path = os.path.join(root, filename)
                        rois = self.image_class()(path, self._cache_manager, self._regions_per_image).tiles()
                        label_path = None

                        if label_folder: # Append label path to the end of the line
                            rel_path   = os.path.relpath(path, top_folder)
                            label_path = os.path.join(label_folder, rel_path) + LABEL_POSTFIX
                            # If labels are provided then we need a label file for every image in the data set!
                            if not os.path.exists(label_path):
                                raise Exception('Error: Expected label file to exist at path: ' + label_path)

                        for r in rois:
                            line = '%s,%d,%d,%d,%d' % (path, r.min_x, r.min_y, r.max_x, r.max_y)
                            if label_path:
                                line += ',' + label_path
                            f.write(line + '\n')
                            num_entries += 1
                        num_images += 1

                        if just_one:
                            return num_entries, num_images

        return num_entries, num_images
