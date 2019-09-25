"""
Tools for loading input images into the TensorFlow Dataset class.
"""
import functools
import os
#import sys
import math
import psutil

import numpy as np
import tensorflow as tf

from delta.imagery import rectangle
from delta.imagery import utilities
from delta.imagery import tfrecord_utils
from delta.imagery import disk_folder_cache
from delta.imagery.sources import basic_sources
from delta.imagery.sources import landsat # TODO: Remove this dependency!
from delta.imagery.sources import worldview # TODO: Remove this dependency!


# Map text strings to the Image wrapper classes defined above
IMAGE_CLASSES = {
        'landsat' : landsat.LandsatImage,
        'landsat-simple' : landsat.SimpleLandsatImage,
        'worldview' : worldview.WorldviewImage,
        'rgba' : basic_sources.RGBAImage,
        'tif' : basic_sources.SimpleTiff
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

#========================================================================================

class ImageryDataset:
    '''
    Interface for loading files for the DELTA project.  We are assuming that
    the input is going to be rectangular image data with some number of
    channels > 0.
    '''
    def __init__(self, config_values, no_dataset=False):
        '''
        '''
        # Record some of the config values
        self._chunk_size    = config_values['ml']['chunk_size']
        self._chunk_overlap = config_values['ml']['chunk_overlap']
        self._ds = None
        self._num_bands  = None
        self._num_images = None

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


        list_path    = os.path.join(config_values['cache']['cache_dir'], 'input_list.csv')
        data_folder  = config_values['input_dataset']['data_directory']
        label_folder = config_values['input_dataset']['label_directory']

        self._cache_manager = disk_folder_cache.DiskCache(config_values['cache']['cache_dir'],
                                                          config_values['cache']['cache_limit'])

        # Generate a text file list of all the input images, plus region indices.
        self._num_regions, self._num_images = self._make_image_list(data_folder, label_folder,
                                                                    list_path, input_extensions,
                                                                    just_one=no_dataset)


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

                # Regenerate the list file with the new number of regions
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
        ds = ds.filter(lambda chunk, label: tf.math.equal(tf.math.zero_fraction(chunk), 0))
        ds = ds.prefetch(None)

        # Filter out all chunks with zero (nodata) values
        self._ds = ds

    ### end __init__

    def image_class(self):
        """Return the image handling class for the data type"""
        return self._image_class

    def dataset(self):
        """Return the underlying TensorFlow dataset object that this class creates"""
        if self._ds is None:
            raise RuntimeError('The TensorFlow dataset is None. Has the dataset been constructed?')
        return self._ds

    def num_bands(self):
        """Return the number of bands in each image of the data set"""
        if self._num_bands is None:
            raise RuntimeError('The number of frequency bands is None. Data has not been loaded?')
        return self._num_bands

    def chunk_size(self):
        return self._chunk_size

    def num_images(self):
        """Return the number of images in the data set"""
        if self._num_images is None:
            raise RuntimeError('The number of images None. Has the data been specified?')
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

        roi = rectangle.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
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

        roi = rectangle.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
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

        if not os.path.exists(top_folder):
            raise Exception('No data found in folder: ' + top_folder)

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
### end class ImageryDataset

# Attempt at optimizing the dataset class to minimize python function use

class ImageryDatasetTFRecord:
    """Create dataset with all files as described in the provided config file.
    """

    def __init__(self, config_values, no_dataset=False):
        """If no_dataset is True, just populate some dataset information and then finish
           without setting up the actual TF dataset."""

        # Record some of the config values
        self._chunk_size    = config_values['ml']['chunk_size']
        self._chunk_overlap = config_values['ml']['chunk_overlap']

        try:
            image_type = config_values['input_dataset']['image_type']
            self._data_scale_factor = PREPROCESS_APPROX_MAX_VALUE[image_type]
        except KeyError:
            print('WARNING: No data scale factor defined for image type: ' + image_type
                  + ', defaulting to 1.0 (no scaling)')
            self._data_scale_factor = 1.0

        # Use the image_class object to get the default image extensions
        if config_values['input_dataset']['extension']:
            input_extensions = [config_values['input_dataset']['extension']]
        else:
            input_extensions = ['.tfrecord']
            print('"input_dataset:extension" value not found in config file, using default value of .tfrecord')

        # TODO: Store somewhere else!
        list_path = os.path.join(config_values['cache']['cache_dir'], 'input_list.csv')
        label_list_path = os.path.join(config_values['cache']['cache_dir'], 'label_list.csv')
        if not os.path.exists(config_values['cache']['cache_dir']):
            os.mkdir(config_values['cache']['cache_dir'])

        # Generate a text file list of all the input images, plus region indices.
        data_folder  = config_values['input_dataset']['data_directory']
        label_folder = config_values['input_dataset']['label_directory']
        self._num_images = self._make_image_list(data_folder, label_folder,
                                                 list_path, label_list_path, input_extensions,
                                                 just_one=no_dataset)

        # Load the first image to get tile dimensions for the input files.
        # - All of the input tiles must have the same dimensions!
        with open(list_path, 'r') as f:
            line  = f.readline().strip()
            self._num_bands, self._input_region_height, self._input_region_width = tfrecord_utils.get_record_info(line)

        if no_dataset: # Quit early
            return

        # Tell TF to use the functions above to load our data.
        num_parallel_calls = config_values['input_dataset']['num_input_threads']

        # Randomize the input image order in Python to simplify working with TF
        shuffle_list, shuffle_label_list = self._load_shuffle_file_lists(list_path, label_list_path)

        # Go from the file path list to the TFRecord reader
        ds_input = tf.data.Dataset.from_tensor_slices(shuffle_list)
        ds_input = tf.data.TFRecordDataset(ds_input, compression_type=tfrecord_utils.TFRECORD_COMPRESSION_TYPE)
        chunk_set = ds_input.map(self._load_data, num_parallel_calls=num_parallel_calls)

        if label_folder:
            ds_label = tf.data.Dataset.from_tensor_slices(shuffle_label_list)
            ds_label = tf.data.TFRecordDataset(ds_label, compression_type=tfrecord_utils.TFRECORD_COMPRESSION_TYPE)
            label_set = ds_label.map(self._load_labels, num_parallel_calls=num_parallel_calls)
        else:
            label_set = ds_input.map(self._load_fake_labels, num_parallel_calls=num_parallel_calls)

        # Break up the chunk sets to individual chunks
        chunk_set = chunk_set.flat_map(tf.data.Dataset.from_tensor_slices)
        label_set = label_set.flat_map(tf.data.Dataset.from_tensor_slices)

        self.ds_chunks = chunk_set

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((chunk_set, label_set))

        def is_chunk_non_zero(data, label): #pylint: disable=W0613
            """Return True if the chunk has no zeros"""
            return tf.math.equal(tf.math.zero_fraction(data), 0)

        # Filter out all chunks with zero (nodata) values
        ds = ds.filter(is_chunk_non_zero)
        #tf.print(tf.shape(ds), output_stream=sys.stderr)

        self._ds = ds.shuffle(buffer_size=config_values['input_dataset']['shuffle_buffer_size'])

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

    def scale_factor(self):
        return self._data_scale_factor

    def _chunk_tf_image(self, image, is_label):
        """Split up a tensor image into tensor chunks"""

        # We use the built-in TF chunking function
        stride  = self._chunk_size - self._chunk_overlap
        ksizes  = [1, self._chunk_size, self._chunk_size, 1] # Size of the chunks
        strides = [1, stride, stride, 1] # SPacing between chunk starts
        rates   = [1, 1, 1, 1] # Pixel sampling of the input images within the chunks
        result  = tf.image.extract_image_patches(image, ksizes, strides, rates, padding='VALID')
        # Output is [1, M, N, chunk*chunk*bands]
        if is_label:
            result = tf.reshape(result, [-1, self._chunk_size, self._chunk_size, 1])
        else:
            result = tf.reshape(result, [-1, self._chunk_size, self._chunk_size, self._num_bands])
        # TODO: Change the format we use to match how the chunks come out?
        result = tf.transpose(result, [0, 3, 1, 2]) # Get info format: [chunks, bands, x, y]
        return result

    def _chunk_label_image(self, image):
        """Split up a tensor label image into tensor chunks.
           This seems like it should be faster, but maybe it isn't!"""

        # Set up parameters for the TF chunking function to extract the chunk centers
        stride = self._chunk_size - self._chunk_overlap
        ksizes  = [1, 1, 1, 1] # Size of the chunks
        strides = [1, stride, stride, 1] # Spacing between chunk starts
        rates   = [1, 1, 1, 1] # Pixel sampling of the input images within the chunks

        # Operate on a cropped input image so we get the same centers as with the full patch case
        offset = int(self._chunk_size / 2)
        height = image.shape[1] - 2*offset
        width  = image.shape[2] - 2*offset
        cropped_image = tf.image.crop_to_bounding_box(image, offset, offset, height, width)

        result = tf.image.extract_image_patches(cropped_image, ksizes, strides, rates, padding='VALID')
        result = tf.reshape(result, [-1, 1, 1, 1])
        return result

    def _load_data(self, example_proto):
        """Load the next TFRecord image segment and split it into chunks"""

        image = tfrecord_utils.load_tfrecord_data_element(example_proto, self._num_bands,
                                                          self._input_region_height, self._input_region_width)
        result = self._chunk_tf_image(image, is_label=False)
#         result = chunk_tf_image(self._chunk_size, self._num_bands, image, is_label=False)
        result = tf.math.divide(result, self._data_scale_factor) # Get into 0-1 range
        return result

    # TODO: Try concatenating the label on to the image, then splitting post-chunk operation!
    # https://stackoverflow.com/questions/54105110/generate-image-patches-with-tf-extract-image-patches-for-a-pair-of-images-effici

    def _load_fake_labels(self, example_proto):
        """Load the next TFRecord image segment and split it into chunks"""

        # Load the input image data normally just to get the size
        chunk_data = self._load_data(example_proto)

        # Make one fake label for each input chunk.
        num_chunks = tf.to_int32(chunk_data.shape[0])
        labels = tf.random_uniform(dtype=tf.int32, minval=0, maxval=10, shape=[num_chunks])
        #tf.print('Loaded label!', output_stream=sys.stderr)
        return labels

    def _load_labels(self, example_proto):

        # Load from the label image in the same way as the input image so we get the locations correct
        NUM_LABEL_BANDS = 1 # This may change later!
        image = tfrecord_utils.load_tfrecord_label_element(example_proto, NUM_LABEL_BANDS,
                                                           self._input_region_height, self._input_region_width)

        chunk_data = self._chunk_tf_image(image, is_label=True) # First method of getting center pixels
        center_pixel = int(self._chunk_size/2)
        labels = tf.to_int32(chunk_data[:, 0, center_pixel, center_pixel])

        #label_data = self._chunk_label_image(image) # Second method, why is it slower?
        #labels = tf.to_int32(label_data[:,0,0,0])
        #diff = tf.math.subtract(labels, label_data) # Error checking
        #tf.print(tf.count_nonzero(diff), output_stream=sys.stderr)
        return labels


    def _get_label_for_input_image(self, input_path, top_folder, label_folder): #pylint: disable=R0201
        """Returns the path to the expected label for for the given input image file"""

        LABEL_EXT = '.tfrecordlabel'

        # Label file should have the same name but different extension in the label folder
        rel_path   = os.path.relpath(input_path, top_folder)
        label_path = os.path.join(label_folder, rel_path)
        label_path = os.path.splitext(label_path)[0] + LABEL_EXT
        # If labels are provided then we need a label file for every image in the data set!
        if not os.path.exists(label_path):
            raise Exception('Error: Expected label file to exist at path: ' + label_path)
        return label_path


    def _make_image_list(self, top_folder, label_folder, input_list_path, label_list_path, #pylint: disable=R0201
                         extensions, just_one=False):
        """Write a file listing all of the files in a (recursive) folder
           matching the provided extension.
           If a label folder is provided, look for corresponding label files which
           have the same relative path in that folder but ending with "_label.tif".
           If just_one is set, only find one file!
        """

        if label_folder:
            if not os.path.exists(label_folder):
                raise Exception('Supplied label folder does not exist: ' + label_folder)
            print('Using image labels from folder: ' + label_folder)
        else:
            print('Using fake label data!')

        num_images  = 0
        f_input = open(input_list_path, 'w')
        f_label = open(label_list_path, 'w')

        for root, dummy_directories, filenames in os.walk(top_folder):
            for filename in filenames:
                if os.path.splitext(filename)[1] in extensions:
                    path = os.path.join(root, filename.strip())

                    f_input.write(path + '\n')

                    if label_folder:
                        label_path = self._get_label_for_input_image(path, top_folder, label_folder)
                        f_label.write(label_path + '\n')

                    num_images += 1

                    if just_one:
                        f_input.close()
                        f_label.close()
                        return num_images

        f_input.close()
        f_label.close()
        return num_images

    def _load_shuffle_file_lists(self, image_list, label_list):
        """Applies the same shuffle to the lists of image and label files and returns as vectors"""

        images = []
        labels = []
        with open(image_list, 'r') as f:
            for line in f:
                images.append(line.strip())
        with open(label_list, 'r') as f:
            for line in f:
                labels.append(line.strip())

        indices = np.arange(self.num_images())
        np.random.shuffle(indices)

        shuffle_images = np.array(images)[indices]
        shuffle_labels = np.array(labels)[indices]
        return (shuffle_images, shuffle_labels)


class AutoencoderDataset(ImageryDatasetTFRecord):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""

#    constructor provided by supercass
#    def __init__(self, config_values, no_dataset=False):

    def _get_label_for_input_image(self, input_path, top_folder, label_folder):
        # For the autoencoder, the label is the same as the input data!
        return input_path


    def _load_labels(self, example_proto):
        #return tf.to_int32(self._load_data(example_proto))
        return self._load_data(example_proto)
