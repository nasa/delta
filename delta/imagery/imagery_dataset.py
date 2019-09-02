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

#========================================================================================

class ImageryDataset:
    '''
    Interface for loading files for the DELTA project.  We are assuming that 
    the input is going to be rectangular image data with some number of 
    channels > 0.
    '''
    def __init__(self, config_values):
        '''
        '''
        # Record some of the config values
        self._chunk_size    = config_values['ml']['chunk_size']
        self._chunk_overlap = config_values['ml']['chunk_overlap']
        self._ds = None
        self._num_bands = None
        self._num_images = None

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
                                                 list_path, label_list_path, 
                                                 input_extensions)

        # Load the first image to get tile dimensions for the input files.
        # - All of the input tiles must have the same dimensions!
        with open(list_path, 'r') as f:
            line  = f.readline().strip()
            if line == '':
                # Quit early, no data.
                return
            ### end if
            self._num_bands, self._input_region_height, self._input_region_width = tfrecord_utils.get_record_info(line)

        # Tell TF to use the functions above to load our data.
        num_parallel_calls = config_values['input_dataset']['num_input_threads']

        # Go from the file path list to the TFRecord reader
        ds_input = tf.data.TextLineDataset(list_path)
        ds_input = tf.data.TFRecordDataset(ds_input)
        chunk_set = ds_input.map(self._load_data, num_parallel_calls=num_parallel_calls)

        ds_label = tf.data.TextLineDataset(label_list_path)
        ds_label = tf.data.TFRecordDataset(ds_label)
        label_set = ds_label.map(self._load_labels, num_parallel_calls=num_parallel_calls)

        # Break up the chunk sets to individual chunks
        # TODO: Does this improve things?
        chunk_set = chunk_set.flat_map(tf.data.Dataset.from_tensor_slices)
        label_set = label_set.flat_map(tf.data.Dataset.from_tensor_slices)

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((chunk_set, label_set))
        self._ds = self._dataset_filter(ds)

        pass
    ### end __init__

    def _make_image_list(self, data_folder, label_folder, list_path, label_list_path, input_extensions):
        return 0

    def _dataset_filter(self,ds):
        return ds

    def dataset(self):
        """Return the underlying TensorFlow dataset object that this class creates"""
        if self._ds is None:
            raise RuntimeError('The TensorFlow dataset is None. Has the dataset been constructed?')
        ### end if
        return self._ds
    ### end dataset

    def num_bands(self):
        """Return the number of bands in each image of the data set"""
        if self._num_bands is None:
            raise RuntimeError('The number of frequency bands is None. Data has not been loaded?')
        ### end if
        return self._num_bands
    ### end num_bands 

    def chunk_size(self):
        return self._chunk_size
    ### end chunk_size 

    def num_images(self):
        """Return the number of images in the data set"""
        if self._num_images is None:
            raise RuntimeError('The number of images None. Has the data been specified?')
        ### end if
        return self._num_images
    ### end num_images 

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

        # Go from the file path list to the TFRecord reader
        ds_input = tf.data.TextLineDataset(list_path)
        ds_input = tf.data.TFRecordDataset(ds_input, compression_type=tfrecord_utils.TFRECORD_COMPRESSION_TYPE)
        chunk_set = ds_input.map(self._load_data, num_parallel_calls=num_parallel_calls)

        if label_folder:
            ds_label = tf.data.TextLineDataset(label_list_path)
            ds_label = tf.data.TFRecordDataset(ds_label, compression_type=tfrecord_utils.TFRECORD_COMPRESSION_TYPE)
            label_set = ds_label.map(self._load_labels, num_parallel_calls=num_parallel_calls)
        else:
            label_set = ds_input.map(self._load_fake_labels, num_parallel_calls=num_parallel_calls)

        # Break up the chunk sets to individual chunks
        # TODO: Does this improve things?
        chunk_set = chunk_set.flat_map(tf.data.Dataset.from_tensor_slices)
        label_set = label_set.flat_map(tf.data.Dataset.from_tensor_slices)

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((chunk_set, label_set))
# <<<<<<< HEAD
#         # HACK: this is a bad solution.
#         self._steps_per_epoch = len(list(ds))
# 
#         # Filter out all chunks with zero (nodata) values
#         self._ds = ds.filter(lambda chunk, label: tf.math.equal(tf.math.zero_fraction(chunk), 0))
# 
#     def __del__(self):
#         if self.__temp_path is not None:
#             os.remove(self.__temp_path)
# 
#     def image_class(self):
#         return IMAGE_CLASSES[self._image_type]
# 
#     def __load_data(self, text_line):
#         text_line = text_line.numpy().decode() # Convert from TF to string type
#         parts  = text_line.split(',')
#         path   = parts[0].strip()
#         roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
#                                   int(parts[3].strip()), int(parts[4].strip()))
# 
#         image = self.image_class()(path)
#         return image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap)
# 
#     def __make_image_list(self, top_folder, output_path, ext):
#         '''Write a file listing all of the files in a (recursive) folder
#            matching the provided extension.
#         '''
# 
#         num_entries = 0
#         num_images = 0
#         with open(output_path, 'w') as f:
#             for root, dummy_directories, filenames in os.walk(top_folder):
#                 for filename in filenames:
#                     if os.path.splitext(filename)[1] == ext:
#                         path = os.path.join(root, filename)
#                         rois = self.image_class()(path).tiles()
#                         for r in rois:
#                             f.write('%s,%d,%d,%d,%d\n' % (path, r.min_x, r.min_y, r.max_x, r.max_y))
#                             num_entries += 1
#                     num_images += 1
#         return num_entries, num_images
# 
#     def dataset(self):
#         return self._ds
# 
#     def data_shape(self):
#         return (self._chunk_size, self._chunk_size)
# 
#     def steps_per_epoch(self):
#         return self._steps_per_epoch
#     def num_images(self):
#         return
#     def total_num_regions(self):
#         return self._num_regions
# =======

        # Filter out all chunks with zero (nodata) values
        #self._ds = ds.filter(lambda chunk, label: tf.math.equal(tf.math.zero_fraction(chunk), 0))
        self._ds = ds

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

    def _chunk_tf_image(self, image, is_label):
        """Split up a tensor image into tensor chunks"""

        # We use the built-in TF chunking function
        stride = self._chunk_size - self._chunk_overlap
        ksizes  = [1, self._chunk_size, self._chunk_size, 1] # Size of the chunks
        strides = [1, stride, stride, 1] # SPacing between chunk starts
        rates   = [1, 1, 1, 1] # Pixel sampling of the input images within the chunks
        result  = tf.image.extract_image_patches(image, ksizes, strides, rates, padding='VALID')
        if is_label:
            result = tf.reshape(result, [-1, self._chunk_size, self._chunk_size, 1])
        else:
            result = tf.reshape(result, [-1, self._chunk_size, self._chunk_size, self._num_bands])
        # TODO: Change the format we expect to match how the chunks come out?
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
        return result

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


    def _get_label_for_input_image(self, input_path, top_folder, label_folder):
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


class AutoencoderDataset(ImageryDatasetTFRecord):
    """Slightly modified dataset class for the Autoencoder which does not use separate label files"""

#    def __init__(self, config_values, no_dataset=False):
#        # Want to make sure that the input is the same as the output.
#        super(AutoencoderDataset, self).__init__(config_values,no_dataset=no_dataset)

    def _get_label_for_input_image(self, input_path, top_folder, label_folder):
        # For the autoencoder, the label is the same as the input data!
        return input_path


    def _load_labels(self, example_proto):
        return tf.to_int32(self._load_data(example_proto))
