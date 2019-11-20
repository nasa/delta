"""
Functions to support images stored as TFRecord.
"""
import os.path
import random

import portalocker
import tensorflow as tf

from delta.imagery import rectangle

from . import basic_sources, tiff

# Create a dictionary describing the features.
TFRECORD_COMPRESSION_TYPE = 'GZIP'
IMAGE_FEATURE_DESCRIPTION = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'num_bands': tf.io.FixedLenFeature([], tf.int64),
    'col'      : tf.io.FixedLenFeature([], tf.int64),
    'row'      : tf.io.FixedLenFeature([], tf.int64),
    'width'    : tf.io.FixedLenFeature([], tf.int64),
    'height'   : tf.io.FixedLenFeature([], tf.int64),
    'bytes_per_num': tf.io.FixedLenFeature([], tf.int64)
}

class TFRecordImage(basic_sources.DeltaImage):
    def __init__(self, path, compressed=True):
        super(TFRecordImage, self).__init__()
        self._compressed = compressed
        self._num_bands = None
        self._size = None
        self._path = path

    def _read(self, roi, bands, buf=None):
        raise NotImplementedError("Random read access not supported in TFRecord.")

    def __get_bands_size(self):
        if not os.path.exists(self._path):
            raise Exception('Missing file: ' + self._path)

        record_path = tf.convert_to_tensor(self._path)
        if self._compressed:
            dataset = tf.data.TFRecordDataset(record_path, compression_type='GZIP')
        else:
            dataset = tf.data.TFRecordDataset(record_path, compression_type='')

        dataset = dataset.map(lambda x : tf.io.parse_single_example(x, IMAGE_FEATURE_DESCRIPTION))

        iterator = iter(dataset)
        value = next(iterator)
        self._num_bands = int(value['num_bands'])
        self._size = (int(value['height']), int(value['width']))

    def num_bands(self):
        if self._num_bands is None:
            self.__get_bands_size()
        return self._num_bands

    def size(self):
        if self._size is None:
            self.__get_bands_size()
        return self._size


def load_tensor(tf_filename, num_bands, data_type=tf.float32):
    """Unpacks a single input image section from a TFRecord file we created.
       The image is returned in format [1, channels, height, width]"""

    value = tf.io.parse_single_example(tf_filename, IMAGE_FEATURE_DESCRIPTION)
    array = tf.io.decode_raw(value['image_raw'], data_type)
    # num_bands must be static in graph, width and height will not matter after patching
    shape = tf.stack([1, value['width'], value['height'], num_bands])
    return tf.reshape(array, shape)

def _wrap_int64(value):
    """Helper-function for wrapping an integer so it can be saved to the TFRecords file"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _wrap_bytes(value):
    """Helper-function for wrapping raw bytes so they can be saved to the TFRecords file"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_tfrecord_writer(output_path, compress=True):
    """Set up a TFRecord writer with the correct options"""
    if not compress:
        return tf.io.TFRecordWriter(output_path)
    options = tf.io.TFRecordOptions(TFRECORD_COMPRESSION_TYPE)
    return tf.io.TFRecordWriter(output_path, options)

def write_tfrecord_image(image, tfrecord_writer, col, row, width, height, num_bands):
    """Pack an image stored as a 3D numpy array and write it to an open TFRecord file"""
    array_bytes = image.tostring()
    # Along with the data, record enough info to recreate the image
    data = {'image_raw': _wrap_bytes(array_bytes),
            'num_bands': _wrap_int64(num_bands),
            'col'      : _wrap_int64(col),
            'row'      : _wrap_int64(row),
            'width'    : _wrap_int64(width),
            'height'   : _wrap_int64(height),
            'bytes_per_num': _wrap_int64(4) # TODO: Vary this!
           }

    features= tf.train.Features(feature=data)
    example = tf.train.Example(features=features)

    tfrecord_writer.write(example.SerializeToString())

def image_to_tfrecord(image, record_paths, tile_size, bands_to_use=None, overlap_amount=0, show_progress=True):
    if not bands_to_use:
        bands_to_use = range(image.num_bands())

    # Make a list of output ROIs, only keeping whole ROIs because TF requires them all to be the same size.
    input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())
    include_partials = (overlap_amount > 0) # These are always used together
    output_rois = input_bounds.make_tile_rois(tile_size[0], tile_size[1], include_partials, overlap_amount)

    write_compressed = (len(record_paths) == 1)
    if write_compressed:
        # Set up the output file, it will contain all the tiles from this input image.
        writer = make_tfrecord_writer(record_paths[0], compress=True)

    def callback_function(output_roi, array):
        """Callback function to write the first channel to the output file."""

        if write_compressed: # Single output file
            write_tfrecord_image(array, writer, output_roi.min_x, output_roi.min_y,
                                 output_roi.width(), output_roi.height(), image.num_bands())
        else: # Choose a random output file
            this_record_path = random.choice(record_paths)
            if not os.path.exists(this_record_path):
                os.system('touch ' + this_record_path) # Should be safe multithreaded
            # We need to write a new uncompressed tfrecord file,
            # concatenate them together, and then delete the temporary file.
            print(this_record_path)
            with portalocker.Lock(this_record_path, 'r') as unused: #pylint: disable=W0612
                temp_path = this_record_path + '_temp.tfrecord'
                this_writer = make_tfrecord_writer(temp_path, compress=False)
                write_tfrecord_image(array, this_writer, output_roi.min_x, output_roi.min_y,
                                     output_roi.width(), output_roi.height(), image.num_bands())
                this_writer = None # Make sure the writer is finished
                os.system('cat %s >> %s' % (temp_path, this_record_path))
                os.remove(temp_path)

    print('Writing TFRecord data...')

    # If this is a single file the ROIs must be written out in order, otherwise we don't care.
    image.process_rois(output_rois, callback_function, strict_order=write_compressed, show_progress=show_progress)

def tiffs_to_tf_record(input_paths, record_paths, tile_size,
                       bands_to_use=None, overlap_amount=0):
    """Convert a image consisting of one or more .tif files into a TFRecord file
       split into multiple tiles so that it is easy to read using TensorFlow.
       All bands are used unless bands_to_use is set to a list of one-indexed band indices,
       in which case there should only be one input path.
       If multiple record paths are passed in, each tile one is written to a random output file."""

    # Open the input image and get information about it
    input_reader = tiff.TiffImage(input_paths)
    (num_cols, num_rows) = input_reader.size()
    num_bands = input_reader.num_bands()
    #print('Input data type: ' + str(input_reader.data_type()))
    #print('Using output data type: ' + str(data_type))

    # Make a list of output ROIs, only keeping whole ROIs because TF requires them all to be the same size.
    input_bounds = rectangle.Rectangle(0, 0, width=num_cols, height=num_rows)
    include_partials = (overlap_amount > 0) # These are always used together
    output_rois = input_bounds.make_tile_rois(tile_size[0], tile_size[1],
                                              include_partials, overlap_amount)


    write_compressed = (len(record_paths) == 1)
    if write_compressed:
        # Set up the output file, it will contain all the tiles from this input image.
        writer = make_tfrecord_writer(record_paths[0], compress=True)

    if not bands_to_use:
        bands_to_use = range(num_bands)

    def callback_function(output_roi, array):
        """Callback function to write the first channel to the output file."""

        if write_compressed: # Single output file
            write_tfrecord_image(array, writer, output_roi.min_x, output_roi.min_y,
                                 output_roi.width(), output_roi.height(), num_bands)
        else: # Choose a random output file
            this_record_path = random.choice(record_paths)
            if not os.path.exists(this_record_path):
                os.system('touch ' + this_record_path) # Should be safe multithreaded
            # We need to write a new uncompressed tfrecord file,
            # concatenate them together, and then delete the temporary file.
            with portalocker.Lock(this_record_path, 'r') as unused: #pylint: disable=W0612
                temp_path = this_record_path + '_temp.tfrecord'
                this_writer = make_tfrecord_writer(temp_path, compress=False)
                write_tfrecord_image(array, this_writer, output_roi.min_x, output_roi.min_y,
                                     output_roi.width(), output_roi.height(), num_bands)
                this_writer = None # Make sure the writer is finished
                os.system('cat %s >> %s' % (temp_path, this_record_path))
                os.remove(temp_path)

    print('Writing TFRecord data...')

    # If this is a single file the ROIs must be written out in order, otherwise we don't care.
    input_reader.process_rois(output_rois, callback_function, strict_order=write_compressed, show_progress=True)
