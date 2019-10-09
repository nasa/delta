"""
Utilities for writing and reading images to TFRecord files.
"""

import os
import random
import portalocker
import numpy as np

import tensorflow as tf
from delta.imagery import rectangle
from delta.imagery import utilities
from delta.imagery import image_reader

#------------------------------------------------------------------------------


def wrap_int64(value):
    """Helper-function for wrapping an integer so it can be saved to the TFRecords file"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    """Helper-function for wrapping raw bytes so they can be saved to the TFRecords file"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_tfrecord_writer(output_path, compress=True):
    """Set up a TFRecord writer with the correct options"""
    if not compress:
        return tf.python_io.TFRecordWriter(output_path)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    return tf.python_io.TFRecordWriter(output_path, options)

TFRECORD_COMPRESSION_TYPE = 'GZIP' # Needs to be synchronized with the function above

def write_tfrecord_image(image, tfrecord_writer, col, row, width, height, num_bands):
    """Pack an image stored as a 3D numpy array and write it to an open TFRecord file"""
    array_bytes = image.tostring()
    # Along with the data, record enough info to recreate the image
    data = {'image_raw': wrap_bytes(array_bytes),
            'num_bands': wrap_int64(num_bands),
            'col'      : wrap_int64(col),
            'row'      : wrap_int64(row),
            'width'    : wrap_int64(width),
            'height'   : wrap_int64(height),
            'bytes_per_num': wrap_int64(4) # TODO: Vary this!
           }

    features= tf.train.Features(feature=data)
    example = tf.train.Example(features=features)

    tfrecord_writer.write(example.SerializeToString())


# Create a dictionary describing the features.
IMAGE_FEATURE_DESCRIPTION = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'num_bands': tf.FixedLenFeature([], tf.int64),
    'col'      : tf.FixedLenFeature([], tf.int64),
    'row'      : tf.FixedLenFeature([], tf.int64),
    'width'    : tf.FixedLenFeature([], tf.int64),
    'height'   : tf.FixedLenFeature([], tf.int64),
    'bytes_per_num': tf.FixedLenFeature([], tf.int64)
}


def load_tfrecord_raw(example_proto):
    """Just get the handle to the tfrecord element"""
    return tf.parse_single_example(example_proto, IMAGE_FEATURE_DESCRIPTION)

def load_tfrecord_data_element(example_proto, num_bands, height, width):
    """Unpacks a single input image section from a TFRecord file we created.
       Unfortunately we can't dynamically choose the size of the output images in TF so
       they have to be "constant" input arguments.  This means that each tile must be
       the same size!
       The image is returned in format [1, channels, height, width]"""

    value = tf.parse_single_example(example_proto, IMAGE_FEATURE_DESCRIPTION)
    #height = tf.cast(value['height'], tf.int32)
    #width = tf.cast(value['width'], tf.int32)
    #num_bands = tf.cast(value['num_bands'], tf.int32)
    array = tf.decode_raw(value['image_raw'], tf.float32)
    shape = tf.stack([1, height, width, num_bands])
    #tf.print(array.shape, output_stream=sys.stderr)
    #tf.print(shape, output_stream=sys.stderr)
    array2 = tf.reshape(array, shape)

    return array2

def load_tfrecord_label_element(example_proto, num_bands, height, width):
    """Unpacks a single label image section from a TFRecord file we created.
       Very similar to the previous function, but uses a different data type."""

    value = tf.parse_single_example(example_proto, IMAGE_FEATURE_DESCRIPTION)
    array = tf.decode_raw(value['image_raw'], tf.uint8)
    shape = tf.stack([1, height, width, num_bands])
    array2 = tf.reshape(array, shape)
    return array2



def get_record_info(record_path, compressed=True):
    """Queries a record file and returns (num_bands, height, width) of the contained tiles"""

    if not os.path.exists(record_path):
        raise Exception('Missing file: ' + record_path)

    if compressed:
        raw_image_dataset = tf.data.TFRecordDataset(record_path, compression_type='GZIP')
    else:
        raw_image_dataset = tf.data.TFRecordDataset(record_path, compression_type='')

    parsed_image_dataset = raw_image_dataset.map(load_tfrecord_raw)

    iterator = parsed_image_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    with tf.Session() as sess:

        value = sess.run(next_batch)
        num_bands = int(value['num_bands'])
        height = int(value['height'])
        width  = int(value['width'])
        return (num_bands, height, width)


def tiffs_to_tf_record(input_paths, record_paths, tile_size, bands_to_use=None):
    """Convert a image consisting of one or more .tif files into a TFRecord file
       split into multiple tiles so that it is easy to read using TensorFlow.
       All bands are used unless bands_to_use is set to a list of one-indexed band indices,
       in which case there should only be one input path.
       If multiple record paths are passed in, each tile one is written to a random output file."""

    # Open the input image and get information about it
    input_reader = image_reader.MultiTiffFileReader(input_paths)
    (num_cols, num_rows) = input_reader.image_size()
    num_bands = input_reader.num_bands()
    data_type = utilities.gdal_dtype_to_numpy_type(input_reader.data_type())
    #print('Input data type: ' + str(input_reader.data_type()))
    #print('Using output data type: ' + str(data_type))

    # Make a list of output ROIs, only keeping whole ROIs because TF requires them all to be the same size.
    input_bounds = rectangle.Rectangle(0, 0, width=num_cols, height=num_rows)
    output_rois = input_bounds.make_tile_rois(tile_size[0], tile_size[1], include_partials=False)

    write_compressed = (len(record_paths) == 1) or (isinstance(record_paths, str))
    if write_compressed:
        # Set up the output file, it will contain all the tiles from this input image.
        writer = make_tfrecord_writer(record_paths, compress=True)

    if not bands_to_use:
        bands_to_use = range(1,num_bands+1)

    def callback_function(output_roi, read_roi, data_vec):
        """Callback function to write the first channel to the output file."""

        # Figure out where the desired output data falls in read_roi
        ((col, row), (x0, y0, x1, y1)) = image_reader.get_block_and_roi(output_roi, read_roi, tile_size) #pylint: disable=W0612

        # Pack all bands into a numpy array in the shape TF will expect later.
        array = np.zeros(shape=[output_roi.height(), output_roi.width(), num_bands], dtype=data_type)
        for band in bands_to_use:
            band_data = data_vec[band-1]
            array[:,:, band-1] = band_data[y0:y1, x0:x1] # Crop the correct region

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
    input_reader.process_rois(output_rois, callback_function, strict_order=write_compressed)
    if write_compressed:
        print('Done writing: ' + str(input_paths))
    else:
        print('Done writing: ' + input_paths[0])
