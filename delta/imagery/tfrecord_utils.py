"""
Utilities for writing and reading images to TFRecord files.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

import tensorflow as tf #pylint: disable=C0413


#------------------------------------------------------------------------------

# Helper-function for wrapping an integer so it can be saved to the TFRecords file.
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Helper-function for wrapping raw bytes so they can be saved to the TFRecords file.
def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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



def get_record_info(record_path):
    """Queries a record file and returns (num_bands, height, width) of the contained tiles"""

    if not os.path.exists(record_path):
        raise Exception('Missing file: ' + record_path)

    raw_image_dataset = tf.data.TFRecordDataset(record_path)

    parsed_image_dataset = raw_image_dataset.map(load_tfrecord_raw)

    iterator = parsed_image_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    with tf.Session() as sess:

        value = sess.run(next_batch)
        num_bands = int(value['num_bands'])
        height = int(value['height'])
        width  = int(value['width'])
        return (num_bands, height, width)
