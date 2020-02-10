"""
Functions to support images stored as TFRecord.
"""
import functools
import os.path
import random

import portalocker
import tensorflow as tf

from delta.imagery import rectangle

from . import delta_image

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

class TFRecordImage(delta_image.DeltaImage):
    def __init__(self, path, compressed=True):
        super(TFRecordImage, self).__init__()
        self._compressed = compressed
        self._num_bands = None
        self._size = None
        self._path = path

    def _read(self, roi, bands, buf=None):
        raise NotImplementedError("Random read access not supported in TFRecord.")

    def __get_num_bands(self):
        if not os.path.exists(self._path):
            raise Exception('Missing file: ' + self._path)

        record_path = tf.convert_to_tensor(self._path)
        if self._compressed:
            dataset = tf.data.TFRecordDataset(record_path, compression_type=TFRECORD_COMPRESSION_TYPE)
        else:
            dataset = tf.data.TFRecordDataset(record_path)

        dataset = dataset.map(lambda x : tf.io.parse_single_example(x, IMAGE_FEATURE_DESCRIPTION))

        iterator = iter(dataset)
        value = next(iterator)
        self._num_bands = int(value['num_bands'])

    def num_bands(self):
        if self._num_bands is None:
            self.__get_num_bands()
        return self._num_bands

    def size(self):
        raise NotImplementedError("TFRecord files don't have a 'size'!")



def __load_tensor(record, num_bands, data_type=tf.float32):
    """Unpacks a single input image section from a TFRecord file we created.
       The image is returned in format [1, channels, height, width]"""

    value = tf.io.parse_single_example(record, IMAGE_FEATURE_DESCRIPTION)
    array = tf.io.decode_raw(value['image_raw'], data_type)
    # num_bands must be static in graph, width and height will not matter after patching
    return tf.reshape(array, [value['width'], value['height'], num_bands])

def create_dataset(file_list, num_bands, data_type, num_parallel_calls=1, compressed=True):
    """
    Returns a tensorflow dataset for the tfrecord files in file_list.

    Each entry is a tensor of size (width, height, num_bands) and of type data_type.
    """
    ds_input = tf.data.Dataset.from_tensor_slices(file_list)
    ds_input = tf.data.TFRecordDataset(ds_input, compression_type=TFRECORD_COMPRESSION_TYPE if compressed else None)
    return ds_input.map(functools.partial(__load_tensor, num_bands=num_bands, data_type=data_type),
                        num_parallel_calls=num_parallel_calls)

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

def write_tfrecord_image(image, tfrecord_writer, col, row):
    """Pack an image stored as a 3D numpy array and write it to an open TFRecord file"""
    array_bytes = image.tostring()
    num_bands = 1
    if len(image.shape) >= 3:
        num_bands = image.shape[2]
    else:
        num_bands = 1
    # Along with the data, record enough info to recreate the image
    data = {'image_raw': _wrap_bytes(array_bytes),
            'num_bands': _wrap_int64(num_bands),
            'col'      : _wrap_int64(col),
            'row'      : _wrap_int64(row),
            'width'    : _wrap_int64(image.shape[0]),
            'height'   : _wrap_int64(image.shape[1]),
            'bytes_per_num': _wrap_int64(4) # TODO: Vary this!
           }

    features= tf.train.Features(feature=data)
    example = tf.train.Example(features=features)

    tfrecord_writer.write(example.SerializeToString())

def image_to_tfrecord(image, record_paths, tile_size=None, bands_to_use=None,
                      overlap_amount=0, include_partials=True, show_progress=False):
    """Convert a TiffImage into a TFRecord file
       split into multiple tiles so that it is easy to read using TensorFlow.
       All bands are used unless bands_to_use is set to a list of zero-indexed band indices.
       If multiple record paths are passed in, each tile is written to a random output file."""
    if not tile_size:
        tile_size = image.size()
    if not bands_to_use:
        bands_to_use = range(image.num_bands())

    # Make a list of output ROIs, only keeping whole ROIs because TF requires them all to be the same size.
    input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())
    output_rois = input_bounds.make_tile_rois(tile_size[0], tile_size[1],
                                              include_partials=include_partials,
                                              overlap_amount=overlap_amount)

    write_compressed = (len(record_paths) == 1)
    if write_compressed:
        # Set up the output file, it will contain all the tiles from this input image.
        partial_name = record_paths[0] + '.partial'
        writer = make_tfrecord_writer(partial_name, compress=True)
        def compressed_write(output_roi, array):
            write_tfrecord_image(array[:, :, bands_to_use], writer, output_roi.min_x, output_roi.min_y)
        complete = False
        try:
            image.process_rois(output_rois, compressed_write, show_progress=show_progress)
            complete = True
        finally:
            writer.close()
            if complete:
                os.rename(partial_name, record_paths[0]) # don't use incomplete files
            else:
                os.remove(partial_name)
    else:
        def mixed_write(output_roi, array):
            this_record_path = random.choice(record_paths)
            if not os.path.exists(this_record_path):
                os.system('touch ' + this_record_path) # Should be safe multithreaded
            # We need to write a new uncompressed tfrecord file,
            # concatenate them together, and then delete the temporary file.
            with portalocker.Lock(this_record_path, 'r') as unused: #pylint: disable=W0612
                temp_path = this_record_path + '_temp.tfrecord'
                this_writer = make_tfrecord_writer(temp_path, compress=False)
                write_tfrecord_image(array[:, :, bands_to_use], this_writer, output_roi.min_x, output_roi.min_y)
                this_writer = None # Make sure the writer is finished
                os.system('cat %s >> %s' % (temp_path, this_record_path))
                os.remove(temp_path)
        image.process_rois(output_rois, mixed_write, show_progress=show_progress)
