"""
Functions for converting input images to TFRecords
"""
import os
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from delta.imagery import utilities
from delta.imagery import rectangle
from delta.imagery.sources import landsat, tiff, tfrecord, worldview


#------------------------------------------------------------------------------

def compress_tfrecord_file(input_path, output_path):
    """Make a compressed copy of an uncompressed TFRecord file"""

    writer   = tfrecord.make_tfrecord_writer(output_path, compress=True)
    reader   = tf.data.TFRecordDataset(input_path, compression_type="")
    iterator = reader.make_one_shot_iterator()

    next_element = iterator.get_next()
    sess = tf.Session()

    count = 0
    while True:
        try:
            value = sess.run(next_element)
            writer.write(value)
            count += 1

        except OutOfRangeError:
            break
    return count

def _convert_image_to_tfrecord_tif(input_path):
    """Convert one input tif image"""
    return (tiff.TiffImage([input_path]), None)

def _convert_image_to_tfrecord_rgba(input_path):
    """Ignore the 4th channel of an RGBA image"""
    return (tiff.TiffImage([input_path]), [1,2,3])


def _convert_image_to_tfrecord_landsat(input_path):
    """Convert one input Landsat file (containing multiple tif tiles)"""

    image = landsat.LandsatImage(input_path)
    landsat.toa_preprocess(image, calc_reflectance=True)

    return (image, None)


def _convert_image_to_tfrecord_worldview(input_path):
    """Convert one input WorldView file"""

    bands_to_use = [0, 1, 2, 3, 4, 5, 6, 7]

    image = worldview.WorldviewImage(input_path)
    worldview.toa_preprocess(image, calc_reflectance=False)

    return (image, bands_to_use)


def convert_image_to_tfrecord(input_path, output_paths, tile_size, image_type,
                              redo=False, tile_overlap=0):
    """
    Convert a single image file (possibly compressed) of image_type into TFRecord format.
    If one output path is provided, all output data will be stored there compressed.  If
    multiple output paths are provided, the output data will be divided randomly among those
    files and they will be uncompressed.
    """

    single_output = (len(output_paths) == 1)

    CONVERT_FUNCTIONS = {'worldview':_convert_image_to_tfrecord_worldview,
                         'landsat'  :_convert_image_to_tfrecord_landsat,
                         'tif'      :_convert_image_to_tfrecord_tif,
                         'rgba'     :_convert_image_to_tfrecord_rgba}
    try:
        function = CONVERT_FUNCTIONS[image_type]
    except KeyError:
        raise Exception('Unrecognized image type: ' + image_type)

    # Generate the intermediate tiff files
    image, bands_to_use = function(input_path)

    if single_output and utilities.file_is_good(output_paths[0]) and not redo:
        print('Using existing TFRecord file: ' + str(output_paths))
    else:
        print('Applying conversions and writing tfrecord file...')
        tfrecord.image_to_tfrecord(image, output_paths, tile_size, bands_to_use, tile_overlap, show_progress=True)

    return (image.size(), image.metadata())

# TODO: do we really need this
def convert_and_divide_worldview(input_path, output_prefix, work_folder, is_label, tile_size,
                                 keep=False, redo=False, tile_overlap=0):
    """Specialized convertion function that splits one Worldview image into 8 output TFrecord files."""

    if not os.path.exists(work_folder):
        os.mkdir(work_folder)

    # Generate the intermediate tiff files
    if is_label:
        tif_paths, bands_to_use = _convert_image_to_tfrecord_tif(input_path)
    else:
        tif_paths, bands_to_use = _convert_image_to_tfrecord_worldview(input_path)

    # Gather some image information which is hard to get later on
    reader = tiff.TiffImage(tif_paths[0])
    image_size = reader.size()
    metadata   = reader.metadata()

    # Split the image into eight parts
    split_width  = image_size[0] / 4
    split_height = image_size[1] / 2
    rect = rectangle.Rectangle(0,0,width=image_size[0],height=image_size[1])
    rois = rect.make_tile_rois(split_width, split_height, include_partials=False, overlap_amount=0)

    for (i, roi) in enumerate(rois):

        # Use gdal_translate to create a subset of the TOA image
        output_path  = output_prefix + '_section_' + str(i) + '.tfrecord'
        section_path = os.path.join(work_folder, 'section_' + str(i) + '.tif')
        if utilities.file_is_good(section_path) and not redo:
            print('Section file already exists: ' + section_path)
        else:
            cmd = ('gdal_translate %s %s -srcwin %d %d %d %d'
                   % (tif_paths[0], section_path, roi.min_x, roi.min_y, roi.width(), roi.height()))
            print(cmd)
            os.system(cmd)

        # Convert the subset image to tfrecord
        if utilities.file_is_good(output_path) and not redo:
            print('Using existing TFRecord file: ' + str(output_path))
        else:
            image = tiff.TiffImage([section_path])
            tfrecord.image_to_tfrecord(image, [output_path], tile_size, bands_to_use, tile_overlap, show_progress=True)

    if not keep: # Remove all of the temporary files
        os.system('rm -rf ' + work_folder)

    return (image_size, metadata)
