"""
Functions for converting input images to TFRecords
"""
import os
import zipfile
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from delta.imagery import utilities
from delta.imagery import rectangle
from delta.imagery.sources import landsat, tiff, tfrecord, worldview
from delta.imagery.sources import landsat_toa
from delta.imagery.sources import worldview_toa


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

def _convert_image_to_tfrecord_tif(input_path, _):
    """Convert one input tif image"""
    return ([input_path], None)

def _convert_image_to_tfrecord_rgba(input_path, _):
    """Ignore the 4th channel of an RGBA image"""
    return ([input_path], [1,2,3])


def _convert_image_to_tfrecord_landsat(input_path, work_folder):
    """Convert one input Landsat file (containing multiple tif tiles)"""

    scene_info = landsat.get_scene_info(input_path)

    # Unzip the input file
    print('Untar file: ', input_path)
    utilities.unpack_to_folder(input_path, work_folder)

    meta_path = landsat.find_mtl_file(work_folder)
    meta_data = landsat.parse_mtl_file(meta_path)
    bands_to_use = landsat.get_landsat_bands_to_use(scene_info['sensor'])

    toa_folder = os.path.join(work_folder, 'toa_output')
    print('Writing TOA corrected file to %s...' % (toa_folder))
    landsat_toa.do_landsat_toa_conversion(meta_path, toa_folder, calc_reflectance=True, num_processes=1)

    if not landsat.check_if_files_present(meta_data, toa_folder):
        raise Exception('TOA conversion failed for: ', input_path)

    toa_paths = landsat.get_band_paths(meta_data, toa_folder, bands_to_use)

    return (toa_paths, None)


def _convert_image_to_tfrecord_worldview(input_path, work_folder, redo=False):
    """Convert one input WorldView file"""

    toa_path     = os.path.join(work_folder, 'toa.tif')
    license_path = os.path.join(work_folder, 'license', 'NEXTVIEW.TXT')
    scene_info   = worldview.get_scene_info(input_path)
    bands_to_use = worldview.get_worldview_bands_to_use(scene_info['sensor'])

    have_files = False
    if utilities.file_is_good(license_path) and not redo:
        (tif_path, meta_path) = worldview.get_files_from_unpack_folder(work_folder)
        have_files = utilities.file_is_good(tif_path) and utilities.file_is_good(meta_path)

    if have_files:
        print('File ', input_path, ' is already unzipped.')
    else:
        # Unzip the input file
        print('Unzip file: ', input_path)
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(work_folder)


    (tif_path, meta_path) = worldview.get_files_from_unpack_folder(work_folder)

    # TODO: Any benefit to passing in the tile size here?
    # TODO get reflectance working!
    if utilities.file_is_good(toa_path) and not redo:
        print('TOA file ' + toa_path + ' already exists.')
    else:
        print('Writing TOA corrected file to %s...' % (toa_path))
        worldview_toa.do_worldview_toa_conversion(tif_path, meta_path, toa_path, calc_reflectance=False)
    if not os.path.exists(toa_path):
        raise Exception('TOA conversion failed for: ', input_path)

    return ([toa_path], bands_to_use)


def convert_image_to_tfrecord(input_path, output_paths, work_folder, tile_size, image_type,
                              redo=False, tile_overlap=0):
    """Convert a single image file (possibly compressed) of image_type into TFRecord format.
       If one output path is provided, all output data will be stored there compressed.  If
       multiple output paths are provided, the output data will be divided randomly among those
       files and they will be uncompressed.
       work_folder is deleted if the conversion is successful."""

    single_output = (len(output_paths) == 1)

    if not os.path.exists(work_folder):
        os.mkdir(work_folder)

    CONVERT_FUNCTIONS = {'worldview':_convert_image_to_tfrecord_worldview,
                         'landsat'  :_convert_image_to_tfrecord_landsat,
                         'tif'      :_convert_image_to_tfrecord_tif,
                         'rgba'     :_convert_image_to_tfrecord_rgba}
    try:
        function = CONVERT_FUNCTIONS[image_type]
    except KeyError:
        raise Exception('Unrecognized image type: ' + image_type)

    # Generate the intermediate tiff files
    tif_paths, bands_to_use = function(input_path, work_folder) #, redo)

    # Gather some image information which is hard to get later on
    reader = tiff.TiffReader()
    reader.open_image(tif_paths[0]) # TODO: Check this indexing!
    image_size = reader.image_size()
    metadata   = reader.get_all_metadata()

    if single_output and utilities.file_is_good(output_paths[0]) and not redo:
        print('Using existing TFRecord file: ' + str(output_paths))
    else:
        tfrecord.tiffs_to_tf_record(tif_paths, output_paths, tile_size, bands_to_use,
                                    tile_overlap)

    return (image_size, metadata)


def convert_and_divide_worldview(input_path, output_prefix, work_folder, is_label, tile_size,
                                 keep=False, redo=False, tile_overlap=0):
    """Specialized convertion function that splits one Worldview image into 8 output TFrecord files."""

    if not os.path.exists(work_folder):
        os.mkdir(work_folder)

    # Generate the intermediate tiff files
    if is_label:
        tif_paths, bands_to_use = _convert_image_to_tfrecord_tif(input_path, work_folder)
    else:
        tif_paths, bands_to_use = _convert_image_to_tfrecord_worldview(input_path, work_folder, redo)

    # Gather some image information which is hard to get later on
    reader = tiff.TiffReader()
    reader.open_image(tif_paths[0]) # TODO: Check this indexing!
    image_size = reader.image_size()
    metadata   = reader.get_all_metadata()

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
            tfrecord.tiffs_to_tf_record([section_path], [output_path], tile_size, bands_to_use,
                                        tile_overlap)

    if not keep: # Remove all of the temporary files
        os.system('rm -rf ' + work_folder)

    return (image_size, metadata)
