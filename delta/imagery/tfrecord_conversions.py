"""
Functions for converting input images to TFRecords
"""
import os
import zipfile

from delta.imagery import utilities #pylint: disable=C0413
from delta.imagery import tfrecord_utils #pylint: disable=C0413
from delta.imagery.sources import landsat #pylint: disable=C0413
from delta.imagery.sources import worldview #pylint: disable=C0413
from delta.imagery.sources import landsat_toa #pylint: disable=C0413
from delta.imagery.sources import worldview_toa #pylint: disable=C0413


#------------------------------------------------------------------------------


def _convert_image_to_tfrecord_tif(input_path, work_folder): #pylint: disable=W0613
    """Convert one input tif image"""
    return ([input_path], None)

def _convert_image_to_tfrecord_rgba(input_path, work_folder): #pylint: disable=W0613
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

    print('TOA conversion...')
    toa_folder = os.path.join(work_folder, 'toa_output')
    landsat_toa.do_landsat_toa_conversion(meta_path, toa_folder, calc_reflectance=True, num_processes=1)

    if not landsat.check_if_files_present(meta_data, toa_folder):
        raise Exception('TOA conversion failed for: ', input_path)
    print('TOA conversion finished')

    toa_paths = landsat.get_band_paths(meta_data, toa_folder, bands_to_use)

    return (toa_paths, None)


def _convert_image_to_tfrecord_worldview(input_path, work_folder):
    """Convert one input WorldView file"""

    toa_path     = os.path.join(work_folder, 'toa.tif')
    scene_info   = worldview.get_scene_info(input_path)
    bands_to_use = worldview.get_worldview_bands_to_use(scene_info['sensor'])

    # Unzip the input file
    print('Unzip file: ', input_path)
    with zipfile.ZipFile(input_path, 'r') as zip_ref:
        zip_ref.extractall(work_folder)

    (tif_path, meta_path) = worldview.get_files_from_unpack_folder(work_folder)

    # TODO: Any benefit to passing in the tile size here?
    print('TOA conversion...')
    # TODO get reflectance working!
    worldview_toa.do_worldview_toa_conversion(tif_path, meta_path, toa_path, calc_reflectance=False)
    if not os.path.exists(toa_path):
        raise Exception('TOA conversion failed for: ', input_path)
    print('TOA conversion finished')

    return ([toa_path], bands_to_use)


def convert_image_to_tfrecord(input_path, output_path, work_folder, tile_size, image_type):
    """Convert a single image file (possibly compressed) of image_type into a single tfrecord
       file at output_path.  work_folder is deleted if the conversion is successful."""

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
    tif_paths, bands_to_use = function(input_path, work_folder)

    tfrecord_utils.tiffs_to_tf_record(tif_paths, output_path, tile_size, bands_to_use)

    # Remove all of the temporary files
    os.system('rm -rf ' + work_folder)
