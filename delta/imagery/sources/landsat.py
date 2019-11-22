"""
Functions to support the Landsat satellites.
"""

import math
import functools
import os
import numpy as np

from delta.config import config
from delta.imagery import utilities
from . import tiff

# Use this for all the output Landsat data we write.
OUTPUT_NODATA = 0.0

def _allocate_bands_for_spacecraft(landsat_number):
    """Set up value storage for _parse_mtl_file()"""

    BAND_COUNTS = {'5':7, '7':9, '8':11}

    num_bands = BAND_COUNTS[landsat_number]
    data = dict()

    # There are fewer K constants but we store in the the
    # appropriate band indices.
    data['FILE_NAME'       ] = [''] * num_bands
    data['RADIANCE_MULT'   ] = [None] * num_bands
    data['RADIANCE_ADD'    ] = [None] * num_bands
    data['REFLECTANCE_MULT'] = [None] * num_bands
    data['REFLECTANCE_ADD' ] = [None] * num_bands
    data['K1_CONSTANT'     ] = [None] * num_bands
    data['K2_CONSTANT'     ] = [None] * num_bands

    return data

def _parse_mtl_file(mtl_path):
    """Parse out the needed values from the MTL file"""

    if not os.path.exists(mtl_path):
        raise Exception('MTL file not found: ' + mtl_path)

    # These are all the values we want to read in
    DESIRED_TAGS = ['FILE_NAME', 'RADIANCE_MULT', 'RADIANCE_ADD',
                    'REFLECTANCE_MULT', 'REFLECTANCE_ADD',
                    'K1_CONSTANT', 'K2_CONSTANT']

    data = None
    with open(mtl_path, 'r') as f:
        for line in f:

            line = line.replace('"','') # Clean up

            # Get the spacecraft ID and allocate storage
            if 'SPACECRAFT_ID = LANDSAT_' in line:
                spacecraft_id = line.split('_')[-1].strip()
                data = _allocate_bands_for_spacecraft(spacecraft_id)

            if 'SUN_ELEVATION = ' in line:
                value = line.split('=')[-1].strip()
                data['SUN_ELEVATION'] = float(value)

            # Look for the other info we want
            for tag in DESIRED_TAGS:
                t = tag + '_BAND'
                if t in line: # TODO: Better to do regex here

                    # Break out the name, value, and band
                    parts = line.split('=')
                    name  = parts[0].strip()
                    value = parts[1].strip()
                    try:
                        # Landsat 7 has two thermal readings from the same wavelength band
                        # bit with different gain settings.  Just treat the second file
                        # as another band (9).
                        name = name.replace('BAND_6_VCID_1', 'BAND_6')
                        name = name.replace('BAND_6_VCID_2', 'BAND_9')
                        band  = int(name.split('_')[-1]) -1 # One-based to zero-based
                    except ValueError: # Means this is not a proper match
                        break

                    if tag == 'FILE_NAME':
                        data[tag][band] = value # String
                    else:
                        data[tag][band] = float(value)

    return data


def _get_scene_info(path):
    """Extract information about the landsat scene from the file name"""
    fname  = os.path.basename(path)
    parts  = fname.split('_')
    output = {}
    output['sensor'] = parts[0]
    output['lpath' ] = parts[2][0:3]
    output['lrow'  ] = parts[2][3:6]
    output['date'  ] = parts[3]
    return output

def _get_landsat_bands_to_use(sensor_name):
    """Return the list of one-based band indices that we are currently
       using to process the given landsat sensor.
    """

    # For now just the 30 meter bands, in original order.
    LS5_DESIRED_BANDS = [1, 2, 3, 4, 5, 6, 7]
    LS7_DESIRED_BANDS = [1, 2, 3, 4, 5, 6, 7] # Don't forget the extra thermal band!
    LS8_DESIRED_BANDS = [1, 2, 3, 4, 5, 6, 7, 9]

    if '5' in sensor_name:
        bands = LS5_DESIRED_BANDS
    else:
        if '7' in sensor_name:
            bands = LS7_DESIRED_BANDS
        else:
            if '8' in sensor_name:
                bands = LS8_DESIRED_BANDS
            else:
                raise Exception('Unknown landsat type: ' + sensor_name)
    return bands

def _get_band_paths(mtl_data, folder, bands_to_use=None):
    """Return full paths to all band files that should be in the folder.
       Optionally specify a list of bands to use, otherwise all are used"""

    paths = []
    if not bands_to_use: # Default is to use all bands
        bands_to_use = range(1,len(mtl_data['FILE_NAME'])+1)
    for b in bands_to_use:
        filename = mtl_data['FILE_NAME'][b-1]
        band_path = os.path.join(folder, filename)
        paths.append(band_path)
    return paths

def _check_if_files_present(mtl_data, folder, bands_to_use=None):
    """Return True if all the files associated with the MTL data are present."""

    band_paths = _get_band_paths(mtl_data, folder, bands_to_use)
    for b in band_paths:
        if not utilities.file_is_good(b):
            return False
    return True

def _find_mtl_file(folder):
    """Returns the path to the MTL file in a folder.
       Returns None if there is no MTL file.
       Raises an Exception if there are multiple MTL files."""

    file_list = os.listdir(folder)
    meta_files = [f for f in file_list if '_MTL.txt' in f]
    if len(meta_files) > 1:
        raise Exception('Error: Too many MTL files in ', folder, ', file list: ', str(file_list))
    if not meta_files:
        return None
    return os.path.join(folder, meta_files[0])


class LandsatImage(tiff.TiffImage):
    """Compressed Landsat image tensorflow dataset wrapper (see imagery_dataset.py)"""

    def _prep(self, paths):
        """Prepares a Landsat file from the archive for processing.
           Returns [band, paths, in, order, ...]
           Uses the bands specified in _get_landsat_bands_to_use()
           TODO: Handle bands which are not 30 meters!
           TODO: Apply TOA conversion!
        """
        scene_info = _get_scene_info(paths)
        self._sensor = scene_info['sensor']
        self._lpath = scene_info['lpath']
        self._lrow = scene_info['lrow']
        self._date = scene_info['date']

        # Get the folder where this will be stored from the cache manager
        name = '_'.join([self._sensor, self._lpath, self._lrow, self._date])
        untar_folder = config.cache_manager().register_item(name)

        # Check if we already unpacked this data
        all_files_present = False
        if os.path.exists(untar_folder):
            mtl_path = _find_mtl_file(untar_folder)
            if mtl_path:
                mtl_data = _parse_mtl_file(mtl_path)
                all_files_present = _check_if_files_present(mtl_data, untar_folder)

        if all_files_present:
            print('Already have unpacked files in ' + untar_folder)
        else:
            print('Unpacking tar file ' + paths + ' to folder ' + untar_folder)
            utilities.unpack_to_folder(paths, untar_folder)

        bands_to_use = _get_landsat_bands_to_use(self._sensor)

        # Generate all the band file names (the MTL file is not returned)
        self._mtl_path = _find_mtl_file(untar_folder)
        self._mtl_data = _parse_mtl_file(self._mtl_path)
        output_paths = _get_band_paths(self._mtl_data, untar_folder, bands_to_use)

        # Check that the files exist
        for p in output_paths:
            if not os.path.exists(p):
                raise Exception('Did not find expected file: ' + p
                                + ' after unpacking tar file ' + paths)

        return output_paths

    def radiance_mult(self):
        return self._mtl_data['RADIANCE_MULT']
    def radiance_add(self):
        return self._mtl_data['RADIANCE_ADD']
    def reflectance_mult(self):
        return self._mtl_data['REFLECTANCE_MULT']
    def reflectance_add(self):
        return self._mtl_data['REFLECTANCE_ADD']
    def k1_constant(self):
        return self._mtl_data['K1_CONSTANT']
    def k2_constant(self):
        return self._mtl_data['K2_CONSTANT']
    def sun_elevation(self):
        return self._mtl_data['SUN_ELEVATION']

# top of atmosphere correction
def _apply_toa_radiance(data, _, bands, factors, constants):
    """Apply a top of atmosphere radiance conversion to landsat data"""
    buf = np.zeros(data.shape, dtype=np.float32)
    for b in bands:
        f = factors[b]
        c = constants[b]
        buf[:, :, b] = np.where(data[:, :, b] > 0, data[:, :, b] * f + c, OUTPUT_NODATA)
    return buf

def _apply_toa_temperature(data, _, bands, factors, constants, k1, k2):
    """Apply a top of atmosphere radiance + temp conversion to landsat data"""
    buf = np.zeros(data.shape, dtype=np.float32)
    for b in bands:
        f = factors[b]
        c = constants[b]
        k1 = k1[b]
        k2 = k2[b]
        buf[:, :, b] = np.where(data[:, :, b] > 0, k2 / np.log(k1 / (data[:, :, b] * f + c) + 1.0), OUTPUT_NODATA)
    return buf

def _apply_toa_reflectance(data, _, bands, factors, constants, sun_elevation):
    """Apply a top of atmosphere radiance + temp conversion to landsat data"""
    buf = np.zeros(data.shape, dtype=np.float32)
    for b in bands:
        f = factors[b]
        c = constants[b]
        se = sun_elevation[b]
        buf[:, :, b] = np.where(data[:, :, b] > 0, (data[:, :, b] * f + c) / math.sin(se), OUTPUT_NODATA)
    return buf

def toa_preprocess(image, calc_reflectance=False):
    """Convert landsat files in one folder to TOA corrected files in the output folder.
       Using the reflectance calculation is slightly more complicated but may be more useful.
       Multiprocessing is used if multiple processes are specified."""

    if calc_reflectance:
        if image.k1_constant() is None:
            user_function = functools.partial(_apply_toa_reflectance, factors=image.reflectance_mult(),
                                              constants=image.reflectance_add(),
                                              sun_elevation=math.radians(image.sun_elevation()))
        else:
            user_function = functools.partial(_apply_toa_temperature, factors=image.radiance_mult(),
                                              constants=image.radiance_add(), k1=image.k1_constant(),
                                              k2=image.k2_constant())
    else:
        user_function = functools.partial(_apply_toa_radiance, factors=image.radiance_mult(),
                                          constants=image.radiance_add())

    image.set_preprocess(user_function)
