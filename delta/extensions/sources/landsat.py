# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Read Landsat images.
"""

import math
import functools
import os
import os.path
import numpy as np

from delta.config import config
from delta.imagery import utilities
from . import tiff

# Use this for all the output Landsat data we write.
OUTPUT_NODATA = 0.0

def _parse_mtl_file(mtl_path):
    """Parse out the needed values from the MTL file"""

    if not os.path.exists(mtl_path):
        raise FileNotFoundError('MTL file not found: ' + mtl_path)

    # These are all the values we want to read in
    DESIRED_TAGS = ['FILE_NAME', 'RADIANCE_MULT', 'RADIANCE_ADD',
                    'REFLECTANCE_MULT', 'REFLECTANCE_ADD',
                    'K1_CONSTANT', 'K2_CONSTANT']

    data = {}
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.replace('"','') # Clean up

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

                    if tag not in data:
                        data[tag] = {}
                    if tag == 'FILE_NAME':
                        data[tag][band] = value # String
                    else:
                        data[tag][band] = float(value)

    return data


def get_scene_info(path):
    """Extract information about the landsat scene from the file name"""
    fname  = os.path.basename(path)
    parts  = fname.split('_')
    output = {}
    output['sensor'] = parts[0]
    output['lpath' ] = parts[2][0:3]
    output['lrow'  ] = parts[2][3:6]
    output['date'  ] = parts[3]
    return output

__LANDSAT_BANDS_DICT = {
  '5': [1, 2, 3, 4, 5, 6, 7],
  '7': [1, 2, 3, 4, 5, 6, 7], # Don't forget the extra thermal band!
  '8': [1, 2, 3, 4, 5, 6, 7, 9]
}

def _get_landsat_bands_to_use(sensor_name):
    """Return the list of one-based band indices that we are currently
       using to process the given landsat sensor.
    """

    for (k, v) in __LANDSAT_BANDS_DICT.items():
        if k in sensor_name:
            return v
    print('Unknown landsat type: ' + sensor_name)
    return None

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
        if not os.path.exists(b):
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
    """Compressed Landsat image. Loads a compressed zip or tar file with a .mtl file."""

    def __init__(self, paths, nodata_value=None, bands=None):
        self._bands = bands
        super().__init__(paths, nodata_value)

    def _prep(self, paths):
        """Prepares a Landsat file from the archive for processing.
           Returns [band, paths, in, order, ...]
           Uses the bands specified in _get_landsat_bands_to_use()
           TODO: Handle bands which are not 30 meters!
           TODO: Apply TOA conversion!
        """
        scene_info = get_scene_info(paths)
        self._sensor = scene_info['sensor']
        self._lpath = scene_info['lpath']
        self._lrow = scene_info['lrow']
        self._date = scene_info['date']

        # Get the folder where this will be stored from the cache manager
        name = '_'.join([self._sensor, self._lpath, self._lrow, self._date])
        untar_folder = config.io.cache.manager().register_item(name)

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

        bands_to_use = _get_landsat_bands_to_use(self._sensor) if self._bands is None else self._bands

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
