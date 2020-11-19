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
Functions to support Sentinel1 satellites.
"""

import os
import sys
import portalocker

import tensorflow as tf

from delta.config import config
from delta.imagery import utilities
from . import tiff


# Unpack procedure:
# - Start with .zip files
# - Unpack to .SAFE folders containing .tif files
# - Use gdalbuildvrt to creat a merged.vrt file
# - TODO: Still need to perform terrain correction using SNAP!!

def get_merged_path(unpack_folder):
    return os.path.join(unpack_folder, 'merged.vrt')

def get_files_from_unpack_folder(folder):
    """Return the source image file paths from the given unpack folder.
       Returns [] if the files were not found.
    """

    # All of the image files are located in the measurement folier
    measurement_folder = os.path.join(folder, 'measurement')
    if not os.path.exists(folder) or not os.path.exists(measurement_folder):
        return []

    tiff_files = []
    measure_files = os.listdir(measurement_folder)
    for f in measure_files:
        ext = os.path.splitext(f)[1]
        if (ext.lower() == '.tiff') or (ext.lower() == '.tif'):
            tiff_files.append(os.path.join(measurement_folder, f))
            break

    return tiff_files


def unpack_s1_to_folder(zip_path, unpack_folder):
    '''Returns the merged image path from the unpack folder.
       Unpacks the zip file and merges the source images as needed.'''

    with portalocker.Lock(zip_path, 'r', timeout=300) as unused: #pylint: disable=W0612

        merged_path = get_merged_path(unpack_folder)
        try:
            test_image = tiff.TiffImage(merged_path) #pylint: disable=W0612
        except Exception: #pylint: disable=W0703
            test_image = None

        if test_image: # Merged image is ready to use
            tf.print('Already have unpacked files in ' + unpack_folder,
                     output_stream=sys.stdout)
            return merged_path
        # Otherwise go through the entire unpack process

        tf.print('Unpacking file ' + zip_path + ' to folder ' + unpack_folder,
                 output_stream=sys.stdout)
        utilities.unpack_to_folder(zip_path, unpack_folder)
        subdirs = os.listdir(unpack_folder)
        if len(subdirs) != 1:
            raise Exception('Unexpected Sentinel1 subdirectories: ' + str(subdirs))
        cmd = 'mv ' + os.path.join(unpack_folder, subdirs[0]) +'/* ' + unpack_folder
        print(cmd)
        os.system(cmd)
        source_image_paths = get_files_from_unpack_folder(unpack_folder)

        if not source_image_paths:
            raise Exception('Did not find any image files in ' + zip_path)

        # Generate a merged file containing all input images as an N channel image
        cmd = 'gdalbuildvrt -separate ' + merged_path
        for f in source_image_paths:
            cmd += ' ' + f
        print(cmd)
        os.system(cmd)

        # Verify that we generated a valid image file
        try:
            test_image = tiff.TiffImage(merged_path) #pylint: disable=W0612
        except Exception as e: #pylint: disable=W0703
            raise Exception('Failed to generate merged Sentinel1 file: ' + merged_path) from e

    return merged_path


class Sentinel1Image(tiff.TiffImage):
    """Sentinel1 image tensorflow dataset wrapper (see imagery_dataset.py)"""
    def __init__(self, paths, nodata_value=None):
        self._meta_path = None
        self._meta   = None
        self._sensor = None
        self._date   = None
        self._name   = None
        super().__init__(paths, nodata_value)

    def _unpack(self, zip_path):
        # Get the folder where this will be stored from the cache manager
        unpack_folder = config.io.cache.manager().register_item(self._name)

        return unpack_s1_to_folder(zip_path, unpack_folder)

    # This function is currently set up for the HDDS archived WV data, files from other
    #  locations will need to be handled differently.
    def _prep(self, paths):
        """Prepares a Sentinel1 file from the archive for processing.
           Returns the path to the file ready to use.
           --> This version does not do any preprocessing!!!
        """
        assert isinstance(paths, str)
        ext = os.path.splitext(paths)[1]

        tif_path = None
        if ext == '.zip': # Need to unpack

            tif_path = self._unpack(paths)

        if ext == '.vrt': # Already unpacked

            unpack_folder = os.path.dirname(paths)
            tif_path = get_merged_path(unpack_folder)

        assert tif_path is not None, f'Error: Unsupported extension {ext}'

        return [tif_path]
