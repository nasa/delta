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
import shutil
import portalocker

from delta.config import config
from delta.imagery import utilities
from . import tiff


# Using the .vrt does not make much sense with SNAP but it is consistent
#  with the gdalbuildvrt option and makes it easier to search for unpacked
#  Sentinel1 images
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

    return tiff_files

def run_ffilipponi_preprocessing(source_file, target_file):
    '''Execute the ffilipponi SNAP preprocessing script using the gpt tool

       Filipponi, F. (2019). Sentinel-1 GRD Preprocessing Workflow. In
       Multidisciplinary Digital Publishing Institute Proceedings (Vol. 18, No. 1, p. 11).

       https://github.com/ffilipponi/Sentinel-1_GRD_preprocessing
    '''

    gpt_path    = shutil.which('gpt')
    this_folder = os.path.dirname(os.path.abspath(__file__))
    xml_path    = os.path.join(this_folder, 'sentinel1_ffilipponi_snap_preprocess_graph.xml')

    parameters = """-Pfilter='None' -Presolution=10.0 -Porigin=5.0 -Pdem='SRTM 1Sec HGT' -Pfilter='None' -Pcrs='GEOGCS["WGS84(DD)", DATUM["WGS84", SPHEROID["WGS84", 6378137.0, 298.257223563]], PRIMEM["Greenwich", 0.0], UNIT["degree", 0.017453292519943295], AXIS["Geodetic longitude", EAST], AXIS["Geodetic latitude", NORTH]]'""" ##pylint: disable=C0301

    cmd = gpt_path +' '+ xml_path +' -e -Poutput=' + target_file + ' -Pinput=' + source_file +' '+ parameters
    print(cmd)
    os.system(cmd)


def unpack_s1_to_folder(zip_path, unpack_folder):
    '''Returns the merged image path from the unpack folder.
       Unpacks the zip file and merges the source images as needed.'''

    if shutil.which('gpt') is None:
        raise Exception('Could not find the tool "gpt", do you have ESA SNAP installed and on your PATH?')

    with portalocker.Lock(zip_path, 'r', timeout=300) as unused: #pylint: disable=W0612

        merged_path = get_merged_path(unpack_folder)
        try:
            test_image = tiff.TiffImage(merged_path) #pylint: disable=W0612
        except Exception: #pylint: disable=W0703
            test_image = None

        if test_image: # Merged image is ready to use
            print('Already have unpacked files in ' + unpack_folder)
            return merged_path
        # Otherwise go through the entire unpack process

        NUM_SOURCE_CHANNELS = 2
        need_to_unpack = True
        if os.path.exists(unpack_folder):
            source_image_paths = get_files_from_unpack_folder(unpack_folder)
            if len(source_image_paths) == NUM_SOURCE_CHANNELS:
                need_to_unpack = False
                print('Already have files')
            else:
                print('Clearing unpack folder missing image files.')
                os.system('rm -rf ' + unpack_folder)

        if need_to_unpack:
            print('Unpacking file ' + zip_path + ' to folder ' + unpack_folder)
            utilities.unpack_to_folder(zip_path, unpack_folder)
            subdirs = os.listdir(unpack_folder)
            if len(subdirs) != 1:
                raise Exception('Unexpected Sentinel1 subdirectories: ' + str(subdirs))
            cmd = 'mv ' + os.path.join(unpack_folder, subdirs[0]) +'/* ' + unpack_folder
            print(cmd)
            os.system(cmd)
        source_image_paths = get_files_from_unpack_folder(unpack_folder)

        if len(source_image_paths) != NUM_SOURCE_CHANNELS:
            raise Exception('Did not find two image files in ' + zip_path)

        USE_SNAP = True # To get real results we need to use SNAP

        if USE_SNAP: # Requires the Sentinel1 processing software to be installed
            # Run the preconfigured SNAP preprocessing graph
            # - The SNAP tool *must* write to a .tif extension, so we have to
            #   rename the file if we want something else.
            temp_out_path = merged_path.replace('.vrt', '.tif')
            run_ffilipponi_preprocessing(unpack_folder, temp_out_path)

            dimap_path = temp_out_path + '.dim'
            cmd = 'pconvert -s 0,0 -f GeoTIFF-BigTiff -o ' + os.path.dirname(temp_out_path) +' '+ dimap_path
            print(cmd)
            os.system(cmd)
            MIN_IMAGE_SIZE = 1024*1024*500 # 500 MB, expected size is much larger
            if not os.path.exists(temp_out_path):
                raise Exception('Failed to run ESA SNAP preprocessing!')
            if os.path.getsize(temp_out_path) < MIN_IMAGE_SIZE:
                raise Exception('SNAP encountered a problem processing the file!')
            os.system('mv ' + temp_out_path + ' ' + merged_path)
        else:
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

        if ext == '.tif': # Manually unpacked
            tif_path = paths

        assert tif_path is not None, f'Error: Unsupported extension {ext}'

        return [tif_path]
