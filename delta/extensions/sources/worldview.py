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
Functions to support the WorldView satellites.
"""

import zipfile
import functools
import os
import numpy as np
import portalocker

from delta.config import config
from delta.imagery import utilities
from . import tiff

# Use this value for all WorldView nodata values we write, though they usually don't have any nodata.
OUTPUT_NODATA = 0.0

def get_files_from_unpack_folder(folder):
    """Return the image and header file paths from the given unpack folder.
       Returns (None, None) if the files were not found.
    """
    vendor_folder = os.path.join(folder, 'vendor_metadata')
    if not os.path.exists(folder) or not os.path.exists(vendor_folder):
        return (None, None)

    # Check if we already unpacked this data
    imd_path = None
    tif_path = None

    main_files   = os.listdir(folder)
    vendor_files = os.listdir(vendor_folder)
    for f in vendor_files:
        ext = os.path.splitext(f)[1]
        if ext.lower() == '.imd':
            imd_path = os.path.join(vendor_folder, f)
            break
    for f in main_files:
        ext = os.path.splitext(f)[1]
        if ext.lower() == '.tif':
            tif_path = os.path.join(folder, f)
            break
    return (tif_path, imd_path)


def unpack_wv_to_folder(zip_path, unpack_folder):

    with portalocker.Lock(zip_path, 'r', timeout=300) as unused: #pylint: disable=W0612
        # Check if we already unpacked this data
        (tif_path, imd_path) = get_files_from_unpack_folder(unpack_folder)

        if imd_path and tif_path:
            pass
        else:
            print('Unpacking file ' + zip_path + ' to folder ' + unpack_folder)
            utilities.unpack_to_folder(zip_path, unpack_folder)
            # some worldview zip files have a subdirectory with the name of the image
            if not os.path.exists(os.path.join(unpack_folder, 'vendor_metadata')):
                subdir = os.path.join(unpack_folder, os.path.splitext(os.path.basename(zip_path))[0])
                if not os.path.exists(os.path.join(subdir, 'vendor_metadata')):
                    raise Exception('vendor_metadata not found in %s.' % (zip_path))
                for filename in os.listdir(subdir):
                    os.rename(os.path.join(subdir, filename), os.path.join(unpack_folder, filename))
                os.rmdir(subdir)
            (tif_path, imd_path) = get_files_from_unpack_folder(unpack_folder)
    return (tif_path, imd_path)


class WorldviewImage(tiff.TiffImage):
    """Compressed WorldView image. Loads an image from a zip file with a tiff and a .imd file."""
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
        return unpack_wv_to_folder(zip_path, unpack_folder)

    def _set_info_from_tif_name(self, tif_name):
        parts = os.path.basename(tif_name).split('_')
        self._sensor = parts[0][0:4]
        self._date   = parts[2][6:14]
        self._name   = os.path.splitext(os.path.basename(tif_name))[0]


    # This function is currently set up for the HDDS archived WV data, files from other
    #  locations will need to be handled differently.
    def _prep(self, paths):
        """Prepares a WorldView file from the archive for processing.
           Returns the path to the file ready to use.
           TODO: Apply TOA conversion!
        """
        assert isinstance(paths, str)
        (_, ext) = os.path.splitext(paths)
        tif_name = None

        if ext == '.zip': # Need to unpack

            zip_file = zipfile.ZipFile(paths, 'r')
            tif_names = list(filter(lambda x: x.lower().endswith('.tif'), zip_file.namelist()))
            assert len(tif_names) > 0, f'Error: no tif files in the file {paths}'
            assert len(tif_names) == 1, f'Error: too many tif files in {paths}: {tif_names}'
            tif_name = tif_names[0]

            self._set_info_from_tif_name(tif_name)

            (tif_path, imd_path) = self._unpack(paths)

        if ext == '.tif': # Already unpacked

            # Both files should be present in the same folder
            tif_name = paths
            unpack_folder = os.path.dirname(paths)
            (tif_path, imd_path) = get_files_from_unpack_folder(unpack_folder)

            if not (imd_path and tif_path):
                raise Exception('vendor_metadata not found in %s.' % (paths))
            self._set_info_from_tif_name(tif_name)

        assert tif_name is not None, f'Error: Unsupported extension {ext}'

        self._meta_path = imd_path
        self.__parse_meta_file(imd_path)

        return [tif_path]

    def meta_path(self):
        return self._meta_path

    def __parse_meta_file(self, meta_path):
        """Parse out the needed values from the IMD or XML file"""

        if not os.path.exists(meta_path):
            raise Exception('Metadata file not found: ' + meta_path)

        # TODO: Add more tags!
        # These are all the values we want to read in
        DESIRED_TAGS = ['ABSCALFACTOR', 'EFFECTIVEBANDWIDTH']

        data = {'ABSCALFACTOR':[],
                'EFFECTIVEBANDWIDTH':[]}

        with open(meta_path, 'r') as f:
            for line in f:

                upline = line.replace(';','').upper().strip()

                if 'MEANSUNEL = ' in upline:
                    value = upline.split('=')[-1]
                    data['MEANSUNEL'] = float(value)

                if 'SATID = ' in upline:
                    value = upline.split('=')[-1].replace('"','').strip()
                    data['SATID'] = value

                # Look for the other info we want
                for tag in DESIRED_TAGS:
                    if tag in upline:

                        # Add the value to the appropriate list
                        # -> If the bands are not in order we will need to be more careful here.
                        parts = upline.split('=')
                        value = parts[1]
                        data[tag].append(float(value))

        self._meta_path = meta_path
        self._meta = data

    def scale(self):
        return self._meta['ABSCALFACTOR']
    def bandwidth(self):
        return self._meta['EFFECTIVEBANDWIDTH']

# TOA correction
#def _get_esun_value(sat_id, band):
#    """Get the ESUN value for the given satellite and band"""
#
#    VALUES = {'WV02':[1580.814, 1758.2229, 1974.2416, 1856.4104,
#                      1738.4791, 1559.4555, 1342.0695, 1069.7302, 861.2866],
#              'WV03':[1583.58, 1743.81, 1971.48, 1856.26,
#                      1749.4, 1555.11, 1343.95, 1071.98, 863.296]}
#    try:
#        return VALUES[sat_id][band]
#    except Exception as e:
#        raise Exception('No ESUN value for ' + sat_id
#                        + ', band ' + str(band)) from e

#def _get_earth_sun_distance():
#    """Returns the distance between the Earth and the Sun in AU for the given date"""
#    # TODO: Copy the calculation from the WV manuals.
#    return 1.0

# The np.where clause handles input nodata values.

def _apply_toa_radiance(data, _, bands, factors, widths):
    """Apply a top of atmosphere radiance conversion to WorldView data"""
    buf = np.zeros(data.shape, dtype=np.float32)
    for b in bands:
        f = factors[b]
        w = widths[b]
        buf[:, :, b] = np.where(data[:, :, b] > 0, (data[:, :, b] * f) / w, OUTPUT_NODATA)
    return buf

#def _apply_toa_reflectance(data, band, factor, width, sun_elevation,
#                           satellite, earth_sun_distance):
#    """Apply a top of atmosphere reflectance conversion to WorldView data"""
#    f = factor[band]
#    w = width [band]
#
#    esun    = _get_esun_value(satellite, band)
#    des2    = earth_sun_distance*earth_sun_distance
#    theta   = np.pi/2.0 - sun_elevation
#    scaling = (des2*np.pi) / (esun*math.cos(theta))
#    return np.where(data>0, ((data*f)/w)*scaling, OUTPUT_NODATA)


def toa_preprocess(image, calc_reflectance=False):
    """
    Set a WorldviewImage's preprocessing function to do worldview TOA correction.
    Using the reflectance calculation is slightly more complicated but may be more useful.
    """

    #ds = get_earth_sun_distance() # TODO: Implement this function!

    if not calc_reflectance:
        user_function = functools.partial(_apply_toa_radiance, factors=image.scale(), widths=image.bandwidth())
    else:
        raise Exception('TODO: WV reflectance calculation is not fully implemented!')

        #user_function = functools.partial(apply_toa_reflectance, factor=scale, width=bwidth,
        #                                  sun_elevation=math.radians(data['MEANSUNEL']),
        #                                  satellite=data['SATID'],
        #                                  earth_sun_distance=ds)

    image.set_preprocess(user_function)
