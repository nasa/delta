"""
Functions to support the WorldView satellites.
"""

import os

from delta.config import config
from delta.imagery import utilities
from . import tiff

def _get_files_from_unpack_folder(folder):
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
        if os.path.splitext(f)[1] == '.IMD':
            imd_path = os.path.join(vendor_folder, f)
            break
    for f in main_files:
        if os.path.splitext(f)[1] == '.tif':
            tif_path = os.path.join(folder, f)
            break
    return (tif_path, imd_path)

class WorldviewImage(tiff.TiffImage):
    """Compressed WorldView image tensorflow dataset wrapper (see imagery_dataset.py)"""
    def __init__(self, paths):
        super(WorldviewImage, self).__init__(paths)
        self._meta_path = None
        self._meta = None

    def _unpack(self, paths):
        # Get the folder where this will be stored from the cache manager
        name = '_'.join([self._sensor, self._date])
        unpack_folder = config.cache_manager().register_item(name)

        # Check if we already unpacked this data
        (tif_path, imd_path) = _get_files_from_unpack_folder(unpack_folder)

        if imd_path and tif_path:
            #print('Already have unpacked files in ' + unpack_folder)
            pass
        else:
            print('Unpacking file ' + paths + ' to folder ' + unpack_folder)
            utilities.unpack_to_folder(paths, unpack_folder)
            (tif_path, imd_path) = _get_files_from_unpack_folder(unpack_folder)
        return (tif_path, imd_path)

    # This function is currently set up for the HDDS archived WV data, files from other
    #  locations will need to be handled differently.
    def _prep(self, paths):
        """Prepares a WorldView file from the archive for processing.
           Returns the path to the file ready to use.
           TODO: Apply TOA conversion!
        """
        assert isinstance(paths, str)
        parts = os.path.basename(paths).split('_')
        self._sensor = parts[0][0:4]
        self._date = parts[2][6:14]

        (tif_path, imd_path) = self._unpack(paths)

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
