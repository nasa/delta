"""
Functions to support the WorldView satellites.
"""

import os

from delta.config import config
from delta.imagery import utilities
from . import basic_sources

def parse_meta_file(meta_path):
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

            #line = line.replace('"','') # Clean up
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

    return data


def get_worldview_bands_to_use(sensor_name):
    """Return the list of zero-based band indices that we are currently
       using to process the given WorldView sensor.
    """

    WV2_DESIRED_BANDS = [0, 1, 2, 3, 4, 5, 6, 7]
    WV3_DESIRED_BANDS = [0, 1, 2, 3, 4, 5, 6, 7] # TODO: More bands?

    if '2' in sensor_name:
        bands = WV2_DESIRED_BANDS
    else:
        if '3' in sensor_name:
            bands = WV3_DESIRED_BANDS
        else:
            raise Exception('Unknown WorldView type: ' + sensor_name)
    return bands


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
        if os.path.splitext(f)[1] == '.IMD':
            imd_path = os.path.join(vendor_folder, f)
            break
    for f in main_files:
        if os.path.splitext(f)[1] == '.tif':
            tif_path = os.path.join(folder, f)
            break
    return (tif_path, imd_path)

def get_scene_info(path):
    """Extract information about the landsat scene from the file name"""
    # ex: WV02N42_939570W073_2520792013040400000000MS00_GU004003002.zip
    fname  = os.path.basename(path)
    parts  = fname.split('_')
    output = {}
    output['sensor'] = parts[0][0:4]
    output['date'  ] = parts[2][6:14]
    return output

class WorldviewImage(basic_sources.TiffImage):
    """Compressed WorldView image tensorflow dataset wrapper (see imagery_dataset.py)"""

    # This function is currently set up for the HDDS archived WV data, files from other
    #  locations will need to be handled differently.
    def prep(self):
        """Prepares a WorldView file from the archive for processing.
           Returns the path to the file ready to use.
           TODO: Apply TOA conversion!
        """
        scene_info = get_scene_info(self.path)

        # Get the folder where this will be stored from the cache manager
        name = '_'.join([scene_info['sensor'], scene_info['date']])
        unpack_folder = config.cache_manager().register_item(name)

        # Check if we already unpacked this data
        (tif_path, imd_path) = get_files_from_unpack_folder(unpack_folder)

        if imd_path and tif_path:
            #print('Already have unpacked files in ' + unpack_folder)
            pass
        else:
            print('Unpacking file ' + self.path + ' to folder ' + unpack_folder)
            utilities.unpack_to_folder(self.path, unpack_folder)
            (tif_path, imd_path) = get_files_from_unpack_folder(unpack_folder)

        return [tif_path]
