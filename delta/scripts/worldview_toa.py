#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __BEGIN_LICENSE__
#  Copyright (c) 2009-2013, United States Government as represented by the
#  Administrator of the National Aeronautics and Space Administration. All
#  rights reserved.
#
#  The NGT platform is licensed under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance with the
#  License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# __END_LICENSE__

"""
Script to apply Top of Atmosphere correction to WorldView 2 and 3 files.
"""
import os
import sys
import argparse
import math
import functools
import multiprocessing
import traceback
import numpy as np

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

import utilities
from image_reader import *
from image_writer import *

#------------------------------------------------------------------------------

OUTPUT_NODATA = 0.0

def get_esun_value(sat_id, band):
    """Get the ESUN value for the given satellite and band"""

    VALUES = {'WV02':[1580.814, 1758.2229, 1974.2416, 1856.4104,
                      1738.4791, 1559.4555, 1342.0695, 1069.7302, 861.2866],
              'WV03':[1583.58, 1743.81, 1971.48, 1856.26,
                      1749.4, 1555.11, 1343.95, 1071.98, 863.296]}
    try:
        return VALUES[sat_id][band]
    except Exception:
        raise Exception('No ESUN value for ' + sat_id
                        + ', band ' + str(band))

def get_earth_sun_distance():
    """Returns the distance between the Earth and the Sun in AU for the given date"""
    # TODO: Copy the calculation from the WV manuals.
    return 1.0

def apply_function_to_file(input_path, output_path, user_function, tile_size=(0,0)):
    """Apply the given function to the entire input image and write the
       result into the output path.  The function is applied to each tile of data.
    """

    print('Starting function for: ' + input_path)

    # Open the input image and get information about it
    input_paths = [input_path]
    input_reader = MultiTiffFileReader()
    input_reader.load_images(input_paths)
    (num_cols, num_rows) = input_reader.image_size()
    num_bands = input_reader.num_bands()
    #nodata_val = input_reader.nodata_value() # Not provided for Landsat.
    nodata_val = OUTPUT_NODATA
    (block_size_in, num_blocks_in) = input_reader.get_block_info(band=1)
    input_metadata = input_reader.get_all_metadata()
    #print(input_metadata)

    input_bounds = Rectangle(0, 0, width=num_cols, height=num_rows)


    X = 0 # Make indices easier to read
    Y = 1

    # Use the input tile size unless the user specified one.
    block_size_out = block_size_in
    if tile_size[X] > 0:
        block_size_out[X] = int(tile_size[X])
    if tile_size[Y] > 0:
        block_size_out[Y] = int(tile_size[Y])

    print('Using output tile size: ' + str(block_size_out))
    sys.stdout.flush()

    # Make a list of output ROIs
    num_blocks_out = (int(math.ceil(num_cols / block_size_out[X])),
                      int(math.ceil(num_rows / block_size_out[Y])))

    # Set up the output image
    writer = TiffWriter()
    writer.init_output_geotiff(output_path, num_rows, num_cols, nodata_val,
                               block_size_out[X], block_size_out[Y],
                               input_metadata,
                               utilities.get_gdal_data_type('float'),
                               num_bands)

    # Setting up output ROIs
    output_rois = []
    for r in range(0,num_blocks_out[Y]):
        for c in range(0,num_blocks_out[X]):

            # Get the ROI for the block, cropped to fit the image size.
            roi = Rectangle(c*block_size_out[X], r*block_size_out[Y],
                            width=block_size_out[X], height=block_size_out[Y])
            roi = roi.get_intersection(input_bounds)
            output_rois.append(roi)
    #print('Made ' + str(len(output_rois))+ ' output ROIs.')

    # TODO: Perform this processing in multiple threads!
    def callback_function(output_roi, read_roi, data_vec):
        """Callback function to write the first channel to the output file."""

        # Figure out the output block
        col = output_roi.min_x / block_size_out[X]
        row = output_roi.min_y / block_size_out[Y]

        # Figure out where the desired output data falls in read_roi
        x0 = output_roi.min_x - read_roi.min_x
        y0 = output_roi.min_y - read_roi.min_y
        x1 = x0 + output_roi.width()
        y1 = y0 + output_roi.height()

        # Loop on bands
        for band in range(0,num_bands):

            data = data_vec[band]

            # Crop the desired data portion and apply the user function.
            output_data = user_function(data[y0:y1, x0:x1], band)

            # Write out the result
            writer.write_geotiff_block(output_data, col, row, band)

    print('Writing TIFF blocks...')
    input_reader.process_rois(output_rois, callback_function)

    writer.finish_writing_geotiff()

    time.sleep(2)
    writer.cleanup()

    print('Done writing: ' + output_path)

    image = None # Close the image

# Cleaner ways to do this don't work with multiprocessing!
def try_catch_and_call(*args, **kwargs):
    """Wrap the previous function in a try/catch statement"""
    try:
        return apply_function_to_file(*args, **kwargs)
    except Exception as e:
        traceback.print_exc()
        sys.stdout.flush()
        return -1



def parse_meta_file(meta_path):
    """Parse out the needed values from the MTL file"""

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

# The np.where clause handles input nodata values.

def apply_toa_radiance(data, band, factor, width):
    """Apply a top of atmosphere radiance conversion to WorldView data"""
    f = factor[band]
    w = width [band]
    return np.where(data>0, (data*f)/w, OUTPUT_NODATA)

def apply_toa_reflectance(data, band, factor, width, sun_elevation,
                          satellite, earth_sun_distance):
    """Apply a top of atmosphere reflectance conversion to WorldView data"""
    f = factor[band]
    w = width [band]

    esun    = get_esun_value(satellite, band)
    des2    = earth_sun_distance*earth_sun_distance
    theta   = np.pi/2.0 - sun_elevation
    scaling = (des2*np.pi) / (esun*math.cos(theta))
    return np.where(data>0, ((data*f)/w)*scaling, OUTPUT_NODATA)


def main(argsIn):

    try:

        usage  = "usage: worldview_toa [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--image-path", dest="image_path", required=True,
                            help="Path to the image file.")

        parser.add_argument("--meta-path", dest="meta_path", required=True,
                            help="Path to the metadata file (.IMD or .xml).")

        parser.add_argument("--output-path", dest="output_path", required=True,
                            help="Where to write the output image.")

        parser.add_argument("--calc-reflectance", action="store_true",
                            dest="calc_reflectance", default=False,
                            help="Compute TOA reflectance (and temperature) instead of radiance.")

        #parser.add_argument("--num-processes", dest="num_processes", type=int, default=1,
        #                    help="Number of parallel processes to use.")

        #parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
        #                    help="Number of threads to use per process.")

        parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                            dest='tile_size', default=[256, 256], type=int,
                            help="Specify the output tile size.  Default is to keep the input tile size.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError as msg:
        raise Usage(msg)

    # Get all of the TOA coefficients and input file names
    data = parse_meta_file(options.meta_path)
    print(data)

    scale  = data['ABSCALFACTOR']
    bwidth = data['EFFECTIVEBANDWIDTH']

    ds = get_earth_sun_distance() # TODO: Implement this function!

    if options.calc_reflectance:
        user_function = functools.partial(apply_toa_reflectance, factor=scale, width=bwidth,
                                          sun_elevation=math.radians(data['MEANSUNEL']),
                                          satellite=data['SATID'],
                                          earth_sun_distance=ds)
    else:
        user_function = functools.partial(apply_toa_radiance, factor=scale, width=bwidth)

    try_catch_and_call(options.image_path, options.output_path, user_function, options.tile_size)

    print('WorldView TOA conversion is finished.')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
