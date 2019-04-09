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
Tools for loading data into the TensorFlow Dataset class.
"""
import sys
import os
import math

import numpy as np

import image_reader
import utilities


def make_landsat_list(top_folder, output_path, ext, num_regions):
    '''Write a file listing all of the files in a (recursive) folder
       matching the provided extension.
    '''

    num_entries = 0
    with open(output_path, 'w') as f:
        for root, directories, filenames in os.walk(top_folder):
            for filename in filenames:
                if os.path.splitext(filename)[1] == ext:
                    path = os.path.join(root,filename)
                    for r in num_regions:
                        f.write(path + ',' + str(r) +'\n')
                        num_entries += 1
    return num_entries

def get_roi_horiz_band_split(image_size, region, num_splits):
    """Return the ROI of an image to load given the region.
       Each region represents one horizontal band of the image.
    """

    assert region < num_splits, 'Input region ' + str(region) \
           + ' is greater than num_splits: ' + str(num_splits)

    min_x = 0
    max_x = image_size[0]

    # Fractional height here is fine
    band_height = image_size[1] / num_splits

    # TODO: Check boundary conditions!
    min_y = math.floor(band_height*region)
    max_y = math.floor(band_height*(region+1.0))

    return utilities.Rectangle(min_x, min_y, max_x, max_y)


def get_roi_tile_split(image_size, region, num_splits):
    """Return the ROI of an image to load given the region.
       Each region represents one tile in a grid split.
    """
    num_tiles = num_splits*num_splits
    assert region < num_tiles, 'Input region ' + str(region) \
           + ' is greater than num_tiles: ' + str(num_tiles)

    tile_row = math.floor(region / num_splits)
    tile_col = region % num_splits

    # Fractional sizes are fine here
    tile_width  = floor(image_size[0] / side)
    tile_height = floor(image_size[1] / side)

    # TODO: Check boundary conditions!
    min_x = math.floor(tile_width  * tile_col)
    max_x = math.floor(tile_width  * (tile_col+1.0))
    min_y = math.floor(tile_height * tile_row)
    max_y = math.floor(tile_height * (tile_row+1.0))

    return utilities.Rectangle(min_x, min_y, max_x, max_y)


def get_landsat_bands_to_use(sensor_name):
    """Return the list of one-based band indices that we are currently
       using to process the given landsat sensor.
    """

    # For now just the 30 meter bands, in original order.
    LS5_DESIRED_BANDS = [1, 2, 3, 4, 5, 6, 7]
    LS7_DESIRED_BANDS = [1, 2, 3, 4, 5, 6, 7]
    LS8_DESIRED_BANDS = [1, 2, 3, 4, 5, 6, 7, 9]

    if '5' sensor_name:
        bands = LS5_DESIRED_BANDS
        else:
            if '7' sensor_name:
                bands = LS7_DESIRED_BANDS
            else:
                if '8' sensor_name:
                    bands = LS8_DESIRED_BANDS
                else:
                    raise Exception('Unknown landsat type: ' + sensor_name)
    return bands

def prep_landsat_image(path)
    """Prepares a Landsat file from the archive for processing.
       Returns [mtl_path, band, paths, in, order, ...]
       TODO: Apply TOA conversion!
       TODO: Intelligent caching!
    """

    BASE_FOLDER = '/nobackup/smcmich1/delta/landsat' # TODO

    # Get info out of the filename
    fname  = os.path.basename(path)
    parts  = fname.split('_')
    sensor = parts[0]
    lpath  = parts[2][0:3]
    lrow   = parts[2][3:6]
    date   = parts[3]

    # Unpack the input file
    untar_folder = os.path.join(BASE_FOLDER, sensor, lpath, lrow, date)
    utilities.untar_to_folder(path, untar_folder)

    # Get the files we are interested in
    new_path = os.path.join(untar_folder, fname)

    bands = get_landsat_bands_to_use(sensor)

    # Generate all the band file names
    mtl_path     = new_path.replace('.tar.gz', '_MTL.txt')
    output_paths = [mtl_path]
    for band in bands:
        band_path = new_path.replace('.tar.gz', '_B'+str(band)+'.TIF')
        output_paths.append(band_path)

    # Check that the files exist
    for p in output_paths:
        if not os.path.exists(p):
            raise Exception('Did not find expected file: ' + p
                            + ' after unpacking tar file ' + path)

    return output_paths



def load_image_region(line, prep_function, roi_function, chunk_size, chunk_overlap, num_threads):
    """Load all image chunks for a given region of the image.
       The provided function converts the region to the image ROI.
    """

    # Our input list is stored as "path, region" strings
    parts  = line.split(',')
    path   = parts[0].strip()
    region = int(parts[1].strip())

    # Set up the input image handle
    input_paths  = prep_function(paths)
    input_reader = image_reader.MultiTiffFileReader()
    input_reader.load_images(input_paths)
    image_size = input_reader.image_size()

    # Call the provided function to get the ROI to load
    roi = roi_function(image_size, region)

    ## Until we are ready to do a larger test, just return a short vector
    #return np.array([roi.min_x, roi.min_y, roi.max_x, roi.max_y], dtype=np.int32) # DEBUG

    # Load the chunks from inside the ROI
    print('Loading chunk data from file ' + path + ' using ROI: ' + str(roi))
    chunk_data = input_reader.parallel_load_chunks(roi, chunk_size, chunk_overlap, num_threads)

    return chunk_data
