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
import os
import math

import numpy as np

import image_reader
import utilities

# TODO: Generalize
def make_landsat_list(top_folder, output_path, ext, num_regions):
    '''Write a file listing all of the files in a (recursive) folder
       matching the provided extension.
    '''

    num_entries = 0
    with open(output_path, 'w') as f:
        for root, directories, filenames in os.walk(top_folder):
            for filename in filenames:
                if os.path.splitext(filename)[1] == ext:
                    path = os.path.join(root, filename)
                    for r in range(0, num_regions):
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



def load_image_region(line, prep_function, roi_function, chunk_size, chunk_overlap, num_threads):
    """Load all image chunks for a given region of the image.
       The provided function converts the region to the image ROI.
    """

    # Our input list is stored as "path, region" strings
    line   = line.decode() # Convert from TF to string type
    parts  = line.split(',')
    path   = parts[0].strip()
    region = int(parts[1].strip())

    # Set up the input image handle
    input_paths  = prep_function(path)
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

    print('Chunk data shape = ' + str(chunk_data.shape))
    return chunk_data

def load_fake_labels(line, prep_function, roi_function, chunk_size, chunk_overlap):
    """Use to generate fake label data for load_image_region"""

    # Our input list is stored as "path, region" strings
    #print('Label data input = ' + str(line))
    line   = line.decode() # Convert from TF format to string
    parts  = line.split(',')
    path   = parts[0].strip()
    region = int(parts[1].strip())

    # Set up the input image handle
    input_paths  = prep_function(path)
    input_reader = image_reader.MultiTiffFileReader()
    input_reader.load_images([input_paths[0]]) # Just the first band
    image_size = input_reader.image_size()

    # Call the provided function to get the ROI to load
    roi = roi_function(image_size, region)

    #return np.array([0, 1, 2, 3], dtype=np.int32) # DEBUG

    # Load the chunks from inside the ROI
    chunk_data = input_reader.parallel_load_chunks(roi, chunk_size, chunk_overlap)

    # Make a fake label
    full_shape = chunk_data.shape[0]
    print('label shape in = ' + str(chunk_data.shape))
    chunk_data = np.zeros(full_shape, dtype=np.int32)
    print('label shape out = ' + str(chunk_data.shape))
    chunk_data[ 0:10] = 1 # Junk labels
    chunk_data[10:20] = 2
    return chunk_data


# TODO: Not currently used, but could be if the TF method of filtering chunks is inefficient.
def parallel_filter_chunks(data, num_threads):
    """Filter out chunks that contain the Landsat nodata value (zero)"""

    (num_chunks, num_bands, width, height) = data.shape()
    num_chunk_pixels = width * height

    print('Num input chunks = ' + str(num_chunks))

    valid_chunks = [True] * num_chunks
    splits = []
    thread_size = float(num_chunks) / float(num_threads)
    for i in range(0,num_threads):
        start_index = math.floor(i    *thread_size)
        stop_index  = math.floor((i+1)*thread_size)
        splits.append((start_index, stop_index))

    # Internal function to flag nodata chunks from the start to stop indices (non-inclusive)
    def check_chunks(pair):
        (start_index, stop_index) = pair
        for i in range(start_index, stop_index):
            chunk = data[i, 0, :, :]
            print(chunk.shape())
            print(chunk)
            if np.count_nonzero(chunk) != num_chunk_pixels:
                valid_chunks[i] = False
                print('INVALID')

    # Call check_chunks in parallel using a thread pool
    pool = ThreadPool(num_threads)
    pool.map(check_chunks, splits)
    pool.close()
    pool.join()

    # Remove the bad chunks
    valid_indices = []
    for i in range(0,num_chunks):
        if valid_chunks[i]:
            valid_indices.append(i)

    print('Num remaining chunks = ' + str(len(valid_indices)))

    return data[valid_indices, :, :, :]

