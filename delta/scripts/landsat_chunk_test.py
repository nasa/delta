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
Script to apply Top of Atmosphere correction to Landsat 5, 7, and 8 files.
"""
import sys, os
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
import landsat_utils
from image_reader import *
from image_writer import *

#------------------------------------------------------------------------------

def main(argsIn):

    try:

        # Use parser that ignores unknown options
        usage  = "usage: landsat_toa [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--mtl-path", dest="mtl_path", required=True,
                            help="Path to the MTL file in the same folder as the image band files.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Write output band files to this output folder with the same names.")

        parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
                            help="Number of threads to use per process.")

        parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                            dest='tile_size', default=[0,0], type=int,
                            help="Specify the output tile size.  Default is to keep the input tile size.")

        # This call handles all the parallel_mapproject specific options.
        options = parser.parse_args(argsIn)

        # Check the required positional arguments.

    except argparse.ArgumentError as msg:
        raise Usage(msg)

    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    # Get all of the TOA coefficients and input file names
    data = landsat_utils.parse_mtl_file(options.mtl_path)

    input_folder = os.path.dirname(options.mtl_path)

    input_paths = []
    for fname in data['FILE_NAME']:
        input_path = os.path.join(input_folder, fname)
        input_paths.append(input_path)

    # Open the input image and get information about it
    input_reader = MultiTiffFileReader()
    input_reader.load_images(input_paths)
    (num_cols, num_rows) = input_reader.image_size()
    input_bounds = Rectangle(0, 0, width=num_cols, height=num_rows)
    sys.stdout.flush()

    chunk_size = 256
    chunk_overlap = 0
    roi = Rectangle(0,0,width=num_cols,height=num_rows)
    chunk_data = input_reader.parallel_load_chunks(roi, chunk_size, chunk_overlap, options.num_threads)
    
    shape = chunk_data.shape
    num_chunks = shape[0]
    num_bands = shape[1]
    print('num_chunks = ' + str(num_chunks))
    print('num_bands = ' + str(num_bands))
    
    band = 3 # Write this band out to debug.
    for chunk in range(0,num_chunks):
        data = chunk_data[chunk,band,:,:]
        #print('data.shape = ' + str(data.shape))

        # Dump to disk
        output_path = os.path.join(options.output_folder, 'chunk_'+str(chunk) + '.tif')
        write_simple_image(output_path, data, data_type='uint16')

        #raise Exception('DEBUG')

    print('Landsat chunker is finished.')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    
    
    
