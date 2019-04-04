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
Functions to support the Landsat satellites.
"""

import os

def allocate_bands_for_spacecraft(landsat_number):

    BAND_COUNTS = {'5':7, '7':9, '8':11}

    num_bands = BAND_COUNTS[landsat_number]
    data = dict()

    # There are fewer K constants but we store in the the
    # appropriate band indices.
    data['FILE_NAME'       ] = [''] * num_bands
    data['RADIANCE_MULT'   ] = [None] * num_bands
    data['RADIANCE_ADD'    ] = [None] * num_bands
    data['REFLECTANCE_MULT'] = [None] * num_bands
    data['REFLECTANCE_ADD' ] = [None] * num_bands
    data['K1_CONSTANT'     ] = [None] * num_bands
    data['K2_CONSTANT'     ] = [None] * num_bands

    return data

def parse_mtl_file(mtl_path):
    """Parse out the needed values from the MTL file"""

    if not os.path.exists(mtl_path):
        raise Exception('MTL file not found: ' + mtl_path)

    # These are all the values we want to read in
    DESIRED_TAGS = ['FILE_NAME', 'RADIANCE_MULT', 'RADIANCE_ADD',
                    'REFLECTANCE_MULT', 'REFLECTANCE_ADD',
                    'K1_CONSTANT', 'K2_CONSTANT']

    data = None
    with open(mtl_path, 'r') as f:
        for line in f:

            line = line.replace('"','') # Clean up

            # Get the spacecraft ID and allocate storage
            if 'SPACECRAFT_ID = LANDSAT_' in line:
                spacecraft_id = line.split('_')[-1].strip()
                data = allocate_bands_for_spacecraft(spacecraft_id)

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
                        # Landsat 7 has two thermal readings from the same wavelength bad
                        # bit with different gain settings.  Just treat the second file
                        # as another band (9).
                        name = name.replace('BAND_6_VCID_1', 'BAND_6')
                        name = name.replace('BAND_6_VCID_2', 'BAND_9')
                        band  = int(name.split('_')[-1]) -1 # One-based to zero-based
                    except ValueError: # Means this is not a proper match
                        break

                    if tag == 'FILE_NAME':
                        data[tag][band] = value # String
                    else:
                        data[tag][band] = float(value)

    return data
