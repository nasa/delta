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
import subprocess
import traceback

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

import utilities

#------------------------------------------------------------------------------


def unpack_inputs(tar_folder, unpack_folder):
    """Make sure all of the input label files are untarred.
       The unpack folder can be the same as the tar folder.
       Returns the list of label files.
    """

    if not os.path.exists(unpack_folder):
        os.mkdir(unpack_folder)

    input_list = []

    # Loop through tar files
    file_list = os.listdir(tar_folder)
    for f in file_list:
        ext = os.path.splitext(f)[1]
        if ext != '.tar':
            continue
        tar_path   = os.path.join(tar_folder, f)
        label_name = f.replace('SW.tar', 'INWM.tif')
        label_path = os.path.join(unpack_folder, label_name)
        
        # If the expected untarred file is not present, untar the file.
        if not os.path.exists(label_path):
            utilities.untar_to_folder(tar_path, unpack_folder)
            if not os.path.exists(label_path):
                raise Exception('Failed to untar label file: ' + label_path)
        file_list.append(label_path)

    return file_list


def main(argsIn):

    try:

        usage  = "usage: get_landsat_dswe_labels [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--output-path", dest="output_path", required=True,
                            help="Output image path.")

        parser.add_argument("--landsat-path", dest="landsat_path", required=True,
                            help="Path to the landsat image we want the label to match.")

        parser.add_argument("--input-folder", dest="input_folder", required=True,
                            help="Folder containing the DSWE input files.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError as msg:
        raise Usage(msg)

    
    output_folder = os.path.dirname(options.output_path)
    if output_folder and not os.path.exists(output_folder):
        os.mkdir(output_folder)


    # Untar the input files if needed
    untar_folder = options.input_folder
    input_files = unpack_inputs(options.input_folder, untar_folder)

    merge_path = options.output_path + '_merge.vrt'

    # TODO: Set the nodata value?
    
    # TODO: This won't work well if all of the label files go in one folder!
    # Merge all of the label files into a single file
    cmd = 'gdalbuildvrt ' + merge_path + ' ' + os.path.join(options.input_folder, '*INWM.tif')
    print(cmd)
    os.system(cmd)
    if not os.path.exists(merge_path):
        print('Failed to run command: ' + cmd)
        return -1

    # Get the projection and boundary of the file we want to match
    cmd = 'gdalinfo -proj4 ' + options.landsat_path
    print(cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     universal_newlines=True)
    out, err = p.communicate()
    ll_coord = None
    ur_coord = None
    proj_string  = None
    lines = out.split('\n')
    for line in lines:
        if '+proj' in line:
            proj_string = line
        if 'Lower Left' in line:
            parts = line.replace(',','').replace(')','').split()
            ll_coord = (parts[3], parts[4])
        if 'Upper Right' in line:
            parts = line.replace(',','').replace(')','').split()
            ur_coord = (parts[3], parts[4])

    # Reproject the merged label and crop to the landsat extent
    cmd = ('gdalwarp -overwrite -t_srs %s -te %s %s %s %s %s %s ' %
            (proj_string, ll_coord[0], ll_coord[1], ur_coord[0], ur_coord[1], merge_path, options.output_path))
    print(cmd)
    os.system(cmd)
    os.remove(merge_path)
    if not os.path.exists(options.output_path):
        print('Failed to run command: ' + cmd)
        return -1


    print('Landsat label file conversion is finished.')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    
    
    
