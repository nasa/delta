#!/usr/bin/env python

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

#pylint: disable=R0914

"""
Go through all of the images in a folder and verify all the images can be loaded.
"""
import os
import sys
import argparse
import traceback
import delta.config.modules
from delta.config.extensions import image_reader

# Needed for image cache to be created
delta.config.modules.register_all()


#------------------------------------------------------------------------------

def get_label_path(image_name, options):
    """Return the label file path for a given input image or throw if it is
       not found at the expected location."""

    label_name = image_name.replace(options.image_extension, options.label_extension)
    label_path = os.path.join(options.label_folder, label_name)
    if not os.path.exists(label_path):
        raise Exception('Expected label file does not exist: ' + label_path)
    return label_path

def main(argsIn):


    try:

        usage  = "usage: check_inputs [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--image-folder", dest="image_folder", required=True,
                            help="Folder containing the input image files.")

        parser.add_argument("--image-type", dest="image_type", default='worldview',
                            help="Type of image files.")

        parser.add_argument("--image-ext", dest="image_extension", default='.zip',
                            help="Extension for image files.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1


    # Recursively find image files, obtaining the full path for each file.
    input_image_list = [os.path.join(root, name)
                        for root, dirs, files in os.walk(options.image_folder)
                        for name in files
                        if name.endswith((options.image_extension))]

    print('Found ' + str(len(input_image_list)) + ' image files.')

    # Try to load each file and record the ones that fail
    failed_files = []
    for image_path in input_image_list:

        try:
            image_reader(options.image_type)(image_path)
        except Exception as e: #pylint: disable=W0703
            failed_files.append(image_path)
            print('For file: ' + image_path +
                  '\ncaught exception: ' + str(e))
            traceback.print_exc(file=sys.stdout)

    if failed_files:
        print('The following files failed: ')
        for f in failed_files:
            print(f)
    else:
        print('No files failed to load!')

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
