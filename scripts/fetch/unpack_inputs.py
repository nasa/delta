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
Try to unpack compressed input images to an output folder
"""
import os
import sys
import argparse
import traceback
from delta.extensions.sources import worldview
from delta.extensions.sources import sentinel1
from delta.extensions.sources import tiff


#------------------------------------------------------------------------------

def main(argsIn):

    SUPPORTED_IMAGE_TYPES = ['worldview', 'sentinel1']

    try:

        usage  = "usage: unpack_inputs [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-folder", dest="input_folder", required=True,
                            help="Folder containing the input image files.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Unpack images to this folder.")

        parser.add_argument("--image-type", dest="image_type", default='worldview',
                            help="Type of image files: " +
                            ', '.join(SUPPORTED_IMAGE_TYPES))

        parser.add_argument("--image-ext", dest="image_extension", default='.zip',
                            help="Extension for image files.")

        parser.add_argument("--delete-inputs", action="store_true",
                            dest="delete_inputs", default=False,
                            help="Delete input files after unpacking.")

        parser.add_argument("--image-limit", dest="image_limit",
                            default=None, type=int,
                            help="Stop after unpacking this many images.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    if options.image_type not in SUPPORTED_IMAGE_TYPES:
        print('Input image type is not supported!')
        return -1

    # Recursively find image files, obtaining the full path for each file.
    input_image_list = [os.path.join(root, name)
                        for root, dirs, files in os.walk(options.input_folder)
                        for name in files
                        if name.endswith((options.image_extension))]

    print('Found ' + str(len(input_image_list)) + ' image files.')

    # Try to load each file and record the ones that fail
    failed_files = []
    count = 0
    for image_path in input_image_list:

        try:

            if count % 10 == 0:
                print('Progress = ' + str(count) + ' out of ' + str(len(input_image_list)))

            if options.image_limit and (count >= options.image_limit):
                print('Stopping because we hit the image limit.')
                break
            count += 1

            # Mirror the input folder structure in the output folder
            image_name    = os.path.basename(os.path.splitext(image_path)[0])
            image_folder  = os.path.dirname(image_path)
            relative_path = os.path.relpath(image_folder, options.input_folder)
            this_output_folder = os.path.join(options.output_folder,
                                              relative_path, image_name)

            # TODO: Synch up the unpack functions
            tif_path = None
            if not os.path.exists(this_output_folder):
                print('Unpacking input file: ' + image_path)
                if options.image_type == 'worldview':
                    tif_path = worldview.unpack_wv_to_folder(image_path, this_output_folder)[0]
                else: # sentinel1
                    tif_path = sentinel1.unpack_s1_to_folder(image_path, this_output_folder)

            else: # The folder was already unpacked (at least partially)
                if options.image_type == 'worldview':
                    tif_path = worldview.get_files_from_unpack_folder(this_output_folder)[0]
                else: # sentinel1
                    tif_path = sentinel1.unpack_s1_to_folder(image_path, this_output_folder)

            # Make sure the unpacked image loads properly
            test_image = tiff.TiffImage(tif_path) #pylint: disable=W0612

            if options.delete_inputs:
                print('Deleting input file: ' + image_path)
                os.remove(image_path)

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
        print('No files failed to unpack!')

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
