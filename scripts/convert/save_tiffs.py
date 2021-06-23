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

"""
Script to save images from a config file to tiffs, after applying preprocessing.
"""
import argparse
import os
import sys

from delta.config import config
import delta.config.modules
from delta.extensions.sources import tiff

def main(args):
    delta.config.modules.register_all()
    try:

        parser = argparse.ArgumentParser(description='Save Images from Config')
        config.setup_arg_parser(parser, ['general', 'io', 'dataset'])

        parser.add_argument("output_dir", help="Directory to save output to.")

        options = parser.parse_args(args)
    except argparse.ArgumentError:
        parser.print_help(sys.stderr)
        sys.exit(1)
    config.initialize(options)

    os.makedirs(options.output_dir, exist_ok=True)
    images = config.dataset.images()
    for (i, name) in enumerate(images):
        img = images.load(i)
        path = os.path.join(options.output_dir, os.path.splitext(os.path.basename(name))[0] + '.tiff')
        if os.path.exists(path):
            print(path + ' already exists, skipping.')
        else:
            print(name, '-->', path)
            tiff.write_tiff(path, image=img)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
