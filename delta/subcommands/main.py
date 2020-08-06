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

import sys
import argparse

from delta.config import config
import delta.config.modules
from delta.subcommands import commands

def main(args):
    delta.config.modules.register_all()
    parser = argparse.ArgumentParser(description='DELTA Machine Learning Toolkit')
    subparsers = parser.add_subparsers()

    for d in commands.SETUP_COMMANDS:
        d(subparsers)

    try:
        options = parser.parse_args(args[1:])
    except argparse.ArgumentError:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if not hasattr(options, 'function'):
        parser.print_help(sys.stderr)
        sys.exit(1)

    config.initialize(options)
    return options.function(options)
