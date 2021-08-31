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
Provides a command line interface to the config file driven HMTFIST C++ tool
"""
import os
import sys
import argparse


#------------------------------------------------------------------------------

def check_required_data(args):
    '''Verify that all the specified input files exist on disk'''

    have_all_data = True
    required_list = [args.presoak_dir, args.delta_prediction_path,
                     args.parameter_path, args.roughness_path, args.pits_path, args.canopy_path]

    presoak_file_list = ['input_fel.tif', 'merged_bank.tif', 'merged_stream.tif',
                         'merged_srcdir.tif']
    for ps in presoak_file_list:
        full_path = os.path.join(args.presoak_dir, ps)
        required_list.append(full_path)

    for f in required_list:
        if not os.path.exists(f):
            have_all_data = False
            print('Missing required file: ' + f)

    return have_all_data

def assemble_workdir(args):
    '''Set up the working directory to run the tool and return the path to
       the config file'''

    # Create a temporary working folder
    wd = args.work_dir
    os.system('rm -rf ' + wd)
    os.mkdir(wd)

    # Most of the input files are expected to be in the same input folder,
    # so create symlinks for wherever they are to the temporary working folder
    new_srcdir_path = os.path.join(wd, 'srcdir.tif')
    new_delta_path = os.path.join(wd, 'delta.tif')
    new_bank_path = os.path.join(wd, 'bank.tif')
    new_cost_path = os.path.join(wd, 'cost.tif')
    new_pits_path = os.path.join(wd, 'pits.tif')
    new_canopy_path = os.path.join(wd, 'canopy.tif')
    new_roughness_path = os.path.join(wd, 'roughness.tif')
    new_fel_path = os.path.join(wd, 'fel.tif')
    new_stream_path = os.path.join(wd, 'stream.csv')
    new_parameter_path = os.path.join(wd, 'parameters.csv')

    # This file is not always present
    old_stream_path = os.path.join(args.presoak_dir, 'merged_stream.csv')
    if not os.path.exists(old_stream_path):
        old_stream_path = os.path.join(args.presoak_dir, '1_stream.csv')

    os.symlink(os.path.join(args.presoak_dir, 'merged_srcdir.tif'), new_srcdir_path)
    os.symlink(args.delta_prediction_path, new_delta_path)
    os.symlink(os.path.join(args.presoak_dir, 'merged_bank.tif'), new_bank_path)
    os.symlink(os.path.join(args.presoak_dir, 'merged_cost.tif'), new_cost_path)
    os.symlink(args.pits_path, new_pits_path)
    os.symlink(args.canopy_path, new_canopy_path)
    os.symlink(args.roughness_path, new_roughness_path)
    os.symlink(os.path.join(args.presoak_dir, 'input_fel.tif'), new_fel_path)
    os.symlink(old_stream_path, new_stream_path)
    os.symlink(args.parameter_path, new_parameter_path)

    # Add a config file with all the names and paths
    config_path = os.path.join(wd, 'config.csv')
    with open(config_path, 'w') as f:
        f.write(wd + '/\n')
        #f.write(new_srcdir_path +'/\n') # TODO source direction layer
        #f.write(args.delta_prediction_path + '\n')
        for n in [new_srcdir_path, new_delta_path, new_bank_path, new_cost_path, new_pits_path, new_canopy_path,
                  new_roughness_path, new_fel_path, new_parameter_path, new_stream_path]:
            f.write(os.path.basename(n) + '\n')
        f.write(args.output_dir + '/\n')
        f.write('output_prediction.tif\n')

    return config_path


def main(argsIn):

    try:

        usage  = "usage: HMTFIST_caller [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--work-dir", required=True,
                            help="Folder containing the input image files")

        parser.add_argument("--presoak-dir", required=True,
                            help="Folder containing the outputs from the presoak tool")

        parser.add_argument("--output-dir", required=True,
                            help="Folder to write output files to")

        parser.add_argument("--delta-prediction-path", required=True,
                            help="Path to the DELTA prediction output")

        parser.add_argument("--parameter-path", required=True,
                            help="Path to a HMTFIST parameter file")

        parser.add_argument("--roughness-path", required=True,
                            help="Path to the roughness file")

        parser.add_argument("--pits-path", required=True,
                            help="Path to the pits file")

        parser.add_argument("--canopy-path", required=True,
                            help="Path to the tree canopy")

        parser.add_argument("--exe-path", default='hmtfist',
                            help="Path to the executable")

        parser.add_argument("--delete-workdir", action="store_true", default=False,
                            help="Delete the working directory after running")


        args = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    # Make sure the required files are present
    if not check_required_data(args):
        return 1

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #TODO: Identify if output files are already present and skip running the tool

    # Run the tool
    config_path = assemble_workdir(args)
    cmd = args.exe_path + ' ' + config_path
    print(cmd)
    os.system(cmd)

    # Clean up
    if args.delete_workdir:
        os.system('rm -rf ' + args.work_dir)

    output_prediction_path = os.path.join(args.output_dir, 'output_prediction.tif')
    return os.path.exists(output_prediction_path)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
