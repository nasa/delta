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
Provides a convenient command line interface to the somewhat strict,
config file driven HMTFIST C++ tool.
"""
import os
import sys
import argparse


#------------------------------------------------------------------------------

def get_presoak_part_count(presoak_dir):
    '''Return the number of parts (image regions) in a presoak output directory'''

    max_index = 0
    files = os.listdir(presoak_dir)
    for f in files:
        parts = f.split('_')
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
            max_index = max(index, max_index)
        except:
            continue
    return max_index


def check_required_data(args):
    '''Verify that all the specified input files exist on disk'''

    have_all_data = True
    required_list = [args.presoak_dir, args.delta_prediction_path,
                     args.parameter_path, args.roughness_path, args.canopy_path]

    presoak_file_list = ['input_fel.tif', 'merged_bank.tif', 'merged_stream.tif',
                         'merged_srcdir.tif', 'input_pit.tif']
    for ps in presoak_file_list:
        full_path = os.path.join(args.presoak_dir, ps)
        required_list.append(full_path)

    for f in required_list:
        if not os.path.exists(f):
            have_all_data = False
            print('Missing required file: ' + f)

    return have_all_data


def setup_parameter_file(source_path, index, output_path):
    '''Make a copy of the source parameter file with the index updated.
       Each time we run HTMFIST it needs a parameter file with a different index
       number at the top of the file.'''
    first_line = True
    with open(source_path, 'r') as f_in:
        with open(output_path, 'w') as f_out:
            for line in f_in:
                if first_line:
                    f_out.write(str(index) + '\n')
                    first_line = False
                else:
                    f_out.write(line)

def assemble_workdir(args, index):
    '''Set up the working directory to run the tool and return the path to
       the new config file.  Each "index" will get its own set of files in the
       working directory.'''

    wd  = args.work_dir
    i_p = str(index) + '_' # Input prefix

    # Most of the HMTFIST input files are expected to be in the same input folder,
    # so create symlinks from wherever they are actually located to a temporary working
    # folder, then set up a config file that will use everything in the temporary folder.
    new_srcdir_path    = os.path.join(wd, i_p+'srcdir.tif'    )
    new_delta_path     = os.path.join(wd, i_p+'delta.tif'     )
    new_bank_path      = os.path.join(wd, i_p+'bank.tif'      )
    new_cost_path      = os.path.join(wd, i_p+'cost.tif'      )
    new_pits_path      = os.path.join(wd, i_p+'pits.tif'      )
    new_canopy_path    = os.path.join(wd, i_p+'canopy.tif'    )
    new_roughness_path = os.path.join(wd, i_p+'roughness.tif' )
    new_fel_path       = os.path.join(wd, i_p+'fel.tif'       )
    new_stream_path    = os.path.join(wd, i_p+'stream.csv'    )
    new_parameter_path = os.path.join(wd, i_p+'parameters.csv')

    os.symlink(os.path.join(args.presoak_dir, i_p+'srcdir.tif'), new_srcdir_path)
    os.symlink(os.path.join(args.presoak_dir, i_p+'bank.tif'  ), new_bank_path  )
    os.symlink(os.path.join(args.presoak_dir, i_p+'cost.tif'  ), new_cost_path  )
    os.symlink(os.path.join(args.presoak_dir, i_p+'stream.csv'), new_stream_path)
    os.symlink(os.path.join(args.presoak_dir, 'input_pit.tif' ), new_pits_path  )
    os.symlink(os.path.join(args.presoak_dir, 'input_fel.tif' ), new_fel_path   )
    os.symlink(args.delta_prediction_path, new_delta_path)
    os.symlink(args.canopy_path,           new_canopy_path)
    os.symlink(args.roughness_path,        new_roughness_path)

    setup_parameter_file(args.parameter_path, index, new_parameter_path)

    # Add a config file with all the names and paths
    config_path = os.path.join(wd, i_p + 'config.csv')
    with open(config_path, 'w') as f:
        f.write(wd + '/\n')
        for n in [new_srcdir_path, new_delta_path, new_bank_path, new_cost_path,
                  new_pits_path, new_canopy_path, new_roughness_path, new_fel_path,
                  new_parameter_path, new_stream_path]:
            f.write(os.path.basename(n) + '\n')
        f.write(args.output_dir + '/\n')
        f.write('output_prediction.tif\n')

    return config_path


def main(argsIn):

    try:
        usage  = "usage: HMTFIST_caller [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--work-dir", required=True,
                            help="Folder to use to store temporary files")

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

        parser.add_argument("--canopy-path", required=True,
                            help="Path to the tree canopy")

        parser.add_argument("--exe-path", default='hmtfist',
                            help="Path to the executable if not on $PATH")

        parser.add_argument("--delete-workdir", action="store_true", default=False,
                            help="If set, delete the working directory after running")

        args = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return 0

    # Make sure the required files are present
    if not check_required_data(args):
        return 1

    # Figure out how many different image regions the presoak tool produced
    presoak_count = get_presoak_part_count(args.presoak_dir)
    if not presoak_count:
        print('Failed to parse presoak directory: ' + args.presoak_dir)
        return 0

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #TODO: Identify if output files are already present and skip running the tool

    os.system('rm -rf ' + args.work_dir)
    os.mkdir(args.work_dir)

    # For each presoak part, perform configuration and then run HMTFIST
    for i in range(1, presoak_count):
        config_path = assemble_workdir(args, i)
        cmd = args.exe_path + ' ' + config_path
        print(cmd)
        os.system(cmd)

    # Clean up
    if args.delete_workdir:
        os.system('rm -rf ' + args.work_dir)

    # TODO: Correctly check all the output files
    output_prediction_path = os.path.join(args.output_dir, 'output_prediction.tif')
    return os.path.exists(output_prediction_path)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
