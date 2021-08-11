#!/usr/bin/env python
# This script classifies all tiff images in a directory, preserving the
# directory structure. It also copies any .txt files in the input directory
# to the output. Images that have already been classified are skipped.

import os
import sys
import subprocess
import shutil
import argparse


def main(argsIn):

    try:
        usage  = "usage: classify_directory [options] <presoak arguments>"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-dir,-i", required=True,
                            help="Directory containing input images")
        parser.add_argument("--output-dir,-o", required=True,
                            help="Directory containing output images")

        parser.add_argument("--fist-data-dir,-f", required=True,
                            help="Directory containing input images")

        parser.add_argument("--delta-model,-m", required=True,
                            help="Model file for DELTA")

        parser.add_argument("--delta-config,-g", default=None,
                            help="Config file for DELTA")

        parser.add_argument("--hmt-params,-p", required=True,
                            help="Path to HMTFIST parameter file")

        # TODO: Roughness, pits, canopy paths?

        parser.add_argument("--keep-workdir", action="store_true", default=False,
                            help="Don't delete the working directory after running")

        args, unknown_args = parser.parse_known_args()

    except argparse.ArgumentError:
        print(usage)
        return -1


    PREFIX = 'IF_'
    target_paths = []
    for r, d, f in os.walk(args.input_dir):
        for file in f:
            input_path = os.path.join(r, file)
            # Copy text files to the output folder
            if file.endswith('.txt'):               
                relative_path = os.path.relpath(os.path.join(r, file), args.input_dir)
                output_path = os.path.join(args.output_dir, relative_path)
                if not os.path.exists(output_path):
                    shutil.copy(input_path, output_path)
            # Get a list of tiff files that must be processed
            if file.endswith('.tiff') or file.endswith('.tif'):
                relative_path = os.path.relpath(os.path.join(r, PREFIX + file), args.input_dir)
                output_path = os.path.join(args.output_dir, relative_path)
                if not os.path.exists(output_path):
                    target_paths.append((input_path,  + output_path))

    if not target_paths:
        print('No unprocessed input files detected.')
        return 0

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    input_dem_path = os.path.join(args.fist_data_dir, 'fel.vrtd/orig.vrt')
    roughness_path = os.path.join(args.output_dir, 'roughness_map.tif')
    #cmd = ['gdaldem', 'roughness', input_dem_path, roughness_path]
    # TODO: When is this done?

    for (input_path, output_path) in target_paths:

        output_folder = os.path.dirname(output_path)
        # DELTA
        delta_output_folder = os.path.join(output_folder, 'delta')
        cmd = ['delta', 'classify', '--image', input_path,
               '--outdir', delta_output_folder, '--outprefix', PREFIX,
               '--prob', '--overlap', '32', args.delta_model]
        if args.delta_config:
            cmd += ['--config', args.delta_config]
        result = subprocess.run(cmd, capture_output=True)
        #for line in result.stdout.decode('ascii').split(os.linesep):


        # presoak
        presoak_output_folder = os.path.join(output_folder, 'presoak')
        cmd = ['presoak', '--max-cost', '20',
               '--elevation', input_dem_path,
               '--flow', os.path.join(args.fist_data_dir, 'p.vrtd/arcgis.vrt'),
               '--accumulation', os.path.join(args.fist_data_dir, 'ad8.vrtd/arcgis.vrt'),
               '--image', input_path,  '--output_dir', presoak_output_folder]
        cmd += unknown_args
        result = subprocess.run(cmd, capture_output=True)

        #HMTFIST
        hmtfist_work_folder = os.path.join(output_folder, 'hmtfist_work')
        hmtfist_output_folder = os.path.join(output_folder, 'hmtfist')
        cmd = ['HMTFIST_caller',#  '--exe-path', TODO,
               '--work-dir', hmtfist_work_folder, #'--delete-workdir',
               '--presoak-dir', presoak_output_folder,
               '--delta-prediction-path', output_path,
               '--pits-path', TODO,
               '--roughness-path', TODO,
               '--canopy-path', TODO,
               '--parameter-path', args.hmt_params,
               '--output-dir', hmtfist_output_folder]
        result = subprocess.run(cmd, capture_output=True)
