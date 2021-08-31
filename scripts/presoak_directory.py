#!/usr/bin/env python3
# This script classifies all tiff images in a directory, preserving the
# directory structure. It also copies any .txt files in the input directory
# to the output. Images that have already been classified are skipped.

import os
import sys
import subprocess
import shutil
import argparse


def is_valid_image(image_path):
    '''Return True if the given path is a valid image on disk'''
    if (not os.path.exists(image_path)) or (os.path.getsize(image_path) == 0):
        return False
    cmd = ['gdalinfo', image_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.decode('ascii').split(os.linesep):
        if 'Size is' in line:
            parts = line.replace(',', ' ').split()
            if len(parts) < 4:
                print('Unexpected size length!')
                return False
            try:
                i1 = int(parts[2])
                i2 = int(parts[3])
                return (i1 > 0) and (i2 > 0)
            except Exception as e:
                print('Caught exception: ' + str(e))
                return False
    print('Did not find size line!')
    return False

def main(argsIn):

    try:
        usage  = "usage: presoak_directory [options] <presoak arguments>"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-dir", "-i", required=True,
                            help="Directory containing input images")
        parser.add_argument("--output-dir", "-o", required=True,
                            help="Directory containing output images")

        parser.add_argument("--fist-data-dir", "-f", required=True,
                            help="Directory containing input images")

        parser.add_argument("--unfilled-presoak-dem", "-u", required=True,
                            help="Dem to be passed to presoak as --unfilled-elevation")


        args, unknown_args = parser.parse_known_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1



    target_paths = []
    for r, d, f in os.walk(args.input_dir):
        for file in f:
            input_path = os.path.join(r, file)
            # Get a list of tiff files that must be processed
            if file.endswith('.tiff') or file.endswith('.tif'):
                relative_folder = os.path.relpath(r, args.input_dir)
                output_folder = os.path.join(args.output_dir, relative_folder)
                target_paths.append((input_path, output_folder))

    if not target_paths:
        print('No unprocessed input files detected.')
        return 0
    print('Found ' + str(len(target_paths)) + ' input files to process.')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    for (input_path, output_folder) in target_paths:

        print('Starting processing for file: ' + input_path)
        print(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        allSucceeded = True

        # presoak
        presoak_output_folder = output_folder
        if not os.path.exists(presoak_output_folder):
            os.mkdir(presoak_output_folder)
        presoak_output_path = os.path.join(presoak_output_folder, 'merged_cost.tif')
        if is_valid_image(presoak_output_path):
            print('Skipping completed image: ' + presoak_output_path)
            continue
        cmd = ['presoak', '--max_cost', '20',
               '--elevation', os.path.join(args.fist_data_dir, 'fel.vrtd/orig.vrt'),
               '--unfilled_elevation', args.unfilled_presoak_dem,
               '--flow', os.path.join(args.fist_data_dir, 'p.vrtd/arcgis.vrt'),
               '--accumulation', os.path.join(args.fist_data_dir, 'ad8.vrtd/arcgis.vrt'),
               '--image', input_path,  '--output_dir', presoak_output_folder]
        cmd += unknown_args
        print(' '.join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if not is_valid_image(presoak_output_path):
            print('presoak processing FAILED to generate file ' + presoak_output_path)
            allSucceeded = False
        # To save space, delete all other outputs
        all_outputs = os.listdir(presoak_output_folder)
        for o in all_outputs:
            if o != 'merged_cost.tif':
                os.remove(os.path.join(presoak_output_folder, o))

        #raise Exception('DEBUG!')

        if allSucceeded:
            print('Completed processing file: ' + input_path)
        else:
            print('Unable to complete processing for file: ' + input_path)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
