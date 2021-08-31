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

# TODO: Put in the lower level script?
def prepare_canopy_image(main_canopy_path, sample_image_path, output_canopy_path):
    '''Generate the cropped, reprojected copy of the canopy image that HMTFIST needs'''

    if is_valid_image(output_canopy_path):
        return True

    # Collect needed information from the sample image

    # Call gdalwarp
    cmd = ['gdalinfo', '-proj4', sample_image_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.decode('ascii').split(os.linesep):
        try:
            if 'Pixel Size' in line:
                parts = line.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                xres = parts[3]
                yres = parts[4]
                continue
            if '+proj' in line:
                proj_str = line.replace("'", "")
                continue
            if 'Lower Left' in line:
                parts = line.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                minX = parts[2]
                minY = parts[3]
                continue
            if 'Upper Right' in line:
                parts = line.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                maxX = parts[2]
                maxY = parts[3]
                continue

        except Exception as e:
           print('Caught exception: ' + str(e))
           return False

    cmd = ['gdalwarp', main_canopy_path, '-overwrite', '-t_srs', proj_str, '-te_srs', proj_str, '-te', minX, minY, maxX, maxY,
           '-tr', xres, yres, output_canopy_path]
    print(' '.join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #for line in result.stdout.decode('ascii').split(os.linesep):
    #    print(line)

    # Check result
    return is_valid_image(output_canopy_path)



def main(argsIn):

    try:
        usage  = "usage: classify_directory [options] <presoak arguments>"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-dir", "-i", required=True,
                            help="Directory containing input images")
        parser.add_argument("--output-dir", "-o", required=True,
                            help="Directory containing output images")

        parser.add_argument("--fist-data-dir", "-f", required=True,
                            help="Directory containing input images")
        parser.add_argument("--canopy-path", "-c", required=True,
                            help="Path to the main canopy image file")

        parser.add_argument("--delta-model", "-m", required=True,
                            help="Model file for DELTA")

        parser.add_argument("--delta-config", "-g", default=None,
                            help="Config file for DELTA")

        parser.add_argument("--unfilled-presoak-dem", "-u", required=True,
                            help="Dem to be passed to presoak as --unfilled-elevation")

        parser.add_argument("--hmt-params", "-p", required=True,
                            help="Path to HMTFIST parameter file")

        parser.add_argument("--keep-workdir", action="store_true", default=False,
                            help="Don't delete the working directory after running")

        args, unknown_args = parser.parse_known_args(argsIn)

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
        allSucceeded = True

        # DELTA
        fname = PREFIX + os.path.basename(input_path).replace('.tif','.tiff')
        delta_output_folder = os.path.join(output_folder, 'delta')
        delta_output_path = os.path.join(delta_output_folder, fname)
        if os.path.exists(delta_output_path):
            print('DELTA output already exists.')
        else:
            cmd = ['delta', 'classify', '--image', input_path,
                   '--outdir', delta_output_folder, '--outprefix', PREFIX,
                   '--prob', '--overlap', '32', args.delta_model]
            if args.delta_config:
                cmd += ['--config', args.delta_config]
            print(' '.join(cmd))
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            #for line in result.stdout.decode('ascii').split(os.linesep):
        if not is_valid_image(delta_output_path):
            print('delta processing FAILED to generate file ' + delta_output_path)
            allSucceeded = False


        # presoak
        presoak_output_folder = os.path.join(output_folder, 'presoak')
        if not os.path.exists(presoak_output_folder):
            os.mkdir(presoak_output_folder)
        presoak_output_path = os.path.join(presoak_output_folder, 'merged_cost.tif')
        if os.path.exists(presoak_output_path):
            print('presoak output already exists')
        else:
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


        #HMTFIST
        hmtfist_work_folder = os.path.join(output_folder, 'hmtfist_work')
        hmtfist_output_folder = os.path.join(output_folder, 'hmtfist')

        hmtfist_output_path = os.path.join(hmtfist_output_folder, 'output_prediction.tif')
        if is_valid_image(delta_output_path) and is_valid_image(presoak_output_path):

            # Create the required roughness input
            presoak_output_dem_path = os.path.join(presoak_output_folder, 'input_dem.tif')
            presoak_output_pit_path = os.path.join(presoak_output_folder, 'input_pit.tif')
            roughness_path = os.path.join(presoak_output_folder, 'dem_roughness.tif')
            if not os.path.exists(roughness_path):
                cmd = ['gdaldem', 'roughness', presoak_output_dem_path, roughness_path]
                print(' '.join(cmd))
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if not is_valid_image(roughness_path):
                allSucceeded = False

            this_canopy_path = os.path.join(presoak_output_folder, 'canopy_portion.tif')
            if (not prepare_canopy_image(args.canopy_path, presoak_output_pit_path, this_canopy_path)):
                allSucceeded = False

            if allSucceeded:
                this_folder = os.path.dirname(__file__)
                cmd = [os.path.join(this_folder, 'HMTFIST_caller.py'),
                       '--work-dir', hmtfist_work_folder, #'--delete-workdir',
                       '--presoak-dir', presoak_output_folder,
                       '--delta-prediction-path', delta_output_path,
                       '--pits-path', presoak_output_pit_path,
                       '--roughness-path', roughness_path,
                       '--canopy-path', this_canopy_path,
                       '--parameter-path', args.hmt_params,
                       '--output-dir', hmtfist_output_folder]
                print(' '.join(cmd))
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # TODO: What output path is this tool writing?
        if not is_valid_image(hmtfist_output_path):
            print('hmtfist processing FAILED to generate file ' + hmtfist_output_path)
            allSucceeded = False

        if allSucceeded:
            print('Completed processing file: ' + input_path)
        else:
            print('Unable to complete processing for file: ' + input_path)

        raise Exception('debug')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
