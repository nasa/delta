#!/usr/bin/env python3

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

# This tool classifies all of the images in an input directory, running presoak, DELTA,
# and HMTFIST on each input image.  The input directory structure will be mirrored in the
# output folder.
#
# Requirements to run each tool:
# - presoak: The presoak tool must be compiled and on the PATH.  In addition, the FIST
#            data directory (about 1TB) must be available.
# - DELTA: Must be installed per the normal DELTA instructions.  Must have a trained model
#          and a configuration file available.
# - HMTFIST: The hmtfist tool must be compiled and on the PATH.  Also requires the "canopy" dataset
#            FS “Analytical” TCC from  https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/
# - The gdal command line tools must also be on the PATH
#
# HMTFIST always requires the output of DELTA and presoak on order to run.  If DELTA is run with the
# "--s1-delta-presoak-augment" flag and a corresponding model file it also requires the presoak output.
# If HMTFIST is not run and DELTA does not need presoak, presoak will only be run if "--force-presoak" is set.
#
#

import os
import sys
import subprocess
import shutil
import argparse
import yaml

# HMTFIST has not been fully tested yet, so this needs to be enabled for it to run.
ENABLE_HMTFIST = False

def tif_to_tiff(name):
    '''Convert a .tif name into a .tiff name'''
    if name.endswith('.tif'):
        return name + 'f'
    return name


def is_valid_image(image_path):
    '''Return True if the given path is a valid image on disk'''

    if (not os.path.exists(image_path)) or (os.path.getsize(image_path) == 0):
        return False
    cmd = ['gdalinfo', image_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    for line in result.stdout.decode('utf-8').split(os.linesep):
        if 'Size is' in line:
            parts = line.replace(',', ' ').split()
            if len(parts) < 4:
                print('Unexpected size length!')
                return False
            try:
                i1 = int(parts[2])
                i2 = int(parts[3])
                return (i1 > 0) and (i2 > 0)
            except Exception as e: #pylint: disable=W0703
                print('Caught exception: ' + str(e))
                return False
    print('Did not find size line!')
    return False


def get_image_info(image_path):
    '''Helper function to use gdalinfo to get iformation about a geotiff image on disk'''
    REQUIRED_INFO = ['xres', 'yres', 'proj_str', 'minX', 'minY', 'maxX', 'maxY', 'height', 'width']

    result = {}
    cmd = ['gdalinfo', '-proj4', image_path]
    process_output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    for line in process_output.stdout.decode('ascii').split(os.linesep):
        try:
            if 'Pixel Size' in line:
                parts = line.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                result['xres'] = parts[3]
                result['yres'] = parts[4]
                continue
            if '+proj' in line:
                result['proj_str'] = line.replace("'", "")
                continue
            if 'Lower Left' in line:
                parts = line.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                result['minX'] = parts[2]
                result['minY'] = parts[3]
                continue
            if 'Upper Right' in line:
                parts = line.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
                result['maxX'] = parts[2]
                result['maxY'] = parts[3]
                continue
            if 'Size is' in line:
                parts = line.replace(',','').split()
                result['width'] = parts[2]
                result['height'] = parts[3]

        except Exception as e: #pylint: disable=W0703
            print('Caught exception: ' + str(e))
            return None
    for i in REQUIRED_INFO:
        if i not in result:
            print('Did not find required item ' + i + ' in gdalinfo output!')
            return None
    return result


def prepare_canopy_image(main_canopy_path, sample_image_path, output_canopy_path):
    '''Generate the cropped, reprojected copy of the canopy image that matches the
       projection/size/resolution of the sample image'''

    if is_valid_image(output_canopy_path):
        return True

    # Collect needed information from the sample image
    image_info = get_image_info(sample_image_path)
    if not image_info:
        return False

    # Call gdalwarp to generate the output image
    cmd = ['gdalwarp', main_canopy_path, '-overwrite', '-t_srs',  image_info['proj_str'],
           '-te_srs', image_info['proj_str'],
           '-te',  image_info['minX'],  image_info['minY'],  image_info['maxX'],  image_info['maxY'],
           '-tr', image_info['xres'],  image_info['yres'], output_canopy_path]
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

    return is_valid_image(output_canopy_path)

def resize_delta_output(delta_path, sample_image_path, output_path):
    '''Generate a copy of the delta output image matched in size to the sample image,
       padding with nodata as required.'''

    image_info = get_image_info(sample_image_path)
    if not image_info:
        return False

    cmd = ['gdal_translate', delta_path, output_path,
           '-projwin',  image_info['minX'],  image_info['maxY'],  image_info['maxX'],  image_info['minY'],
           '-tr', image_info['xres'],  image_info['yres']]
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

    return is_valid_image(output_path)

def add_dem_channel(source_path, dem_path, output_path):
    '''Resize the DEM to match the source path and append it as a new channel to the source image.
       This is required for DELTA networks which utilize the DEM as an additional channel.'''

    if is_valid_image(output_path):
        return True

    # Collect needed information from the sample image
    image_info = get_image_info(source_path)
    if not image_info:
        return False

    try:
        temp_path1 = output_path + '_temp1.tif'
        temp_path2 = output_path + '_temp2.tif'

        cmd = ['gdal_translate', dem_path, '-outsize', image_info['width'], image_info['height'],
               '-projwin', image_info['minX'], image_info['maxY'], image_info['maxX'], image_info['minY'],
               temp_path1]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        cmd = ['gdal_edit.py', '-unsetnodata', temp_path1]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        cmd = ['gdal_calc.py', '-A', temp_path1, '--calc=numpy.where(numpy.isinf(A), 1.05, 0.1*A)',
               '--outfile='+temp_path2]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        cmd = ['gdal_merge.py', '-a_nodata', 'nan', '-o', output_path, '-separate', source_path, temp_path2]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        os.remove(temp_path1)
        os.remove(temp_path2)

    except Exception as e: #pylint: disable=W0703
        print('Caught exception: ' + str(e))
        return False

    # Check result
    return is_valid_image(output_path)


def add_fist_cost_channel(source_path, cost_path, output_path):
    '''Resize the FIST cost image to match the source path and append it as a new channel to the source image.
       This is required for DELTA networks which utilize the FIST cost as an additional channel.
       The source image must have any DELTA preprocessing applied *before* this function is called.'''

    if is_valid_image(output_path):
        return True

    # Collect needed information from the sample image
    image_info = get_image_info(source_path)
    if not image_info:
        return False

    try:
        temp_path1 = output_path + '_temp1.tif'
        temp_path2 = output_path + '_temp2.tif'
        temp_path3 = output_path + '_temp3.tif'
        temp_path4 = output_path + '_temp4.tif'

        # Crop cost image to the source image extents
        cmd = ['gdalwarp', cost_path, '-overwrite',  '-tr', image_info['xres'],  image_info['yres'],
               '-t_srs', image_info['proj_str'],
               '-te', image_info['minX'], image_info['maxY'], image_info['maxX'], image_info['minY'],
               temp_path1]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        cmd = ['gdal_edit.py', '-unsetnodata', temp_path1]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        # Scale the presoak value (0-20 range) and apply nodata cutoff
        cmd = ['gdal_calc.py', '-A', temp_path1,
               '--calc=numpy.where(numpy.isinf(A), 1.05, 0.05*A)', '--outfile='+temp_path2]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        # Apply nodata cutoff
        cmd = ['gdal_calc.py', '-A', source_path, '--A_band=1',
               '--calc=numpy.where(A <= 0, -0.5, A)', '--outfile='+temp_path3]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
        cmd = ['gdal_calc.py', '-A', source_path, '--A_band=2',
               '--calc=numpy.where(A <= 0, -0.5, A)', '--outfile='+temp_path4]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        # Pack into a three channel image
        cmd = ['gdal_merge.py', '-a_nodata', '-0.5',  '-o', output_path,
               '-separate', temp_path3, temp_path4, temp_path2]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)

        os.remove(temp_path1)
        os.remove(temp_path2)
        os.remove(temp_path3)
        os.remove(temp_path4)

    except Exception as e: #pylint: disable=W0703
        print('Caught exception: ' + str(e))
        return False

    # Check result
    return is_valid_image(output_path)

def call_presoak(args, input_path, output_folder, unknown_args):
    '''Call the presoak tool'''

    if not args.fist_data_dir:
        print('Missing required input argument --fist-data-dir to run presoak!')
        return (False, None, None, None)
    if not args.unfilled_presoak_dem:
        print('Missing required input argument --unfilled-presoak-dem to run presoak!')
        return (False, None, None, None)

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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if not is_valid_image(presoak_output_path):
        for line in result.stdout.decode('ascii').split(os.linesep):
            print(line)
        print('presoak processing FAILED to generate file ' + presoak_output_path)
        return (False, None, None, None)

    presoak_output_cost_path = os.path.join(presoak_output_folder, 'merged_cost.tif')
    presoak_output_dem_path = os.path.join(presoak_output_folder, 'input_dem.tif')
    return (True, presoak_output_folder, presoak_output_cost_path, presoak_output_dem_path)


def set_value_in_dict_recursive(d, name, newValue):
    '''Find the named field in the dictionary d and set it to the given value'''

    index = None
    if isinstance(d, dict):
        for k, v in d.items():
            if k == name:
                index = name
                continue
            if isinstance(v, (dict, list)):
                set_value_in_dict_recursive(v, name, newValue)
    else:
        for i, v in enumerate(d):
            if v == name:
                index = i
                continue
            if isinstance(v, (dict, list)):
                set_value_in_dict_recursive(v, name, newValue)
    if index is not None:
        d[index] = newValue


def make_no_preprocess_config(input_path, output_path):
    '''Generate version of config file with preprocess steps stripped out'''
    with open(input_path) as f:
        config_yaml = yaml.safe_load(f)
        set_value_in_dict_recursive(config_yaml, 'preprocess', None)
        text = yaml.dump(config_yaml)
        with open(output_path, 'w') as f:
            f.write(text)


def call_delta(args, input_path, output_folder, input_name,
               presoak_succeeded, presoak_output_cost_path):
    '''Run the DELTA tool'''

    delta_output_folder = os.path.join(output_folder, 'delta')
    delta_config_to_use = args.delta_config

    no_preprocess_config = None
    delta_input_image = input_path
    model_to_use = args.delta_model
    if (args.sensor == 'sentinel1') and args.s1_delta_presoak_augment_model:
        # Augment the input Sentinel1 image with a presoak cost channel before running DELTA
        if not presoak_succeeded:
            print('presoak failed, unable to use augmented DELTA model')
        else:

            # Apply preprocessing to the input image before we merge with the presoak cost
            preprocessed_input = os.path.join(delta_output_folder, tif_to_tiff(os.path.basename(input_path)))
            this_folder = os.path.dirname(__file__)
            cmd = [os.path.join(this_folder, 'convert/save_tiffs.py'), '--image', input_path,
                   '--image-type', 'sentinel1', '--config', args.delta_config, delta_output_folder]
            print(' '.join(cmd))
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            if not is_valid_image(preprocessed_input):
                raise Exception('Failed to run preprocessing on image: ' + input_path)

            # Add FIST cost as a channel to the input image
            merged_path = os.path.join(delta_output_folder, 'presoak_merged_input_image.tif')
            if not add_fist_cost_channel(preprocessed_input, presoak_output_cost_path, merged_path):
                print('Failed to add channel, unable to use augmented DELTA model')
            else:
                print('Using presoak augmented model file: ' + args.s1_delta_presoak_augment_model)
                model_to_use = args.s1_delta_presoak_augment_model
                delta_input_image = merged_path
                os.remove(preprocessed_input)

                # Generate version of config file with preprocess steps stripped out
                no_preprocess_config = os.path.join(delta_output_folder, 'delta_no_preprocess_config.yaml')
                delta_config_to_use = no_preprocess_config
                make_no_preprocess_config(args.delta_config, no_preprocess_config)

    PREFIX = 'IF_' # This is required by DELTA, but we will remove on output
    fname_in = PREFIX + os.path.basename(delta_input_image)

    delta_output_path_in = os.path.join(delta_output_folder, fname_in)
    delta_output_path = os.path.join(delta_output_folder, input_name + '.tiff')
    delta_output_path_in = tif_to_tiff(delta_output_path_in) # DELTA writes with .tiff extension
    if os.path.exists(delta_output_path):
        print('DELTA output already exists.')
    else:
        cmd = ['delta', 'classify', '--image', delta_input_image,
               '--outdir', delta_output_folder, '--outprefix', PREFIX,
               '--prob', model_to_use, '--config', delta_config_to_use]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        shutil.move(delta_output_path_in, delta_output_path)
        if no_preprocess_config:
            os.remove(no_preprocess_config)
    if not is_valid_image(delta_output_path):
        print('delta processing FAILED to generate file ' + delta_output_path)
        return (False, None, None)

    return (True, delta_output_folder, delta_output_path)


def call_hmtfist(args, presoak_output_folder, presoak_output_dem_path,
                 delta_output_folder, delta_output_path, output_folder):
    '''Run the HMTFIST tool'''

    hmtfist_work_folder = os.path.join(output_folder, 'hmtfist_work')
    hmtfist_output_folder = os.path.join(output_folder, 'hmtfist')

    hmtfist_output_path = os.path.join(hmtfist_output_folder, 'output_prediction.tif')

    # We need to create a roughness version of the presoak DEM output file
    roughness_path = os.path.join(presoak_output_folder, 'dem_roughness.tif')
    if not os.path.exists(roughness_path):
        cmd = ['gdaldem', 'roughness', presoak_output_dem_path, roughness_path]
        print(' '.join(cmd))
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if not is_valid_image(roughness_path):
        print('Failed to generate roughness image!')
        return False

    # We need to extract a matching portion of the canopy input image
    this_canopy_path = os.path.join(presoak_output_folder, 'canopy_portion.tif')
    if not prepare_canopy_image(args.canopy_path, presoak_output_dem_path, this_canopy_path):
        print('Failed to generate canopy image!')
        return False

    # We need to resize the DELTA output to match the presoak output size
    delta_resize_path = os.path.join(delta_output_folder, 'presoak_size_match.tif')
    if not resize_delta_output(delta_output_path, presoak_output_dem_path, delta_resize_path):
        print('Failed to resize DELTA output!')
        return False

    this_folder = os.path.dirname(__file__)
    # Another python sub-script handles the details of running this program
    cmd = [os.path.join(this_folder, 'HMTFIST_caller.py'),
           '--work-dir', hmtfist_work_folder,
           '--presoak-dir', presoak_output_folder,
           '--delta-prediction-path', delta_resize_path,
           '--roughness-path', roughness_path,
           '--canopy-path', this_canopy_path,
           '--parameter-path', args.hmt_params,
           '--output-dir', hmtfist_output_folder]
    if not args.keep_workdir:
        cmd.append('--delete-workdir')
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

    # TODO: This output check needs to be worked out.  There may be multiple output files.
    if not is_valid_image(hmtfist_output_path):
        print('hmtfist processing FAILED to generate file ' + hmtfist_output_path)
        return False
    return True

def unpack_input_image(input_path, args, output_path):
    '''Unpack a single input image'''

    if is_valid_image(output_path):
        print('Already have unpacked image: ' + output_path)
        return True

    # Run a conversion tool with preprocess instructions stripped so it just unpacks
    no_preprocess_config = output_path + '_no_preprocess_config.yaml'
    make_no_preprocess_config(args.delta_config, no_preprocess_config)

    output_folder = os.path.dirname(output_path)
    output_name = os.path.splitext(os.path.basename(input_path))[0] + '.tiff'
    unpacked_input = os.path.join(output_folder, output_name)
    this_folder = os.path.dirname(__file__)
    cmd = [os.path.join(this_folder, 'convert/save_tiffs.py'), '--image', input_path,
           '--image-type', args.sensor, '--config', no_preprocess_config, output_folder]
    print(' '.join(cmd))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if not is_valid_image(unpacked_input):
        raise Exception('Failed to unpack image: ' + input_path)

    shutil.move(unpacked_input, output_path)
    os.remove(no_preprocess_config)
    return True

def find_targets(args):
    '''Find input files and choose output paths'''

    INPUT_FILE_TYPES = ['.tif', '.tiff', '.zip']

    if not os.path.exists(args.input_dir):
        print('Error: Input folder does not exist: ' + args.input_dir)
        return []

    target_paths = []
    for r, _, f in os.walk(args.input_dir):
        for file in f:
            input_path = os.path.join(r, file)
            # Copy text files to the output folder
            if file.endswith('.txt'):
                relative_path = os.path.relpath(os.path.join(r, file), args.input_dir)
                output_path = os.path.join(args.output_dir, relative_path)
                if not os.path.exists(output_path):
                    shutil.copy(input_path, output_path)
            # Get a list of tiff files that must be processed
            is_match = False
            for t in INPUT_FILE_TYPES:
                if file.endswith(t):
                    is_match = True
            if is_match:
                relative_folder = os.path.relpath(r, args.input_dir)
                if relative_folder == '.': # Create subdirectories for input images in the same folder
                    relative_folder = os.path.splitext(file)[0]
                output_folder = os.path.join(args.output_dir, relative_folder)
                target_paths.append((input_path, output_folder))
    return target_paths


def classify_image(input_path, output_folder, input_name, args, unknown_args):
    '''Run all classification steps on a single image'''

    all_succeeded = True

    # presoak
    # - Only run presoak if another tool requires it or if --force-presoak was set
    if (args.force_presoak or ENABLE_HMTFIST or
            ((args.sensor == 'sentinel1') and args.s1_delta_presoak_augment_model)):

        presoak_succeeded, presoak_output_folder, presoak_output_cost_path, presoak_output_dem_path = \
            call_presoak(args, input_path, output_folder, unknown_args)
        if not presoak_succeeded:
            print('presoak processing unsuccessful')
            all_succeeded = False
    else:
        print('presoak is not needed to run DELTA, skipping it')
        presoak_output_folder = None
        presoak_output_cost_path = None

    # DELTA
    delta_succeeded, delta_output_folder, delta_output_path = call_delta(args, input_path,
                                                                         output_folder, input_name,
                                                                         all_succeeded,
                                                                         presoak_output_cost_path)
    if not delta_succeeded:
        print('DELTA processing unsuccessful')
        all_succeeded = False

    # HMTFIST
    if all_succeeded and ENABLE_HMTFIST:
        all_succeeded = call_hmtfist(args, presoak_output_folder, presoak_output_dem_path,
                                     delta_output_folder, delta_output_path, output_folder)

    return all_succeeded

def main(argsIn): #pylint: disable=R0912

    # Parse input arguments
    try:
        usage  = "usage: classify_directory.py [options] <optional presoak arguments>"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-dir", "-i", required=True,
                            help="Directory containing input images")
        parser.add_argument("--unpack-inputs", action="store_true", default=False,
                            help="Add additional unpack step before processing if unputs are still packed")

        parser.add_argument("--output-dir", "-o", required=True,
                            help="Directory to store output images")
        parser.add_argument("--sensor", default="worldview",
                            help="Type of image [worldview, sentinel1]")
        parser.add_argument("--limit", "-l", type=int, default=None,
                            help="Stop after processing this many input images")

        parser.add_argument("--fist-data-dir", "-f", default=None,
                            help="Directory containing FIST required data")
        parser.add_argument("--canopy-path", "-c", default=None,
                            help="Path to the main canopy image file")
        parser.add_argument("--unfilled-presoak-dem", "-u", default=None,
                            help="Dem to be passed to presoak as --unfilled-elevation")

        parser.add_argument("--delta-model", "-m", required=True,
                            help="Model file for DELTA")
        parser.add_argument("--delta-config", "-g", required=True,
                            help="Config file for DELTA")
        parser.add_argument("--s1-delta-presoak-augment-model", default=None,
                            help="If provided, try to use this model with presoak cost output")

        parser.add_argument("--force-presoak", action="store_true", default=False,
                            help="Run presoak even if it is not needed for DELTA or HMTFIST")

        parser.add_argument("--hmt-params", "-p", default=None,
                            help="Path to HMTFIST parameter file")

        parser.add_argument("--keep-workdir", action="store_true", default=False,
                            help="Don't delete the working directory after running")

        # Any unrecognized arguments are passed to the presoak tool
        args, unknown_args = parser.parse_known_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return 1

    if args.sensor not in ['sentinel1', 'worldview']:
        print('Unrecognized sensor type: ' + args.sensor)
        return 1

    if args.s1_delta_presoak_augment_model and not args.fist_data_dir:
        print('WARNING: S1 presoak cost augmented DELTA model cannot be used without FIST!')

    print('Starting classification script')

    if ENABLE_HMTFIST:
        if not args.hmt_params:
            print('Missing required input argument --hmt-params to run HMTFIST!')
            return 1
        if not args.canopy_path:
            print('Missing required input argument --canopy-path to run HMTFIST!')
            return 1
    else:
        print('HMTFIST is disabled, set ENABLE_HMTFIST in this file to enable it.')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Look through the input folder to find the files we should process and decide which
    # output folder the results should go in
    target_paths = find_targets(args)
    if not target_paths:
        print('No unprocessed input files detected.')
        return 0
    print('Found ' + str(len(target_paths)) + ' input files to process.')


    # Loop through all input files...
    num_processed = 0
    for (input_path, output_folder) in target_paths:

        print('Starting processing for file: ' + input_path)
        all_succeeded = True

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        input_name = os.path.splitext(os.path.basename(input_path))[0]

        # Optional unpack step
        if args.unpack_inputs:
            unpacked_path = os.path.join(output_folder, 'unpacked_input.tif')
            if not unpack_input_image(input_path, args, unpacked_path):
                all_succeeded = False
            input_path = unpacked_path

        if all_succeeded:
            all_succeeded = classify_image(input_path, output_folder, input_name, args, unknown_args)

        if all_succeeded:
            print('Completed processing file: ' + input_path)
        else:
            print('Unable to complete processing for file: ' + input_path)

        num_processed += 1
        if args.limit and (num_processed >= args.limit):
            print('Hit limit of input images to process, stopping program.')
            break

    print('Classification script finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
