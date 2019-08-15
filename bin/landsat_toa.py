"""
Script to apply Top of Atmosphere correction to Landsat 5, 7, and 8 files.
"""
import os
import sys
import argparse
import math
import functools
import multiprocessing
import traceback
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta.imagery import utilities #pylint: disable=C0413
from delta.imagery import large_image_tools #pylint: disable=C0413
from delta.imagery.sources import landsat #pylint: disable=C0413


#------------------------------------------------------------------------------

OUTPUT_NODATA = 0.0

# Cleaner ways to do this don't work with multiprocessing!
def try_catch_and_call(*args, **kwargs):
    """Wrap the previous function in a try/catch statement"""
    try:
        return large_image_tools.apply_function_to_file(*args, **kwargs)
    except Exception:  #pylint: disable=W0703
        traceback.print_exc()
        sys.stdout.flush()
        return -1

# The np.where clause handles input nodata values.

def apply_toa_radiance(data, factor, constant):
    """Apply a top of atmosphere radiance conversion to landsat data"""
    return np.where(data>0, (data * factor) + constant, OUTPUT_NODATA)

def apply_toa_temperature(data, factor, constant, k1, k2):
    """Apply a top of atmosphere radiance + temp conversion to landsat data"""
    return np.where(data>0, k2/np.log(k1/((data*factor)+constant) +1.0), OUTPUT_NODATA)

def apply_toa_reflectance(data, factor, constant, sun_elevation):
    """Apply a top of atmosphere radiance + temp conversion to landsat data"""
    return np.where(data>0, ((data*factor)+constant)/math.sin(sun_elevation), OUTPUT_NODATA)


def do_work(mtl_path, output_folder, tile_size=(256, 256), calc_reflectance=False, num_processes=1):
    """Function where all of the work happens"""

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Get all of the TOA coefficients and input file names
    data = landsat.parse_mtl_file(mtl_path)

    if num_processes > 1:
        pool = multiprocessing.Pool(num_processes)
        task_handles = []

    # Loop through the input files (each band is processed separately)
    input_folder = os.path.dirname(mtl_path)
    num_bands    = len(data['FILE_NAME'])
    for band in range(0, num_bands):

        fname = data['FILE_NAME'][band]

        input_path  = os.path.join(input_folder,  fname)
        output_path = os.path.join(output_folder, fname)

        rad_mult = data['RADIANCE_MULT'   ][band]
        rad_add  = data['RADIANCE_ADD'    ][band]
        ref_mult = data['REFLECTANCE_MULT'][band]
        ref_add  = data['REFLECTANCE_ADD' ][band]
        k1_const = data['K1_CONSTANT'     ][band]
        k2_const = data['K2_CONSTANT'     ][band]

        if calc_reflectance:
            if k1_const is None:
                user_function = functools.partial(apply_toa_reflectance, factor=ref_mult,
                                                  constant=ref_add,
                                                  sun_elevation=math.radians(data['SUN_ELEVATION']))
            else:
                user_function = functools.partial(apply_toa_temperature, factor=rad_mult,
                                                  constant=rad_add, k1=k1_const, k2=k2_const)
        else:
            user_function = functools.partial(apply_toa_radiance, factor=rad_mult, constant=rad_add)

        if num_processes > 1:
            task_handles.append(pool.apply_async( \
                try_catch_and_call, (input_path, output_path, user_function, tile_size, OUTPUT_NODATA)))
        else: # Direct call
            try_catch_and_call(input_path, output_path, user_function, tile_size, OUTPUT_NODATA)

        #raise Exception('DEBUG')

    if num_processes > 1:
        # Wait for all the tasks to complete
        print('Finished adding ' + str(len(task_handles)) + ' tasks to the pool.')
        utilities.waitForTaskCompletionOrKeypress(task_handles, interactive=False)

        # All tasks should be finished, clean up the processing pool
        utilities.stop_task_pool(pool)
        print('Jobs finished.')


def main(argsIn):

    try:

        usage  = "usage: landsat_toa [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--mtl-path", dest="mtl_path", required=True,
                            help="Path to the MTL file in the same folder as the image band files.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Write output band files to this output folder with the same names.")

        parser.add_argument("--calc-reflectance", action="store_true",
                            dest="calc_reflectance", default=False,
                            help="Compute TOA reflectance (and temperature) instead of radiance.")

        parser.add_argument("--num-processes", dest="num_processes", type=int, default=1,
                            help="Number of parallel processes to use.")

        parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                            dest='tile_size', default=[256, 256], type=int,
                            help="Specify the output tile size.  Default is to keep the input tile size.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    do_work(options.mtl_path, options.output_folder, options.tile_size,
            options.calc_reflectance, options.num_processes)

    print('Landsat TOA conversion is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
