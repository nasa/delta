"""
Code used to apply Top of Atmosphere correction to Landsat 5, 7, and 8 files.

TOA conversion is a simple per-pixel formula that applied using different constants for each band.
"""
import os
import sys
import math
import functools
import multiprocessing
import traceback
import numpy as np

from delta.imagery import utilities
from delta.imagery import large_image_tools
from delta.imagery.sources import landsat


#------------------------------------------------------------------------------

# Use this for all the output Landsat data we write.
OUTPUT_NODATA = 0.0

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


def do_landsat_toa_conversion(mtl_path, output_folder, tile_size=(256, 256), calc_reflectance=False, num_processes=1):
    """Convert landsat files in one folder to TOA corrected files in the output folder.
       Using the reflectance calculation is slightly more complicated but may be more useful.
       Multiprocessing is used if multiple processes are specified."""

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Get all of the TOA coefficients and input file names
    data = landsat.parse_mtl_file(mtl_path)

    # It is useful to skip the pool when using only one process
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

        rad_mult = data['RADIANCE_MULT'   ][band] # Get the parameters for this particular band
        rad_add  = data['RADIANCE_ADD'    ][band]
        ref_mult = data['REFLECTANCE_MULT'][band]
        ref_add  = data['REFLECTANCE_ADD' ][band]
        k1_const = data['K1_CONSTANT'     ][band]
        k2_const = data['K2_CONSTANT'     ][band]

        # Prepare the function we are going to use later
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

        if num_processes > 1: # Add to the pool
            task_handles.append(pool.apply_async( \
                try_catch_and_call, (input_path, output_path, user_function, tile_size, OUTPUT_NODATA)))
        else: # Direct call
            try_catch_and_call(input_path, output_path, user_function, tile_size, OUTPUT_NODATA)

    if num_processes > 1:
        # Wait for all the tasks to complete
        print('Finished adding ' + str(len(task_handles)) + ' tasks to the pool.')
        utilities.waitForTaskCompletionOrKeypress(task_handles, interactive=False)

        # All tasks should be finished, clean up the processing pool
        utilities.stop_task_pool(pool)
        print('Jobs finished.')
