"""
Code used to apply Top of Atmosphere correction to Landsat 5, 7, and 8 files.

TOA conversion is a simple per-pixel formula that applied using different constants for each band.
"""
import math
import functools
import numpy as np

from delta.imagery.sources import landsat

#------------------------------------------------------------------------------

# Use this for all the output Landsat data we write.
OUTPUT_NODATA = 0.0

def apply_toa_radiance(data, _, bands, factors, constants):
    """Apply a top of atmosphere radiance conversion to landsat data"""
    for b in bands:
        f = factors[b]
        c = constants[b]
        data[:, :, b] = np.where(data[:, :, b] > 0, data[:, :, b] * f + c, OUTPUT_NODATA)
    return data

def apply_toa_temperature(data, _, bands, factors, constants, k1, k2):
    """Apply a top of atmosphere radiance + temp conversion to landsat data"""
    for b in bands:
        f = factors[b]
        c = constants[b]
        k1 = k1[b]
        k2 = k2[b]
        data[:, :, b] = np.where(data[:, :, b] > 0, k2 / np.log(k1 / (data[:, :, b] * f + c) + 1.0), OUTPUT_NODATA)
    return data

def apply_toa_reflectance(data, _, bands, factors, constants, sun_elevation):
    """Apply a top of atmosphere radiance + temp conversion to landsat data"""
    for b in bands:
        f = factors[b]
        c = constants[b]
        se = sun_elevation[b]
        data[:, :, b] = np.where(data[:, :, b] > 0, (data[:, :, b] * f + c) / math.sin(se), OUTPUT_NODATA)
    return data

def do_landsat_toa_conversion(path, output_path, tile_size=(256, 256), calc_reflectance=False):
    """Convert landsat files in one folder to TOA corrected files in the output folder.
       Using the reflectance calculation is slightly more complicated but may be more useful.
       Multiprocessing is used if multiple processes are specified."""

    image = landsat.LandsatImage(path)

    if calc_reflectance:
        if image.k1_constant() is None:
            user_function = functools.partial(apply_toa_reflectance, factors=image.reflectance_mult(),
                                              constants=image.reflectance_add(),
                                              sun_elevation=math.radians(image.sun_elevation()))
        else:
            user_function = functools.partial(apply_toa_temperature, factors=image.radiance_mult(),
                                              constants=image.radiance_add(), k1=image.k1_constant(),
                                              k2=image.k2_constant())
    else:
        user_function = functools.partial(apply_toa_radiance, factors=image.radiance_mult(),
                                          constants=image.radiance_add())

    image.set_preprocess(user_function)
    image.save(output_path, tile_size=tile_size, nodata_value=OUTPUT_NODATA)
