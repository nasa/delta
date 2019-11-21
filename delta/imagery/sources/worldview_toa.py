"""
Script to apply Top of Atmosphere correction to WorldView 2 and 3 files.
"""
import math
import functools
import numpy as np

from delta.imagery.sources import worldview

#------------------------------------------------------------------------------

# Use this value for all WorldView nodata values we write, though they usually don't have any nodata.
OUTPUT_NODATA = 0.0

def get_esun_value(sat_id, band):
    """Get the ESUN value for the given satellite and band"""

    VALUES = {'WV02':[1580.814, 1758.2229, 1974.2416, 1856.4104,
                      1738.4791, 1559.4555, 1342.0695, 1069.7302, 861.2866],
              'WV03':[1583.58, 1743.81, 1971.48, 1856.26,
                      1749.4, 1555.11, 1343.95, 1071.98, 863.296]}
    try:
        return VALUES[sat_id][band]
    except Exception:
        raise Exception('No ESUN value for ' + sat_id
                        + ', band ' + str(band))

def get_earth_sun_distance():
    """Returns the distance between the Earth and the Sun in AU for the given date"""
    # TODO: Copy the calculation from the WV manuals.
    return 1.0

# The np.where clause handles input nodata values.

def apply_toa_radiance(data, _, bands, factors, widths):
    """Apply a top of atmosphere radiance conversion to WorldView data"""
    for b in bands:
        f = factors[b]
        w = widths[b]
        data[:, :, b] = np.where(data[:, :, b] > 0, (data[:, :, b] * f) / w, OUTPUT_NODATA)
    return data

def apply_toa_reflectance(data, band, factor, width, sun_elevation,
                          satellite, earth_sun_distance):
    """Apply a top of atmosphere reflectance conversion to WorldView data"""
    f = factor[band]
    w = width [band]

    esun    = get_esun_value(satellite, band)
    des2    = earth_sun_distance*earth_sun_distance
    theta   = np.pi/2.0 - sun_elevation
    scaling = (des2*np.pi) / (esun*math.cos(theta))
    return np.where(data>0, ((data*f)/w)*scaling, OUTPUT_NODATA)


def do_worldview_toa_conversion(image_path, output_path, tile_size=(256, 256), calc_reflectance=False):
    """Convert a WorldView input file to a TOA corrected output file.
       Using the reflectance calculation is slightly more complicated but may be more useful."""

    image = worldview.WorldviewImage(image_path)

    #ds = get_earth_sun_distance() # TODO: Implement this function!

    if not calc_reflectance:
        user_function = functools.partial(apply_toa_radiance, factors=image.scale(), widths=image.bandwidth())
    else:
        raise Exception('TODO: WV reflectance calculation is not fully implemented!')

        #user_function = functools.partial(apply_toa_reflectance, factor=scale, width=bwidth,
        #                                  sun_elevation=math.radians(data['MEANSUNEL']),
        #                                  satellite=data['SATID'],
        #                                  earth_sun_distance=ds)

    image.set_preprocess(user_function)
    image.save(output_path, tile_size=tile_size, nodata_value=OUTPUT_NODATA)
