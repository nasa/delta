"""
Script to apply Top of Atmosphere correction to WorldView 2 and 3 files.
"""
import os
import sys
import argparse
import math
import functools
import traceback
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta.imagery import large_image_tools #pylint: disable=C0413
from delta.imagery.sources import worldview #pylint: disable=C0413

#------------------------------------------------------------------------------

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

def apply_toa_radiance(data, band, factor, width):
    """Apply a top of atmosphere radiance conversion to WorldView data"""
    f = factor[band]
    w = width [band]
    return np.where(data>0, (data*f)/w, OUTPUT_NODATA)

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


def do_work(image_path, meta_path, output_path, tile_size=(256, 256), calc_reflectance=False):
    """Do all the the work past command line interpretation"""

    # Get all of the TOA coefficients and input file names
    data = worldview.parse_meta_file(meta_path)
    #print(data)

    scale  = data['ABSCALFACTOR']
    bwidth = data['EFFECTIVEBANDWIDTH']

    ds = get_earth_sun_distance() # TODO: Implement this function!

    if calc_reflectance:
        user_function = functools.partial(apply_toa_reflectance, factor=scale, width=bwidth,
                                          sun_elevation=math.radians(data['MEANSUNEL']),
                                          satellite=data['SATID'],
                                          earth_sun_distance=ds)
    else:
        user_function = functools.partial(apply_toa_radiance, factor=scale, width=bwidth)

    large_image_tools.apply_function_to_file(image_path, output_path, user_function, tile_size, OUTPUT_NODATA)


def main(argsIn):

    try:

        usage  = "usage: worldview_toa [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--image-path", dest="image_path", required=True,
                            help="Path to the image file.")

        parser.add_argument("--meta-path", dest="meta_path", required=True,
                            help="Path to the metadata file (.IMD or .xml).")

        parser.add_argument("--output-path", dest="output_path", required=True,
                            help="Where to write the output image.")

        parser.add_argument("--calc-reflectance", action="store_true",
                            dest="calc_reflectance", default=False,
                            help="Compute TOA reflectance (and temperature) instead of radiance.")

        #parser.add_argument("--num-processes", dest="num_processes", type=int, default=1,
        #                    help="Number of parallel processes to use.")

        #parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
        #                    help="Number of threads to use per process.")

        parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                            dest='tile_size', default=[256, 256], type=int,
                            help="Specify the output tile size.  Default is to keep the input tile size.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    try:
        do_work(options.image_path, options.meta_path, options.output_path, options.tile_size, options.calc_reflectance)
    except Exception:  #pylint: disable=W0703
        traceback.print_exc()
        sys.stdout.flush()
        return -1

    print('WorldView TOA conversion is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
