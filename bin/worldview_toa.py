#!/usr/bin/python
"""
Script to apply Top of Atmosphere correction to WorldView 2 and 3 files.
"""
import sys
import argparse
import traceback

from delta.imagery.sources import worldview_toa

#------------------------------------------------------------------------------

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
        worldview_toa.do_worldview_toa_conversion(options.image_path, options.meta_path, options.output_path,
                                                  options.tile_size, options.calc_reflectance)
    except Exception:  #pylint: disable=W0703
        traceback.print_exc()
        sys.stdout.flush()
        return -1

    print('WorldView TOA conversion is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
