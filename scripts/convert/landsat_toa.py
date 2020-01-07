#!/usr/bin/env python
"""
Script to apply Top of Atmosphere correction to Landsat 5, 7, and 8 files.
"""
import sys
import argparse

from delta.imagery.sources import landsat


#------------------------------------------------------------------------------

def main(argsIn):

    try:

        usage  = "usage: landsat_toa [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--image-path", dest="image_path", required=True,
                            help="Path to Landsat file.")

        parser.add_argument("--output-file", dest="output_file", required=True,
                            help="Write the output to this file.")

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

    image = landsat.LandsatImage(options.iamge_path)
    landsat.toa_preprocess(image, options.calc_reflectance)
    image.save(options.output_file, tile_size=options.tile_size, show_progress=True)

    print('Landsat TOA conversion is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
