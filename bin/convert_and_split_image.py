#!/usr/bin/python
"""
Take a single input image and split it into eight tiles which are
then converted into TFRecord files.
"""
import sys
import argparse

from delta.imagery import tfrecord_conversions

#------------------------------------------------------------------------------


def main(argsIn): #pylint: disable=R0914,R0912

    try:
        usage  = "usage: convert_and_split_image.py [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-image", dest="input_image", required=True,
                            help="The input image file.")

        parser.add_argument("--output-prefix", dest="output_prefix", required=True,
                            help="Where to write the converted output images.")

        parser.add_argument("--work-folder", dest="work_folder", required=True,
                            help="Write temporary files here.")

        parser.add_argument("--keep", action="store_true", dest="keep", default=False,
                            help="Don't delete the uncompressed TFRecord files.")

        #parser.add_argument("--image-type", dest="image_type", required=True,
        #                    help="Specify the input image type [worldview, landsat, tif, rgba].")

        parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                            dest='tile_size', default=[256, 256], type=int,
                            help="Specify the size of the tiles the input images will be split up into.")

        parser.add_argument("--redo", action="store_true", dest="redo", default=False,
                            help="Re-write already existing output files.")

        parser.add_argument("--label", action="store_true", dest="label", default=False,
                            help="Set this when the input file is a label file.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    tfrecord_conversions.convert_and_divide_worldview(options.input_image, options.output_prefix,
                                                      options.work_folder,
                                                      options.label, keep=options.keep,
                                                      tile_size=options.tile_size, redo=options.redo)
    print('Finished!')

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
