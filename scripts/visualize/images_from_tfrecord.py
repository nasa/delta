#!/usr/bin/python
"""
Take an input folder of image files and
convert them to a Tensorflow friendly format in the output folder.
"""
import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from delta.imagery.sources import tfrecord

#------------------------------------------------------------------------------


def images_from_tfrecord(input_path, output_prefix, width_cap, compressed=True, label=False):
    """Extract entries from a tfrecord file and write them as plain image files"""

    # Get size information from the file
    image = tfrecord.TFRecordImage(input_path, compressed)
    num_bands = image.num_bands()

    dataset = tfrecord.create_dataset([input_path], num_bands, tf.uint8 if label else tf.float32,
                                      compressed=compressed)
    iterator = iter(dataset)

    scaling = 1
    if label: # Get labels into a good 255 range
        scaling = 80

    # Read images until we hit an exception (end of file) then write out what we have
    full_tile_width = None
    total  = 0
    count  = 0
    concat = None
    while True:
        try:
            image = next(iterator)

            # Get the size of this particulary tile
            (height, width, _) = image.shape
            if not full_tile_width:
                full_tile_width = width

            # Just take the first channel for now
            pic = image[:,:,0]
            pic = tf.reshape(pic, ( height, width, 1))

            pic = tf.cast(pic*scaling, tf.uint8)

            # Assemble the pictures horizontally
            if concat is not None:
                concat = tf.concat([concat, pic], 1)
            else:
                concat = pic

            print('Count, total = ' + str((count, total)))
            # Write out the concatenated image when we hit a narrow (end) tile or get too big
            # - This strategy works well for individual images but not for mixed images.
            if (width != full_tile_width) or (count >= width_cap):
                jpeg = tf.image.encode_jpeg(concat, quality=100, format='grayscale')
                output_path = output_prefix + str(total) + '.jpg'
                tf.io.write_file(output_path, jpeg)
                count  = 0
                concat = None
            count += 1
            total += 1
        except (OutOfRangeError, StopIteration):
            if count > 0:
                jpeg = tf.image.encode_jpeg(concat, quality=100, format='grayscale')
                output_path = output_prefix + str(total) + '.jpg'
                tf.io.write_file(output_path, jpeg)
            break


def main(argsIn):

    try:

        usage  = "usage: images_from_tfrecord.py [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-file", dest="input_path", required=True,
                            help="Path to the input file.")

        parser.add_argument("--output-prefix", dest="output_prefix", required=True,
                            help="Output prefix for images to write.")

        parser.add_argument("--uncompressed", action="store_true", dest="uncompressed", default=False,
                            help="Set if the input file is an uncompressed TFRecord file.")

        parser.add_argument("--label", action="store_true", dest="label", default=False,
                            help="The input file is a label file.")

        parser.add_argument("--width-cap", dest="width_cap", type=int, default=20,
                            help="Cap the number of tiles in a row at this amount.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    ## Make sure the output folder exists
    #if not os.path.exists(options.output_folder):
    #    os.mkdir(options.output_folder)

    images_from_tfrecord(options.input_path, options.output_prefix,
                         options.width_cap, not options.uncompressed, options.label)


    print('Finished!')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
