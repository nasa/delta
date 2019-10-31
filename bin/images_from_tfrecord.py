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


def images_from_tfrecord(input_path, output_prefix, width=4, compressed=True, label=False):
    """Extract entries from a tfrecord file and write them as plain image files"""

    # Get size information from the file
    image = tfrecord.TFRecordImage(input_path, None, 1, compressed)
    num_bands = image.num_bands()

    (input_region_height, input_region_width) = image.size()

    # Set up a reader object
    if compressed:
        reader = tf.data.TFRecordDataset(input_path, compression_type=tfrecord.TFRECORD_COMPRESSION_TYPE)
    else:
        reader = tf.data.TFRecordDataset(input_path, compression_type="")
    iterator = iter(reader)

    scaling = 1
    if label: # Get labels into a good 255 range
        scaling = 80

    # Read images until we hit an exception (end of file) then write out what we have
    total  = 0
    count  = 0
    concat = None
    while True:
        try:
            value = next(iterator)

            # Read the next item from the TFRecord file
            data_type = tf.uint8 if label else tf.float32
            image = tfrecord.load_tensor(value, num_bands, data_type = data_type)
            #print(str((input_region_height, input_region_width)))
            #print(sess.run(image))
            #continue
            pic = image[0,:,:,0] # Just take the first channel for now
            pic = tf.reshape(pic, ( input_region_height, input_region_width, 1))

            pic = tf.cast(pic*scaling, tf.uint8)

            # Assemble the pictures horizontally
            if concat is not None:
                concat = tf.concat([concat, pic], 1)
            else:
                concat = pic

            print('Count, total = ' + str((count, total)))
            # When we hit the desired width write out the data to an image file.
            if count == width:
                jpeg = tf.image.encode_jpeg(concat, quality=100, format='grayscale')
                output_path = output_prefix + str(total) + '.jpg'
                tf.io.write_file(output_path, jpeg)
                count  = 0
                concat = None
            count += 1
            total += 1
        except OutOfRangeError:
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

        parser.add_argument("--width", dest="width", type=int, default=8,
                            help="Default number of tiles to put in each output file.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    ## Make sure the output folder exists
    #if not os.path.exists(options.output_folder):
    #    os.mkdir(options.output_folder)

    images_from_tfrecord(options.input_path, options.output_prefix,
                         options.width, not options.uncompressed, options.label)


    print('Finished!')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
