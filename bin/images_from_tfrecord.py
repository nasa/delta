#!/usr/bin/python
"""
Take an input folder of image files and
convert them to a Tensorflow friendly format in the output folder.
"""
import os
import sys
import argparse
import tensorflow as tf #pylint: disable=C0413
from tensorflow.python.framework.errors_impl import OutOfRangeError #pylint: disable=E0611

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta.imagery import tfrecord_utils #pylint: disable=C0413

#------------------------------------------------------------------------------


def images_from_tfrecord(input_path, output_prefix, width=4, compressed=True):
    """Extract entries from a tfrecord file and write them as plain image files"""

    # Get size information from the file
    num_bands, input_region_height, input_region_width \
      = tfrecord_utils.get_record_info(input_path, compressed)

    # Set up a reader object
    if compressed:
        reader = tf.data.TFRecordDataset(input_path, compression_type=tfrecord_utils.TFRECORD_COMPRESSION_TYPE)
    else:
        reader = tf.data.TFRecordDataset(input_path, compression_type="")
    iterator = reader.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()

    # Read images until we hit an exception (end of file) then write out what we have
    total  = 0
    count  = 0
    concat = None
    while True:
        try:
            value = sess.run(next_element)

            # Read the next item from the TFRecord file
            image = tfrecord_utils.load_tfrecord_data_element(value, num_bands,
                                                              input_region_height, input_region_width)
            pic = image[0,:,:,0] # Just take the first channel for now
            pic = tf.reshape(pic, ( input_region_height, input_region_width, 1))
            #print(sess.run(pic))
            #continue
            pic = tf.cast(pic, tf.uint8) # TODO: May need to scale here

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
                writer = tf.write_file(output_path, jpeg)
                sess.run(writer)
                count  = 0
                concat = None
            count += 1
            total += 1
        except OutOfRangeError:
            if count > 0:
                jpeg = tf.image.encode_jpeg(concat, quality=100, format='grayscale')
                output_path = output_prefix + str(total) + '.jpg'
                writer = tf.write_file(output_path, jpeg)
                sess.run(writer)
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
                         options.width, not options.uncompressed)


    print('Finished!')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
