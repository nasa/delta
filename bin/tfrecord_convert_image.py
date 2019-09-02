"""
Convert .tif image(s) into a single TFrecord file consisting of multiple tiles.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

import tensorflow as tf #pylint: disable=C0413
from delta.imagery import tfrecord_utils #pylint: disable=C0413


#------------------------------------------------------------------------------


def unpack_image_record(record_path, output_folder):
    """Decode a tfrecord file full of images we created into
       separate .tif files on disk for debugging purposes."""

    raise Exception('TODO: Something useful in the reverse function!')

    if not os.path.exists(record_path): #pylint: disable=W0101
        raise Exception('Missing file: ' + record_path)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


    raw_image_dataset = tf.data.TFRecordDataset(record_path)

    parsed_image_dataset = raw_image_dataset.map(tfrecord_utils.load_tfrecord_data_element)

    iterator = parsed_image_dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    i = 0
    with tf.Session() as sess:
        while True:
            value = sess.run(next_batch)
            print(value.shape)
            #print(value.keys())
            #print(value['width'])
            #print(value['height'])
            #print(value['num_bands'])

            #array = tf.decode_raw(value['image_raw'], tf.uint16)
            #print(array.shape)
            #array2 = tf.reshape(array, tf.stack([value['num_bands'], value['height'], value['width']]))
            #print(array2.shape)

            #back = np.fromstring(value['image_raw'], dtype=np.ushort)
            #back2 = back.reshape(value['num_bands'], value['height'], value['width'])
            #debug_path = os.path.join(output_folder, str(i)+'.tif')
            #write_multiband_image(debug_path, back2, data_type=gdal.GDT_UInt16)
            #i = i + 1

            raise Exception('DEBUG')


def main(argsIn):
    try:

        usage  = "usage: tfrecord_convert_image [options] <input images>"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--output-path", dest="output_path", required=True,
                            help="Where to write the converted tfrecord file.")

        #parser.add_argument("--num-processes", dest="num_processes", type=int, default=1,
        #                    help="Number of parallel processes to use.")

        #parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
        #                    help="Number of threads to use per process.")

        # TODO: Auto-calculation option!
        parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                            dest='tile_size', default=[256, 256], type=int,
                            help="Specify the size of the tiles the input images will be split up into.")

        parser.add_argument("--reverse", action="store_true",
                            dest="reverse", default=False,
                            help="Split a tfrecord file into a folder containing the component tiles. TODO")

        parser.add_argument("input_images", nargs='*', help="Path to the input image (may be separate channel images).")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    if not options.input_images:
        print('Must provide at least one input image file!')
        return -1

    if options.reverse:
        unpack_image_record(options.input_images[0], options.output_path)
    else:
        tfrecord_utils.tiffs_to_tf_record(options.input_images, options.output_path, options.tile_size)

    print('Script is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
