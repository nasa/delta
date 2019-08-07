"""
Convert .tif image(s) into a single TFrecord file consisting of multiple tiles.
"""
import os
import sys
import argparse
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

import tensorflow as tf #pylint: disable=C0413
from delta.imagery import utilities #pylint: disable=C0413
from delta.imagery.image_reader import * #pylint: disable=W0614,W0401,C0413
from delta.imagery import tfrecord_utils #pylint: disable=C0413


#------------------------------------------------------------------------------


def tiff_to_tf_record(input_paths, record_path, tile_size):
    """Convert a .tif file into a TFRecord file split into multiple tiles so
       that it is very easy to read using TensorFlow."""

    # Open the input image and get information about it
    input_reader = MultiTiffFileReader()
    input_reader.load_images(input_paths)
    (num_cols, num_rows) = input_reader.image_size()
    num_bands = input_reader.num_bands()
    data_type = utilities.gdal_dtype_to_numpy_type(input_reader.data_type())
    print('Using output data type: ' + str(data_type))

    input_bounds = utilities.Rectangle(0, 0, width=num_cols, height=num_rows)


    X = 0 # Make indices easier to read
    Y = 1

    # Make a list of output ROIs
    # TODO: Smart calculation of the tile size so no tiny tiles!
    num_blocks_out = (int(math.ceil(num_cols / tile_size[X])),
                      int(math.ceil(num_rows / tile_size[Y])))


    # Setting up output ROIs
    output_rois = []
    for r in range(0,num_blocks_out[Y]):
        for c in range(0,num_blocks_out[X]):

            # Get the ROI for the block
            roi = utilities.Rectangle(c*tile_size[X], r*tile_size[Y],
                                      width=tile_size[X], height=tile_size[Y])
            # Only keep whole ROIs, TF requires that all input tiles be the exact same dimensions!
            if input_bounds.contains_rect(roi):
                output_rois.append(roi)
    #print('Made ' + str(len(output_rois))+ ' output ROIs.')


    # Set up the output file, it will contain all the tiles from this input image.
    writer = tf.python_io.TFRecordWriter(record_path)

    def callback_function(output_roi, read_roi, data_vec):
        """Callback function to write the first channel to the output file."""

        # Figure out where the desired output data falls in read_roi
        x0 = output_roi.min_x - read_roi.min_x
        y0 = output_roi.min_y - read_roi.min_y
        x1 = x0 + output_roi.width()
        y1 = y0 + output_roi.height()

        # Pack all bands into a numpy array in the shape TF will expect later.
        array = np.zeros(shape=[output_roi.height(), output_roi.width(), num_bands], dtype=data_type)
        for band in range(0,num_bands):
            band_data = data_vec[band]
            #print(band_data.shape)
            array[:,:, band] = band_data[y0:y1, x0:x1] # Crop the correct region

        # DEBUG: Write the image segments to disk!
        #debug_path = os.path.join('/home/smcmich1/data/delta/test/WV02N42_939570W073_2520792013040400000000MS00/work/dup/', #pylint: disable=C0301
        #                          str(output_roi.min_y)+'_'+str(output_roi.min_x)+'.tif')
        #write_multiband_image(debug_path, array, data_type=gdal.GDT_UInt16)


        #back = np.fromstring(array_bytes, dtype=data_type)
        #back2 = back.reshape(num_bands, output_roi.height(), output_roi.width())
        #print(back.shape)
        #print(back2.shape)

        #raise Exception('DEBUG')

        tfrecord_utils.write_tfrecord_image(array, writer, output_roi.min_x, output_roi.min_y,
                                            output_roi.width(), output_roi.height(), num_bands)

    print('Writing TFRecord data...')
    # Each of the ROIs will be written out in order
    input_reader.process_rois(output_rois, callback_function)

    print('Done writing: ' + record_path)



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
        tiff_to_tf_record(options.input_images, options.output_path, options.tile_size)

    print('Script is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
