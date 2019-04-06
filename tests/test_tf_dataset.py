
import os
import sys
import math
import functools
import numpy as np
import tensorflow as tf

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../delta')))


import image_reader
import utilities

CHUNK_SIZE = 256
CHUNK_OVERLAP = 0


def get_roi_horiz_band_split(image_size, region, num_splits):
    """Return the ROI of an image to load given the region.
       Each region represents one horizontal band of the image.
    """

    assert region < num_splits, 'Input region ' + str(region) \
           + ' is greater than num_splits: ' + str(num_splits)

    min_x = 0
    max_x = image_size[0]

    # Fractional height here is fine
    band_height = image_size[1] / num_splits

    # TODO: Check boundary conditions!
    min_y = math.floor(band_height*region)
    max_y = math.floor(band_height*(region+1.0))

    return utilities.Rectangle(min_x, min_y, max_x, max_y)


def get_roi_tile_split(image_size, region, num_splits):
    """Return the ROI of an image to load given the region.
       Each region represents one tile in a grid split.
    """
    num_tiles = num_splits*num_splits
    assert region < num_tiles, 'Input region ' + str(region) \
           + ' is greater than num_tiles: ' + str(num_tiles)

    tile_row = math.floor(region / num_splits)
    tile_col = region % num_splits

    # Fractional sizes are fine here
    tile_width  = floor(image_size[0] / side)
    tile_height = floor(image_size[1] / side)

    # TODO: Check boundary conditions!
    min_x = math.floor(tile_width  * tile_col)
    max_x = math.floor(tile_width  * (tile_col+1.0))
    min_y = math.floor(tile_height * tile_row)
    max_y = math.floor(tile_height * (tile_row+1.0))

    return utilities.Rectangle(min_x, min_y, max_x, max_y)


def load_image_region(path, region, roi_function):
    """Load all image chunks for a given region of the image.
       The provided function converts the region to the image ROI.
    """

    # Set up the input image handle
    input_paths  = [path]
    input_reader = image_reader.MultiTiffFileReader()
    input_reader.load_images(input_paths)
    image_size = input_reader.image_size()

    # Call the provided function to get the ROI to load
    roi = roi_function(image_size, region)

    # Until we are ready to do a larger test, just return a short vector
    return np.array([roi.min_x, roi.min_y, roi.max_x, roi.max_y], dtype=np.int32) # DEBUG

    # Load the chunks from inside the ROI
    NUM_THREADS = 2
    chunk_data = input_reader.parallel_load_chunks(roi, CHUNK_SIZE, CHUNK_OVERLAP, NUM_THREADS)

    return chunk_data


def prep_input_pairs(image_file_list, num_regions):
    """For each image generate a pair with each region"""

    image_out  = []
    region_out = []
    for i in image_file_list:
        for r in range(0,num_regions):
            image_out.append(i)
            region_out.append(r)

    return (image_out, region_out)


def main(argsIn):

    # The inputs we will feed into the tensor graph
    # TODO: Have a function that makes a list of all of the input files.
    folder = '/home/smcmich1/data/toy_flood_images/landsat/'
    images = ['border.tif', 'border_june.tif']
    images = [os.path.join(folder, i) for i in images]
    num_regions = 4

    NUM_EPOCHS = 1

    # TODO: Consider using Dataset.list_files(pattern) here

    (paths, regions) = prep_input_pairs(images, num_regions)

    # Abstract the inputs a bit, maybe this makes the graph more efficient?
    paths_placeholder   = tf.placeholder(tf.string, len(paths  ))
    regions_placeholder = tf.placeholder(tf.int32,  len(regions))

    # Get our data paths into TF format
    dataset = tf.data.Dataset.from_tensor_slices((paths_placeholder, regions_placeholder))

    # TODO: We can define a different ROI function for each type of input image to
    #       achieve the sizes we want.
    row_roi_split_function  = functools.partial(get_roi_horiz_band_split, num_splits=4)
    tile_roi_split_function = functools.partial(get_roi_tile_split,       num_splits=2)

    data_load_function = functools.partial(load_image_region, roi_function=row_roi_split_function)

    # Tell TF how to load our data
    dataset = dataset.map( lambda paths, regions: tuple(tf.py_func(
                            data_load_function, [paths, regions], [tf.int32])),
                           num_parallel_calls=1)
    dataset = dataset.shuffle(buffer_size=1000) # Use a random order
    dataset = dataset.repeat(NUM_EPOCHS)

    # TODO: Set this up to help with parallelization
    #dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)


    # TODO: Is it a problem if image loads return different number of chunks?
    #       - May need to assure that each ROI contains the same number.

    # Start a CPU-only session
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess   = tf.InteractiveSession(config=config)

    # Make an iterator to our data
    iterator = dataset.make_initializable_iterator()
    op = iterator.get_next()

    # Point the placeholders at the actual inputs
    sess.run(iterator.initializer, feed_dict={paths_placeholder: paths,
                                              regions_placeholder: regions})

    # Fetch data until the iterator runs out
    while True:
        try:
            res = sess.run(op)
            # TODO: This is where something would use the output for training!
            print(res)
        except tf.errors.OutOfRangeError:
            break

    print('TF dataset test finished!')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
