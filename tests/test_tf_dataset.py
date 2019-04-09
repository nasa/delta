
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
from dataset_tools import *

# Test out TensorFlow Dataset classes.


def main(argsIn):

    # The inputs we will feed into the tensor graph
    # TODO: Have a function that makes a list of all of the input files.
    # TODO: If the list gets too long, we could offload it into a dataset file.
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

    data_load_function = functools.partial(load_image_region, roi_function=row_roi_split_function,
                                           chunk_size=256, chunk_overlap=0, num_threads=2)

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
    iterator = dataset.make_initializable_iterator() # -> Can pass this into Keras model.fit() call!
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
