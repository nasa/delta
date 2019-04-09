
import os
import sys
import math
import functools
import numpy as np
import tensorflow as tf
import mlflow

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../delta')))


import image_reader
import utilities
import dataset_tools

# Test out large scale data importing for the supercomputer.


def init_network(num_bands, chunk_size):
    """Create a TF model to train"""

    # TF additions
    seed_val = 12306 # number I typed out randomly on my keyboard

    random.seed(seed_val) # Probably poor form to use the same seed twice.
    tf.random.set_random_seed(seed_val)

    batch_size = 2048

    model = make_model(num_bands, chunk_size)
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])

    return model




def main(argsIn):

    # TODO: Use a much larger list!
    # Make a list of source landsat files from the NEX collection
    input_folder = '/nex/datapool/landsat/collection01/oli/T1/2015/113/052/'
    ext = '.tar.gz'
    list_path = '~/ls_list.txt'
    num_regions = 4
    num_entries = dataset_tools.make_landsat_list(input_folder, list_path, ext, num_regions)
    
    print('Wrote input file list of length: ' + str(num_entries))
    
    dataset = tf.data.TextLineDataset(list_path)
    
    raise Exception('DEBUG')


    CHUNK_SIZE = 256
    NUM_EPOCHS = 1

    # TODO: We can define a different ROI function for each type of input image to
    #       achieve the sizes we want.
    row_roi_split_funct  = functools.partial(dataset_tools.get_roi_horiz_band_split, num_splits=4)
    tile_roi_split_funct = functools.partial(dataset_tools.get_roi_tile_split,       num_splits=2)

    data_load_function = functools.partial(dataset_tools.load_image_region,
                                           prep_function=dataset_tools.prep_landsat_image
                                           roi_function=dataset_tools.row_roi_split_funct,
                                           chunk_size=CHUNK_SIZE, chunk_overlap=0, num_threads=2)

    # Tell TF how to load our data
    dataset = dataset.map( lambda lines: tuple(tf.py_func(data_load_function,
                                                          [lines], [tf.string])),
                           num_parallel_calls=1)
    dataset = dataset.shuffle(buffer_size=1000) # Use a random order
    dataset = dataset.repeat(NUM_EPOCHS)

    # TODO: Set this up to help with parallelization
    #dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    TEST_LIMIT = 1 # Only process this many image areas!
    dataset = dataset.take(TEST_LIMIT) # DEBUG

    # TODO: Is it a problem if image loads return different number of chunks?
    #       - May need to assure that each ROI contains the same number.

    # Start a CPU-only session
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess   = tf.InteractiveSession(config=config)

    # Make an iterator to our data
    #iterator = dataset.make_one_shot_iterator() # -> Can pass this into Keras model.fit() call!

    num_bands = len(dataset_tools.get_landsat_bands_to_use('LS8'))
    print('Num bands = ' + str(num_bands))
    model = init_network(num_bands, CHUNK_SIZE)

    # TODO: Does epochs need to be set here too?
    history = model.fit(dataset, epochs=NUM_EPOCHS, batch_size=batch_size)


    print('TF dataset test finished!')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
