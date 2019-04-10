
import os
import sys
import math
import functools
import random

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # DEBUG: Process only on the CPU!

import numpy as np
import tensorflow as tf
from tensorflow import keras
#import mlflow

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../delta')))


import image_reader
import utilities
import landsat_utils
import dataset_tools

# Test out importing tarred Landsat images into a dataset which is passed
# to a training function.

# WARNING!!!!!!!!
# There appears to be a serious bug in TensorFlow that affects this test,
# but it can be easily fixed by manually applying this fix to your
# TensorFlow installation:
#   https://github.com/tensorflow/tensorflow/pull/24522/files


def make_model(channel, in_len):
    # assumes square chunks.
    fc1_size = channel * in_len ** 2
#     fc2_size = fc1_size * 2
#     fc3_size = fc2_size
#     fc4_size = fc1_size
    # To be consistent with Robert's poster
    fc2_size = 253
    fc3_size = 253
    fc4_size = 81

    dropout_rate = 0.3 # Taken from Robert's code.

    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(channel, in_len, in_len)),
        # Note: use_bias is True by default, which is also the case in pytorch, which Robert used.
        keras.layers.Dense(fc2_size, activation=tf.nn.relu), 
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(fc3_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(fc4_size, activation=tf.nn.relu),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ])
    return model



def init_network(num_bands, chunk_size):
    """Create a TF model to train"""

    # TF additions
    seed_val = 12306 # number I typed out randomly on my keyboard

    random.seed(seed_val) # Probably poor form to use the same seed twice.
    tf.random.set_random_seed(seed_val)

    model = make_model(num_bands, chunk_size)
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])

    return model




def main(argsIn):

    # Make a list of source landsat files from the NEX collection
    
    # Supercomputer
    # TODO: Use a much larger list!
    #input_folder = '/nex/datapool/landsat/collection01/oli/T1/2015/113/052/' 
    #list_path = '/nobackup/smcmich1/delta/ls_list.txt'
    #CACHE_FOLDER = '/nobackup/smcmich1/delta/landsat'
    
    # A local test
    input_folder = '/home/smcmich1/data/landsat/tars'
    list_path = '/home/smcmich1/data/landsat/ls_list.txt'
    CACHE_FOLDER = '/home/smcmich1/data/landsat/cache'
    ext = '.gz'

    # Generate a text file list of all the input images, plus region indices.
    num_regions = 4
    num_entries = dataset_tools.make_landsat_list(input_folder, list_path, ext, num_regions)

    print('Wrote input file list of length: ' + str(num_entries))

    dataset = tf.data.TextLineDataset(list_path)

    CHUNK_SIZE = 256
    NUM_EPOCHS = 1
    
    TEST_LIMIT = 1 # DEBUG: Only process this many image areas!

    # TODO: We can define a different ROI function for each type of input image to
    #       achieve the sizes we want.
    # TODO: These values need to by synchronized with num_regions above!
    row_roi_split_funct  = functools.partial(dataset_tools.get_roi_horiz_band_split, num_splits=4)
    tile_roi_split_funct = functools.partial(dataset_tools.get_roi_tile_split,       num_splits=2)

    # This function prepares landsat images and returns the band paths
    ls_prep_func = functools.partial(landsat_utils.prep_landsat_image,
                                    cache_folder=CACHE_FOLDER)

    # This function loads the data and formats it for TF
    data_load_function = functools.partial(dataset_tools.load_image_region,
                                           prep_function=ls_prep_func,
                                           roi_function=row_roi_split_funct,
                                           chunk_size=CHUNK_SIZE, chunk_overlap=0, num_threads=2)

    # This function generates fake label info for loaded data.
    label_gen_function = functools.partial(dataset_tools.load_fake_labels,
                                           prep_function=ls_prep_func,
                                           roi_function=row_roi_split_funct,
                                           chunk_size=CHUNK_SIZE, chunk_overlap=0)


    # Tell TF to use the functions above to load our data.
    chunk_set = dataset.map( lambda lines: tf.py_func(data_load_function,
                                                      [lines], [tf.float64]),
                             num_parallel_calls=1)

    label_set = dataset.map( lambda lines: tf.py_func(label_gen_function,
                                                      [lines], [tf.int32]),
                             num_parallel_calls=1)

    # Pair the data and labels in our dataset
    dataset = tf.data.Dataset.zip((chunk_set, label_set))


    #dataset = dataset.shuffle(buffer_size=1000) # Use a random order
    #dataset = dataset.repeat(NUM_EPOCHS)

    # TODO: Set this up to help with parallelization
    #dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    dataset = dataset.take(TEST_LIMIT) # DEBUG
    if num_entries > TEST_LIMIT:
        num_entries = TEST_LIMIT

    # TODO: Is it a problem if image loads return different number of chunks?
    #       - May need to assure that each ROI contains the same number.

    ## Start a CPU-only session
    #config = tf.ConfigProto(device_count = {'GPU': 0})
    #sess   = tf.InteractiveSession(config=config)

    num_bands = len(landsat_utils.get_landsat_bands_to_use('LS8'))
    print('Num bands = ' + str(num_bands))
    model = init_network(num_bands, CHUNK_SIZE)

    # TODO: Does epochs need to be set here too?
    BATCH_SIZE = 1 # TODO: Difference between number of regions and number of chunks!
    history = model.fit(dataset, epochs=NUM_EPOCHS,
                        steps_per_epoch=num_entries//BATCH_SIZE)
#, batch_size=2048)


    print('TF dataset test finished!')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
