
import os
import sys
import random

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # DEBUG: Process only on the CPU!

import tensorflow as tf #pylint: disable=C0413
from tensorflow import keras #pylint: disable=C0413

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../delta')))


from imagery import landsat_utils #pylint: disable=C0413,W0611
from imagery import worldview_utils #pylint: disable=C0413,W0611
from imagery import dataset #pylint: disable=C0413

# Test out importing tarred Landsat images into a dataset which is passed
# to a training function.

def make_model(channel, in_len):
    # assumes square chunks.
#    fc1_size = channel * in_len ** 2
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

def main():
    # Make a list of source landsat files from the NEX collection

    # TODO: create config file with paths
    #input_folder = '/home/bcoltin/data/landsat'
    #list_path    = '/home/bcoltin/data/landsat/ls_list.txt'
    #cache_folder = '/home/bcoltin/data/landsat/cache'

    # Supercomputer
    # TODO: Use a much larger list!
    #input_folder = '/nex/datapool/landsat/collection01/oli/T1/2015/113/052/'
    #list_path    = '/nobackup/smcmich1/delta/ls_list.txt'
    #cache_folder = '/nobackup/smcmich1/delta/landsat'

    # A local test
    input_folder = '/home/smcmich1/data/landsat/tars'
    list_path    = '/home/smcmich1/data/landsat/ls_list.txt'
    cache_folder = '/home/smcmich1/data/landsat/cache'

    #user='pfurlong'
    #input_folder = '/home/%s/data/landsat/tars' % (user,)
    #list_path    = '/home/%s/data/landsat/ls_list.txt' % (user,)
    #cache_folder = '/home/%s/data/landsat/cache' % (user,)

    num_regions = 4
    num_bands = len(landsat_utils.get_landsat_bands_to_use('LS8'))
    image_type = 'landsat'

    # WorldView test
    #input_folder = '/home/smcmich1/data/delta/hdds'
    #list_path    = '/home/smcmich1/data/wv_list.txt'
    #cache_folder = '/home/smcmich1/data/delta_cache'
    #image_type = 'worldview'
    #num_regions = 16
    #num_bands = len(worldview_utils.get_worldview_bands_to_use('WV02'))

    cache_limit = 4

    # TODO: Figure out what reasonable values are here
    CHUNK_SIZE = 256
    NUM_EPOCHS = 5
    BATCH_SIZE = 4

    TEST_LIMIT = 256 # DEBUG: Only process this many image areas!

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    ids = dataset.ImageryDataset(input_folder, image_type, list_path,
                                 cache_folder, cache_limit=cache_limit,
                                 chunk_size=CHUNK_SIZE, num_regions=num_regions)
    ds  = ids.dataset()

    ds = ds.batch(BATCH_SIZE)

    #dataset = dataset.shuffle(buffer_size=1000) # Use a random order
    ds = ds.repeat()#(NUM_EPOCHS) # Helps with steps_per_epoch math below...

    # TODO: Set this up to help with parallelization
    #dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    ds = ds.take(TEST_LIMIT) # DEBUG
    num_entries = ids.num_regions()
    if num_entries > TEST_LIMIT:
        num_entries = TEST_LIMIT

    ## Start a CPU-only session
    #config = tf.ConfigProto(device_count = {'GPU': 0})
    #sess   = tf.InteractiveSession(config=config)

    #print('Num bands = ' + str(num_bands))
    model = init_network(num_bands, CHUNK_SIZE)

    unused_history = model.fit(ds, epochs=NUM_EPOCHS,
                               steps_per_epoch=num_entries//BATCH_SIZE)

    print('TF dataset test finished!')

if __name__ == "__main__":
    sys.exit(main())
