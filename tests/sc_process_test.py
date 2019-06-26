import argparse
import os
import sys
import random

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # DEBUG: Process only on the CPU!

import tensorflow as tf #pylint: disable=C0413
from tensorflow import keras #pylint: disable=C0413

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from delta.imagery import landsat_utils #pylint: disable=C0413,W0611
from delta.imagery import worldview_utils #pylint: disable=C0413,W0611
from delta.imagery import imagery_dataset #pylint: disable=C0413

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

def main(args):
    parser = argparse.ArgumentParser(usage='sc_process_test.py [options]')

    parser.add_argument("--input-folder", dest="input_folder", required=True,
                        help="A folder with images to load.")
    parser.add_argument("--image-type", dest="image_type", required=True,
                        help="The imgae type (landsat, worldview, etc.).")

    try:
        options = parser.parse_args(args[1:])
    except argparse.ArgumentError:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if options.image_type == 'landsat':
        num_bands = len(landsat_utils.get_landsat_bands_to_use('LS8'))
    elif options.image_type == 'worldview':
        num_bands = len(worldview_utils.get_worldview_bands_to_use('WV02'))
    elif options.image_type == 'rgba':
        num_bands = 3
    else:
        print('Unsupported image type %s.' % (options.image_type), file=sys.stderr)
        sys.exit(1)

    # TODO: Figure out what reasonable values are here for each input sensor!
    CHUNK_SIZE = 35
    NUM_EPOCHS = 5
    BATCH_SIZE = 2 # Don't let this be greater than the number of input images!

    TEST_LIMIT = 256 # DEBUG: Only process this many image areas!

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    ids = imagery_dataset.ImageryDataset(options.image_type, image_folder=options.input_folder,
                                         chunk_size=CHUNK_SIZE)
    ds  = ids.dataset()

    if ids.total_num_regions() < BATCH_SIZE:
        raise Exception('BATCH_SIZE (%d) is too large for the number of input regions (%d)!'
                        % (BATCH_SIZE, ids.total_num_regions()))
    ds = ds.batch(BATCH_SIZE)

    #dataset = dataset.shuffle(buffer_size=1000) # Use a random order
    ds = ds.repeat()#(NUM_EPOCHS) # Helps with steps_per_epoch math below...

    # TODO: Set this up to help with parallelization
    #dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

    ds = ds.take(TEST_LIMIT) # DEBUG
    num_entries = ids.total_num_regions()
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
    sys.exit(main(sys.argv))
