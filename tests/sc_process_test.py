import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # DEBUG: Process only on the CPU!

import tensorflow as tf #pylint: disable=C0413
from tensorflow import keras #pylint: disable=C0413

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    model = make_model(num_bands, chunk_size)
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])

    return model

def main(args):
    parser = argparse.ArgumentParser(usage='sc_process_test.py [options]')

    parser.add_argument("--dataset-config-file", dest="dataset_config", required=True,
                        help="Dataset configuration file.")
    parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=2,
                        help="Training batch size.")
    parser.add_argument("--test-limit", dest="test_limit", type=int, default=0,
                        help="If set, use a maximum of this many input values for training.")

    try:
        options = parser.parse_args(args[1:])
    except argparse.ArgumentError:
        parser.print_help(sys.stderr)
        sys.exit(1)


    TEST_LIMIT = 256 # DEBUG: Only process this many image areas!

    # Use wrapper class to create a Tensorflow Dataset object.
    # - The dataset will provide image chunks and corresponding labels.
    ids = imagery_dataset.ImageryDataset(options.dataset_config)
    ds = ids.dataset()

    if ids.total_num_regions() < options.batch_size:
        raise Exception('Batch size (%d) is too large for the number of input regions (%d)!'
                        % (options.batch_size, ids.total_num_regions()))
    ds = ds.batch(options.batch_size)

    #dataset = dataset.shuffle(buffer_size=1000) # Use a random order
    ds = ds.repeat()#(NUM_EPOCHS) # Helps with steps_per_epoch math below...

    # TODO: Set this up to help with parallelization
    #dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)


    num_entries = ids.total_num_regions()
    if options.test_limit:
        ds = ds.take(options.test_limit)
        num_entries = ids.total_num_regions()
        if num_entries > options.test_limit:
            num_entries = options.test_limit

    ## Start a CPU-only session
    #config = tf.ConfigProto(device_count = {'GPU': 0})
    #sess   = tf.InteractiveSession(config=config)

    model = init_network(ids.num_bands(), ids.chunk_size())

    unused_history = model.fit(ds, epochs=options.num_epochs,
                               steps_per_epoch=num_entries//options.batch_size)

    print('TF dataset test finished!')

if __name__ == "__main__":
    sys.exit(main(sys.argv))
