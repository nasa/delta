#!/usr/bin/env python
"""
Script test out the image chunk generation calls.
"""
import os
import sys
import argparse

### Tensorflow includes

import random
import mlflow
import tensorflow as tf
from tensorflow import keras

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../delta')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from imagery import landsat_utils #pylint: disable=C0413
from imagery import image_reader #pylint: disable=C0413
from imagery import utilities #pylint: disable=C0413
from ml.train import train #pylint: disable=C0413


#------------------------------------------------------------------------------

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

    mlflow.log_param('fc1_size',fc1_size)
    mlflow.log_param('fc2_size',fc2_size)
    mlflow.log_param('fc3_size',fc3_size)
    mlflow.log_param('fc4_size',fc4_size)
    mlflow.log_param('dropout_rate',dropout_rate)


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
### end make_model


def main(argsIn):
    parser = argparse.ArgumentParser(usage='usage: chunk_and_tensorflow [options]')

    parser.add_argument("--mtl-path", dest="mtl_path", default=None,
                        help="Path to the MTL file in the same folder as Landsat image band files.")

    parser.add_argument("--image-path", dest="image_path", default=None,
                        help="Instead of using an MTL file, just load this one image.")

    parser.add_argument("--output-folder", dest="output_folder",
                        default=os.path.join(os.path.dirname(__file__), '../data/out/test'),
                        help="Write output chunk files to this folder.")

    parser.add_argument("--output-band", dest="output_band", type=int, default=0,
                        help="Only chunks from this band are written to disk.")

    parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
                        help="Number of threads to use for parallel image loading.")

    # Note: changed the default chunk size to 28.  Smaller chunks work better for
    # the toy network defined above.
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=28,
                        help="The length of each side of the output image chunks.")

    parser.add_argument("--chunk-overlap", dest="chunk_overlap", type=int, default=0,
                        help="The amount of overlap of the image chunks.")

    parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=70,
                        help="The number of epochs to train for")
    try:
        options = parser.parse_args(argsIn)
    except argparse.ArgumentError:
        parser.print_help(sys.stderr)
        return -1

    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    if options.mtl_path:
        # Get all of the TOA coefficients and input file names
        data = landsat_utils.parse_mtl_file(options.mtl_path)

        input_folder = os.path.dirname(options.mtl_path)

        input_paths = []
        for fname in data['FILE_NAME']:
            input_path = os.path.join(input_folder, fname)
            input_paths.append(input_path)

    else: # Just load the specified image
        input_paths = [options.image_path]

    # Open the input image and get information about it
    input_reader = image_reader.MultiTiffFileReader()
    input_reader.load_images(input_paths)
    (num_cols, num_rows) = input_reader.image_size()

    # Process the entire input image(s) into chunks at once.
    roi = utilities.Rectangle(0,0,width=num_cols,height=num_rows)
    chunk_data = input_reader.parallel_load_chunks(roi, options.chunk_size,
                                                   options.chunk_overlap, options.num_threads)

    # For debug output, write each individual chunk to disk from a single band
    num_chunks = chunk_data.shape[0]
    num_bands  = chunk_data.shape[1]

    # Here is point where we would want to split the data into training and testing data
    # as well as labels and input data.
    all_data   = chunk_data[:,:7,:,:] # Use bands 0-6 to train on
    all_labels = chunk_data[:,7,:,:] # Use band 7 as the label?

    split_fraction = 0.7 # This percentage of data becomes training data
    # shuffle data:
    shuffled_idxs = list(range(num_chunks))
    random.shuffle(shuffled_idxs)
    split_idx  = int(split_fraction * num_chunks)
    train_idx  = shuffled_idxs[:split_idx]
#    test_idx   = shuffled_idxs[split_idx:]
    train_data = all_data[train_idx,:,:,:]
#    test_data  = all_data[test_idx, :,:,:]
    # Want to get the pixel at the middle (approx) of the chunk.
    train_labels = all_labels[train_idx, int(options.chunk_size / 2 ), int(options.chunk_size / 2)]
#    test_labels  = all_labels[test_idx, center_pixel,center_pixel]


    mlflow.set_tracking_uri('file:../data/out/mlruns')
    mlflow.set_experiment('chunk_and_tensorflow_test')
    mlflow.start_run()
    mlflow.log_param('file list', ' '.join(input_paths))
    mlflow.log_param('data_split', split_fraction)
    mlflow.log_param('chunk_size', options.chunk_size)

    model = make_model(num_bands-1, options.chunk_size)
    history = train(model, train_data, train_labels, options.num_epochs)
    assert history is not None

    mlflow.end_run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
