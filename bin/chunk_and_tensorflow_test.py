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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta.imagery import landsat_utils #pylint: disable=C0413
from delta.imagery import image_reader #pylint: disable=C0413
from delta.imagery import utilities #pylint: disable=C0413
from delta.ml.train import Experiment #pylint: disable=C0413


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


def main(argsIn): # pylint:disable=R0914
    parser = argparse.ArgumentParser(usage='chunk_and_tensorflow [options]')

    parser.add_argument("--mtl-path", dest="mtl_path", default=None,
                        help="Path to the MTL file in the same folder as Landsat image band files.")
    parser.add_argument("--image-path", dest="image_path", default=None,
                        help="Instead of using an MTL file, just load this one image.")
    parser.add_argument("--label-path", dest="label_path", default=None,
                        help="Path to a label file for this image.  If not used, will train on junk labels.")
    parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
                        help="Number of threads to use for parallel image loading.")
    parser.add_argument("--output-folder", dest="output_folder",
                        default=os.path.join(os.path.dirname(__file__), '../data/out/test'),
                        help="Write output chunk files to this folder.")

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

    if options.label_path and not os.path.exists(options.label_path):
        print('Label file does not exist: ' + options.label_path)

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
    if options.label_path:
        # TODO: What is the best way to use the label image?
        # Read in the image as single pixel chunks
        print('Loading label data...')
        label_reader = image_reader.MultiTiffFileReader()
        label_reader.load_images([options.label_path])
        if label_reader.image_size() != input_reader.image_size():
            print('Label image size does not match input image size!')
            return -1
        label_data = label_reader.parallel_load_chunks(roi, options.chunk_size,
                                                       options.chunk_overlap, options.num_threads)
    print('Done loading data.')

    # For debug output, write each individual chunk to disk from a single band
    num_chunks = chunk_data.shape[0]

    # Here is point where we would want to split the data into training and testing data
    # as well as labels and input data.
    NUM_TRAIN_BANDS = 7 # TODO: Pick bands!
    all_data   = chunk_data[:,:NUM_TRAIN_BANDS,:,:] # Use bands 0-6 to train on
    if options.label_path:
        all_labels = label_data[:,0,:,:] # Only one band
    else:
        all_labels = chunk_data[:,NUM_TRAIN_BANDS, :,:] # Use band 7 as the label

    # shuffle data:
    split_fraction = 0.7
    shuffled_idxs = list(range(num_chunks))
    random.shuffle(shuffled_idxs)
    split_idx  = int(split_fraction * num_chunks)  # This percentage of data becomes training data
    train_idx  = shuffled_idxs[:split_idx]
    test_idx   = shuffled_idxs[split_idx:]
    train_data = all_data[train_idx,:,:,:]
    test_data  = all_data[test_idx, :,:,:]
    # Want to get the pixel at the middle (approx) of the chunk.
    center_pixel = int(options.chunk_size/2)

    if options.label_path:
        train_labels = all_labels[train_idx, center_pixel, center_pixel]
        test_labels  = all_labels[test_idx, center_pixel, center_pixel]

    else: # Use junk labels (center pixel value)
        train_labels = all_labels[train_idx, center_pixel, center_pixel]
        test_labels  = all_labels[test_idx, center_pixel,center_pixel]

#     mlflow.set_tracking_uri('file:../data/out/mlruns')
#     mlflow.set_experiment('chunk_and_tensorflow_test')
#     mlflow.start_run()

    experiment = Experiment('file:../data/out/mlruns', 'chunk_and_tensorflow_test')
    
    exp_parameters = {'file list':' '.join(input_paths), 
            'data_split': split_fraction,
            'chunk_size': options.chunk_size}
    experiment.log_parameters(exp_parameters)

    # Remove one band for the labels
    model = make_model(NUM_TRAIN_BANDS, options.chunk_size)
    history = experiment.train(model, train_data, train_labels, options.num_epochs, validation_data=(test_data, test_labels))
    assert history is not None

#     mlflow.end_run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
