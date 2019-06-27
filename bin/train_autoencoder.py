"""
Script test out the image chunk generation calls.
"""
import sys
import os
import argparse
import random
import numpy as np

### Tensorflow includes

import mlflow

import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# TODO: Make sure this goes everywhere!
if sys.version_info < (3, 0, 0):
    print('\nERROR: Must use Python version >= 3.0.')
    sys.exit(1)

from delta.imagery.sources import landsat #pylint: disable=C0413
from delta.imagery.image_reader import * #pylint: disable=W0614,W0401,C0413
from delta.imagery.image_writer import * #pylint: disable=W0614,W0401,C0413


#------------------------------------------------------------------------------
def make_model(in_shape, encoding_size=32):

    mlflow.log_param('input_size',str(in_shape))
    mlflow.log_param('encoding_size',encoding_size)


    # Define network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=in_shape),
        keras.layers.Dense(encoding_size, activation=tf.nn.relu),
        keras.layers.Dense(np.prod(in_shape), activation=tf.nn.sigmoid),
        keras.layers.Reshape(in_shape)
        ])
    return model

### end make_model


def main(argsIn):

    #pylint: disable=R0914

    try:

        usage  = "usage: chunk_and_tensorflow [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--mtl-path", dest="mtl_path", default=None,
                            help="Path to the MTL file in the same folder as Landsat image band files.")

        parser.add_argument("--image-path", dest="image_path", default=None,
                            help="Instead of using an MTL file, just load this one image.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
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

        parser.add_argument("--num-hidden", dest="num_hidden", type=int, default=32,
                            help="The number of hidden elements in the autoencoder")

        parser.add_argument("--model-dest-name", dest="model_dest_name", type=str, default='autoencoder_model.keras.tf',
                            help="The name of the file where autoencoder is stored")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    if options.mtl_path:

        # Get all of the TOA coefficients and input file names
        data = landsat.parse_mtl_file(options.mtl_path)

        input_folder = os.path.dirname(options.mtl_path)

        input_paths = []
        for fname in data['FILE_NAME']:
            input_path = os.path.join(input_folder, fname)
            input_paths.append(input_path)

    else: # Just load the specified image
        input_paths = [options.image_path]

    # Open the input image and get information about it
    input_reader = MultiTiffFileReader()
    input_reader.load_images(input_paths)
    (num_cols, num_rows) = input_reader.image_size()

    # Process the entire input image(s) into chunks at once.
    roi = utilities.Rectangle(0,0,width=num_cols,height=num_rows)
    chunk_data = input_reader.parallel_load_chunks(roi, options.chunk_size,
                                                   options.chunk_overlap, options.num_threads)

    # For debug output, write each individual chunk to disk from a single band
    shape = chunk_data.shape
    num_chunks = shape[0]
    num_bands  = shape[1]
    print('num_chunks = ' + str(num_chunks))
    print('num_bands = ' + str(num_bands))

    print('Landsat chunker is finished.')

    # TF additions
    seed_val = 12306 # number I typed out randomly on my keyboard
    # If the mlfow directory doesn't exist, create it.
    mlflow_tracking_dir = os.path.join(options.output_folder,'mlruns')
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    mlflow.set_tracking_uri('file:%s'%(mlflow_tracking_dir,))

    mlflow.set_experiment('train_autoencoder')
    mlflow.start_run()
    mlflow.log_param('file list', ' '.join(input_paths))
    mlflow.log_param('seed_val', seed_val)

    random.seed(seed_val) # Probably poor form to use the same seed twice.
    tf.random.set_random_seed(seed_val)

    # Here is point where we would want to split the data into training and testing data
    # as well as labels and input data.
    all_data   = chunk_data[:,:7,:,:] # Use bands 0-6 to train on
    #all_labels = chunk_data[:,7,:,:] # Use band 7 as the label?

#     for idx in range(num_chunks):
#         print(np.unique(all_labels[idx,:,:]))
#         print(idx,'/',num_chunks)
#         plt.imshow(all_labels[idx,:,:])
#         plt.show()

    split_fraction = 0.7 # This percentage of data becomes training data
    mlflow.log_param('data_split', split_fraction)
    # shuffle data:
    shuffled_idxs = list(range(num_chunks))
    random.shuffle(shuffled_idxs)
    split_idx  = int(split_fraction * num_chunks)
    train_idx  = shuffled_idxs[:split_idx]
    test_idx   = shuffled_idxs[split_idx:]
    train_data = all_data[train_idx,:,:,:]
    test_data  = all_data[test_idx, :,:,:]
    # Want to get the pixel at the middle (approx) of the chunk.
    #center_pixel = int(options.chunk_size/2)
    #train_labels = all_labels[train_idx,center_pixel,center_pixel] # Center pixel becomes the label?
    #test_labels  = all_labels[test_idx, center_pixel,center_pixel]


    batch_size = 2048
    mlflow.log_param('chunk_size', options.chunk_size)
    mlflow.log_param('num_epochs', options.num_epochs)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('num_train', split_idx)
    mlflow.log_param('num_test', num_chunks - split_idx)

    # Remove one band for the labels
    data_shape = train_data[0,:,:,:].shape
    model = make_model(data_shape,encoding_size=options.num_hidden)
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
    history = model.fit(train_data, train_data, epochs=options.num_epochs, batch_size=batch_size)

    for idx in range(options.num_epochs):
        mlflow.log_metric('train_loss', history.history['loss'][idx])
        mlflow.log_metric('train_acc',  history.history['acc' ][idx])
    ### end for
    val_loss, val_acc = model.evaluate(test_data,test_data)

    mlflow.log_metric('test_loss',val_loss)
    mlflow.log_metric('test_acc',val_acc)

    if options.model_dest_name is not None:
        out_filename = os.path.join(options.output_folder,options.model_dest_name)
        tf.keras.models.save_model(model,out_filename,overwrite=True,include_optimizer=True)
        mlflow.log_artifact(out_filename)
    ### end if


    mlflow.end_run()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
