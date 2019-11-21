import argparse
import os
import sys
import time
import math
import functools
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # DEBUG: Process only on the CPU!

import numpy as np
import matplotlib.pyplot as plt

import mlflow
import tensorflow as tf #pylint: disable=C0413
from tensorflow import keras #pylint: disable=C0413

from delta.config import config #pylint: disable=C0413
from delta.imagery import imagery_dataset #pylint: disable=C0413
from delta.ml.train import Experiment


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
        keras.layers.Flatten(input_shape=(in_len, in_len, channel)),
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


def main(args): #pylint: disable=R0914,R0912,R0915
    parser = argparse.ArgumentParser(usage='sc_process_test.py [options]')

    parser.add_argument("--test-limit", dest="test_limit", type=int, default=0,
                        help="If set, use a maximum of this many input values for training.")

    parser.add_argument("--load-model", action="store_true", dest="load_model", default=False,
                        help="Start with the model saved in the current output location.")

    parser.add_argument("--experimental", action="store_true", dest="experimental", default=False,
                        help="Run experimental code!")

    options = config.parse_args(parser, args)

    # With TF 1.12, the dataset needs to be constructed inside a function passed in to
    # the estimator "train_and_evaluate" function to avoid getting a graph error!
    def assemble_dataset():

        # Use wrapper class to create a Tensorflow Dataset object.
        # - The dataset will provide image chunks and corresponding labels.
        ids = imagery_dataset.ImageryDataset(config.dataset(), config.chunk_size(), config.chunk_stride())
        ds = ids.dataset()

        #print("Num regions = " + str(ids.total_num_regions()))
        #if ids.total_num_regions() < batch_size:
        #    raise Exception('Batch size (%d) is too large for the number of input regions (%d)!'
        #                    % (batch_size, ids.total_num_regions()))
        ds = ds.batch(config.batch_size())

        #dataset = dataset.shuffle(buffer_size=1000) # Use a random order
        ds = ds.repeat(config.num_epochs()) # Need to be set here for use with train_and_evaluate

        if options.test_limit:
            ds = ds.take(options.test_limit)

        return ds

    print('Creating experiment')
    mlflow_tracking_dir = os.path.join(config.output_folder(), 'mlruns')
    if not os.path.exists(config.output_folder()):
        os.mkdir(config.output_folder())
    if not os.path.exists(mlflow_tracking_dir):
        os.mkdir(mlflow_tracking_dir)
    config_d = config.dataset()
    experiment = Experiment(mlflow_tracking_dir,
                            'autoencoder_%s'%(config_d.image_type()),
                            output_dir=config.output_folder())
    mlflow.log_param('image type',   config_d.image_type())
    mlflow.log_param('image folder', config_d.data_directory())
    mlflow.log_param('chunk size',   config.chunk_size())

    # Get these values without initializing the dataset (v1.12)
    ds_info = imagery_dataset.ImageryDataset(config.dataset(), config.chunk_size(), config.chunk_stride())
    model = make_model(ds_info.num_bands(), config.chunk_size())
    print('num images = ', ds_info.num_images())

    out_filename = os.path.join(config.output_folder(), config.model_dest_name())
    if options.load_model:
        print('Loading model from ' + out_filename)
        model_fn = functools.partial(tf.keras.models.load_model, out_filename)
    else:
        model_fn = functools.partial(make_model, ds_info.num_bands(),
                                     config.chunk_size())

    # Estimator interface requires the dataset to be constructed within a function.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO) # TODO 2.0
    model, _ = experiment.train_keras(model_fn, assemble_dataset,
                                      num_epochs=config.num_epochs(),
                                      num_gpus=config.num_gpus())
    #model.evaluate(assemble_dataset(), steps=steps_per_epoch)


    if config.model_dest_name() is not None:
        print('Saving Model to file: ' + out_filename)
        model.save(out_filename, overwrite=True, include_optimizer=True)
        mlflow.log_artifact(out_filename)

    if not options.experimental:
        print('sc_process_test finished!')
        return

    # --> Code below here is to generate an output image!!!
    # Needs to be fixed, then moved to a different tool!

    # TODO: Load the model from disk instead of putting this at the end of training!

    # TODO: Read these from file!
    height = 9216
    width = 14400
    chunk_stride = config.chunk_size()-1 # Set this to 1 to classify every pixel!
    tile_size = 256 # TODO: Where to get this?
    num_tiles_x = int(math.floor(width/tile_size))
    num_tiles_y = int(math.floor(height/tile_size))
    num_tiles = num_tiles_x*num_tiles_y
    print('num_tiles_x = ' + str(num_tiles_x))
    print('num_tiles_y = ' + str(num_tiles_y))

    # Make a non-shuffled dataset for only one image
    predict_batch = 200
    ids = imagery_dataset.ImageryDataset(config.dataset(), config.chunk_size(), chunk_stride)
    ds  = ids.data()
    ds  = ds.batch(predict_batch)

    print('Classifying the image...')
    start = time.time()

    predictions = model.predict(ds, verbose=1)

    stop = time.time()
    print('Output count = ' + str(len(predictions)))
    print('Elapsed time = ' + str(stop-start))

    # TODO: When actually classifying do not crop off partial tiles!
    #       May need to use the old imagery dataset class to do this!
    num_patches = len(predictions)
    patches_per_tile = int(num_patches / num_tiles)
    print('patches_per_tile = ' + str(patches_per_tile))
    patch_edge = int(math.sqrt(patches_per_tile))
    print('patch_edge = ' + str(patch_edge))

    # Convert the single vector of prediction values into the shape of the image
    # TODO: Account for the overlap value!
    i = 0
    #hist = []
    label_scale = 255
    pic = np.zeros([num_tiles_y*patch_edge, num_tiles_x*patch_edge], dtype=np.uint8)
    for ty in range(0,num_tiles_y):
        print(ty)
        for tx in range(0,num_tiles_x):
            row = ty*patch_edge
            for y in range(0,patch_edge): #pylint: disable=W0612
                col = tx*patch_edge
                for x in range(0,patch_edge): #pylint: disable=W0612
                    val = predictions[i]
                    #found = False
                    #for p, pair in enumerate(hist):
                    #    if pair[0] == val:
                    #        hist[p][1] = pair[1] + 1
                    #        found = True
                    #        break
                    #if not found:
                    #    #print(val)
                    #    hist.append([val, 1])
                    pic[row, col] = val * label_scale
                    i += 1
                    col += 1
                row += 1
    #pic = pic * label_scale # Why does this not work?
    #print('Histogram:')
    #for v in hist:
    #    print(str(v))

    output_path = '/home/smcmich1/repo/delta/output2.png'
    plt.imsave(output_path, pic)

    print('sc_process_test finished!')

if __name__ == "__main__":
    sys.exit(main(sys.argv))
