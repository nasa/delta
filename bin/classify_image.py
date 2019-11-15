#!/usr/bin/python
"""
Script test out the image chunk generation calls.
"""
import sys
import os
import math
import time
#import filecmp
import shutil
import argparse
#import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

#import mlflow
import tensorflow as tf

from delta.config import config
from delta.imagery import rectangle
from delta.imagery import imagery_dataset
from delta.imagery import tfrecord_conversions
from delta.imagery.sources import tiff
#from delta.ml.train import Experiment
from delta.ml.train import load_keras_model



def do_work(options): #pylint: disable=R0914,R0912,R0915
    '''Do all of the work after options are parsed'''

    # Convert the input image into a format we can process

    TILE_SIZE = 256 # Any reason to change this?

    config.load(options.config_path)

    print('Loading model from ' + options.keras_model)
    model = load_keras_model(options.keras_model, options.num_gpus)

    if not options.work_folder:
        options.work_folder = options.output_path + '_work'
    tfrecord_path          = os.path.join(options.work_folder, 'converted.tfrecord')
    prediction_backup_path = os.path.join(options.work_folder, 'predict_array.npy' )
    config_copy_path       = os.path.join(options.work_folder, 'delta_config_copy.txt')

    # TODO: Auto remove old work folders?
    #redo = not (os.path.exists(config_copy_path) and filecmp.cmp(config_copy_path, options.config_path))
    if os.path.exists(config_copy_path):
        print('Detected existing work folder.  This tool will break if parameters have changed since the last run!')

    chunk_size = config.chunk_size()
    if options.low_res:
        chunk_overlap = 0
    else: # Full res
        chunk_overlap = int(chunk_size) - 1
    chunk_stride = chunk_size - chunk_overlap # Update this
    tile_gen_overlap = chunk_overlap # TODO: The tfrecord can't be cached if this changes!
    include_partials = (tile_gen_overlap>0) # This is how it is decided in this function call, could be changed.
    image_size, metadata = \
      tfrecord_conversions.convert_image_to_tfrecord(options.input_image, [tfrecord_path],
                                                     options.work_folder,
                                                     (TILE_SIZE, TILE_SIZE), options.image_type,
                                                     tile_overlap=tile_gen_overlap)
    if not os.path.exists(tfrecord_path):
        print('Failed to convert input image!')
        return -1

    shutil.copy(options.config_path, config_copy_path)

    #(num_bands, height, width) = tfrecord_utils.get_record_info(tfrecord_path, compressed=True)
    (width, height) = image_size
    print('width  = ' + str(width))
    print('height = ' + str(height))

    # Estimator interface requires the dataset to be constructed within a function.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO) # TODO 2.0


    # Make a non-shuffled dataset for only one image
    predict_batch = 200
    ids = imagery_dataset.ClassifyDataset(config.dataset(), tfrecord_path,
                                          chunk_size, chunk_stride)
    ds  = ids.dataset()
    ds  = ds.batch(predict_batch)

    print('Classifying the image...')
    start = time.time()

    if os.path.exists(prediction_backup_path):
        predictions = np.load(prediction_backup_path)
    else:
        predictions = model.predict(ds, verbose=1)
        np.save(prediction_backup_path, predictions)

    num_patches = len(predictions)
    stop = time.time()
    print('Output count = ' + str(num_patches))
    print('Classify time = ' + str(stop-start))

    # TODO: Replace with a function call to Rectangle
    if include_partials:
        num_tiles_x = int(math.ceil(width /(TILE_SIZE-tile_gen_overlap)))
        num_tiles_y = int(math.ceil(height/(TILE_SIZE-tile_gen_overlap)))
    else: # No partial tiles
        num_tiles_x = int(math.floor(width /(TILE_SIZE-tile_gen_overlap)))
        num_tiles_y = int(math.floor(height/(TILE_SIZE-tile_gen_overlap)))
    num_tiles   = num_tiles_x*num_tiles_y
    print('num_tiles_x = ' + str(num_tiles_x))
    print('num_tiles_y = ' + str(num_tiles_y))
    print('tile_gen_overlap = ' + str(tile_gen_overlap))

    # Calculate the number of patches in each tile
    image_bounds = rectangle.Rectangle(0, 0, width=width, height=height)
    output_rois = image_bounds.make_tile_rois(TILE_SIZE, TILE_SIZE,
                                              include_partials=include_partials,
                                              overlap_amount=tile_gen_overlap)
    if num_tiles != len(output_rois):
        print('Tile count mismatch!  This is a bug!')
        print('num_tiles = ' + str(num_tiles))
        print('len(output_rois) = ' + str(len(output_rois)))
        return -1

    tile_patch_dims = []
    check_count   = 0
    num_patches_x = 0
    num_patches_y = 0
    chunk_spacing = chunk_size - chunk_overlap
    chunk_bounds  = chunk_size-1
    full_tile_patches = int(math.floor((TILE_SIZE-chunk_bounds) / chunk_spacing))
    for roi in output_rois:
        x_count = int(math.ceil((roi.width() -chunk_bounds) / chunk_spacing))
        y_count = int(math.ceil((roi.height()-chunk_bounds) / chunk_spacing))
        #print('width = ' + str(roi.width()))
        #print('height = ' + str(roi.height()))
        #print('x_count = ' + str(x_count))
        #print('y_count = ' + str(y_count))

        tile_patch_dims.append((x_count, y_count))
        check_count += x_count * y_count
        if roi.min_y == 0:
            num_patches_x += x_count
        if roi.min_x == 0:
            num_patches_y += y_count


    print('chunk_spacing = ' + str(chunk_spacing))
    print('chunk_bounds = ' + str(chunk_bounds))
    print('num_patches = ' + str(num_patches))
    print('full_tile_patches = ' + str(full_tile_patches))
    print('num_patches_x = ' + str(num_patches_x))
    print('num_patches_y = ' + str(num_patches_y))

    if (check_count != num_patches) or ((num_patches_x*num_patches_y) != num_patches):
        print('Error computing the number of output patches!  This is a bug!')
        #print('num_patches = ' + str(num_patches))
        print('check_count = ' + str(check_count))
        print('num_patches_xy = ' + str(num_patches_x*num_patches_y))
        return -1

    # Convert the single vector of prediction values into the shape of the image
    if options.low_res:
        pic = np.zeros([num_patches_y, num_patches_x], dtype=np.uint8)
        pic_offset = 0
    else: # Match the size of the input image
        pic = np.zeros([height, width], dtype=np.uint8)
        pic_offset = int(math.floor(config.chunk_size()/2.0))

    label_scale = 255 # TODO: Maybe not constant?
    i   = 0
    col = 0
    row = 0
    roi_index = 0
    for tile_row in range(0,num_tiles_y):
        for tile_col in range(0,num_tiles_x):
            roi  = output_rois[roi_index]
            dims = tile_patch_dims[roi_index]
            row  = tile_row*full_tile_patches + pic_offset

            for y in range(0,dims[1]): #pylint: disable=W0612
                col = tile_col*full_tile_patches + pic_offset
                for x in range(0,dims[0]): #pylint: disable=W0612
                    val = predictions[i]
                    pic[row, col] = val * label_scale
                    i += 1
                    col += 1
                row += 1
            roi_index += 1

    print('Writing classified image to: ' + options.output_path)
    tiff.write_simple_image(options.output_path, pic, data_type=gdal.GDT_Byte, metadata=metadata)
    #plt.imsave(options.output_path+'.png', pic) # Generates an RGB false color image

    draw = time.time()
    print('Draw elapsed time = ' + str(draw-stop))

    if options.cleanup:
        os.system('rm -rf ' + options.work_folder)
    return 0


def main(argsIn):

    usage  = "usage: classify_image.py [options]"
    parser = argparse.ArgumentParser(usage=usage)

    parser = argparse.ArgumentParser(usage='train_autoencoder.py [options]')


    parser.add_argument("--keras-model", dest="keras_model", required=True,
                        help="Path to the saved Keras model.")

    parser.add_argument("--input-image", dest="input_image", required=True,
                        help="Path to the input image file.")

    parser.add_argument("--config-path", dest="config_path", required=True,
                        help="Path to the config file used to train the keras model.")

    parser.add_argument("--output-path", dest="output_path", required=True,
                        help="Where to write the output image file.")

    parser.add_argument("--image-type", dest="image_type", required=True,
                        help="Specify image type along with the data folder."
                        +"(landsat, landsat-simple, worldview, or rgba)")

    parser.add_argument("--work-folder", dest="work_folder", default=None,
                        help="Folder to use to store intermediate files."
                        +"Default is an extrapolation of the output path.")

    parser.add_argument("--low-res", action="store_true", dest="low_res", default=False,
                        help="Skip pixels for a faster low resolution classification.")

    parser.add_argument("--cleanup", action="store_true", dest="cleanup", default=False,
                        help="Delete the work folder after running.")

    parser.add_argument("--num-gpus", dest="num_gpus", default=0, type=int,
                        help="Try to use this many GPUs.")

    try:
        options = parser.parse_args(argsIn)
    except argparse.ArgumentError:
        print(usage)
        return -1

    return do_work(options)




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
