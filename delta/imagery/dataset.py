"""
Tools for loading data into the TensorFlow Dataset class.
"""
import functools
import os
import math

import numpy as np
import tensorflow as tf

from . import image_reader
from . import landsat_utils
from . import utilities
from . import disk_folder_cache

# TODO: Generalize
def make_landsat_list(top_folder, output_path, ext, num_regions):
    '''Write a file listing all of the files in a (recursive) folder
       matching the provided extension.
    '''

    num_entries = 0
    with open(output_path, 'w') as f:
        for root, directories, filenames in os.walk(top_folder): #pylint: disable=W0612
            for filename in filenames:
                if os.path.splitext(filename)[1] == ext:
                    path = os.path.join(root, filename)
                    for r in range(0, num_regions):
                        f.write(path + ',' + str(r) +'\n')
                        num_entries += 1
    return num_entries

def get_roi_horiz_band_split(image_size, region, num_splits):
    """Return the ROI of an image to load given the region.
       Each region represents one horizontal band of the image.
    """

    assert region < num_splits, 'Input region ' + str(region) \
           + ' is greater than num_splits: ' + str(num_splits)

    min_x = 0
    max_x = image_size[0]

    # Fractional height here is fine
    band_height = image_size[1] / num_splits

    # TODO: Check boundary conditions!
    min_y = math.floor(band_height*region)
    max_y = math.floor(band_height*(region+1.0))

    return utilities.Rectangle(min_x, min_y, max_x, max_y)


def get_roi_tile_split(image_size, region, num_splits):
    """Return the ROI of an image to load given the region.
       Each region represents one tile in a grid split.
    """
    num_tiles = num_splits*num_splits
    assert region < num_tiles, 'Input region ' + str(region) \
           + ' is greater than num_tiles: ' + str(num_tiles)

    # Convert region index to row and column index
    tile_row = math.floor(region / num_splits)
    tile_col = region % num_splits

    # Fractional sizes are fine here
    tile_width  = math.floor(image_size[0] / num_splits)
    tile_height = math.floor(image_size[1] / num_splits)

    # TODO: Check boundary conditions!
    min_x = math.floor(tile_width  * tile_col)
    max_x = math.floor(tile_width  * (tile_col+1.0))
    min_y = math.floor(tile_height * tile_row)
    max_y = math.floor(tile_height * (tile_row+1.0))

    return utilities.Rectangle(min_x, min_y, max_x, max_y)



def load_image_region(line, prep_function, roi_function, chunk_size, chunk_overlap, num_threads):
    """Load all image chunks for a given region of the image.
       The provided function converts the region to the image ROI.
    """

    # Our input list is stored as "path, region" strings
    line   = line.numpy().decode() # Convert from TF to string type
    parts  = line.split(',')
    path   = parts[0].strip()
    region = int(parts[1].strip())

    # Set up the input image handle
    input_paths  = prep_function(path)
    input_reader = image_reader.MultiTiffFileReader()
    input_reader.load_images(input_paths)
    image_size = input_reader.image_size()

    # Call the provided function to get the ROI to load
    roi = roi_function(image_size, region)

    ## Until we are ready to do a larger test, just return a short vector
    #return np.array([roi.min_x, roi.min_y, roi.max_x, roi.max_y], dtype=np.int32) # DEBUG

    # Load the chunks from inside the ROI
    print('Loading chunk data from file ' + path + ' using ROI: ' + str(roi))
    chunk_data = input_reader.parallel_load_chunks(roi, chunk_size, chunk_overlap, num_threads)

    return chunk_data

def load_fake_labels(line, prep_function, roi_function, chunk_size, chunk_overlap):
    """Use to generate fake label data for load_image_region"""

    # Our input list is stored as "path, region" strings
    #print('Label data input = ' + str(line))
    line   = line.numpy().decode() # Convert from TF format to string
    parts  = line.split(',')
    path   = parts[0].strip()
    region = int(parts[1].strip())

    # Set up the input image handle
    input_paths  = prep_function(path)
    input_reader = image_reader.MultiTiffFileReader()
    input_reader.load_images([input_paths[0]]) # Just the first band
    image_size = input_reader.image_size()

    # Call the provided function to get the ROI to load
    roi = roi_function(image_size, region)

    #return np.array([0, 1, 2, 3], dtype=np.int32) # DEBUG

    # Load the chunks from inside the ROI
    chunk_data = input_reader.parallel_load_chunks(roi, chunk_size, chunk_overlap)

    # Make a fake label
    full_shape = chunk_data.shape[0]
    chunk_data = np.zeros(full_shape, dtype=np.int32)
    chunk_data[ 0:10] = 1 # Junk labels
    chunk_data[10:20] = 2
    return chunk_data


# TODO: Not currently used, but could be if the TF method of filtering chunks is inefficient.
def parallel_filter_chunks(data, num_threads):
    """Filter out chunks that contain the Landsat nodata value (zero)"""

    (num_chunks, num_bands, width, height) = data.shape() #pylint: disable=W0612
    num_chunk_pixels = width * height

    valid_chunks = [True] * num_chunks
    splits = []
    thread_size = float(num_chunks) / float(num_threads)
    for i in range(0,num_threads):
        start_index = math.floor(i    *thread_size)
        stop_index  = math.floor((i+1)*thread_size)
        splits.append((start_index, stop_index))

    # Internal function to flag nodata chunks from the start to stop indices (non-inclusive)
    def check_chunks(pair):
        (start_index, stop_index) = pair
        for i in range(start_index, stop_index):
            chunk = data[i, 0, :, :]
            print(chunk.shape())
            print(chunk)
            if np.count_nonzero(chunk) != num_chunk_pixels:
                valid_chunks[i] = False
                print('INVALID')

    # Call check_chunks in parallel using a thread pool
    pool = ThreadPool(num_threads) #pylint: disable=E0602
    pool.map(check_chunks, splits)
    pool.close()
    pool.join()

    # Remove the bad chunks
    valid_indices = []
    for i in range(0,num_chunks):
        if valid_chunks[i]:
            valid_indices.append(i)

    print('Num remaining chunks = ' + str(len(valid_indices)))

    return data[valid_indices, :, :, :]

class ImageryDataset:
    # TODO: default cache in /tmp
    # TODO: something better with num_regions, chunk_size
    """
    Create dataset with all files in image_folder with extension ext.

    Cache list of files to list_path, and use caching folder cache_folder.
    """
    def __init__(self, image_folder, ext, list_path, cache_folder, cache_limit=4, num_regions=4, chunk_size=256):
        self.__disk_cache_manager = disk_folder_cache.DiskFolderCache(cache_folder, cache_limit)

        # Generate a text file list of all the input images, plus region indices.
        self._num_regions = make_landsat_list(image_folder, list_path, ext, num_regions)
        assert self._num_regions > 0

        # This dataset returns the lines from the text file as entries.
        ds = tf.data.TextLineDataset(list_path)

        # TODO: We can define a different ROI function for each type of input image to
        #       achieve the sizes we want.
        # TODO: These values need to by synchronized with num_regions above!
        row_roi_split_funct = functools.partial(get_roi_horiz_band_split, num_splits=num_regions)

        # This function prepares landsat images and returns the band paths
        ls_prep_func = functools.partial(landsat_utils.prep_landsat_image,
                                         cache_manager=self.__disk_cache_manager)

        # This function loads the data and formats it for TF
        data_load_function = functools.partial(load_image_region,
                                               prep_function=ls_prep_func,
                                               roi_function=row_roi_split_funct,
                                               chunk_size=chunk_size, chunk_overlap=0, num_threads=2)

        # This function generates fake label info for loaded data.
        label_gen_function = functools.partial(load_fake_labels,
                                               prep_function=ls_prep_func,
                                               roi_function=row_roi_split_funct,
                                               chunk_size=chunk_size, chunk_overlap=0)

        def generate_chunks(lines):
            y = tf.py_function(data_load_function, [lines], [tf.float64])
            y[0].set_shape((0, 8, chunk_size, chunk_size))
            return y

        def generate_labels(lines):
            y = tf.py_function(label_gen_function, [lines], [tf.int32])
            y[0].set_shape((0, 1))
            return y

        # Tell TF to use the functions above to load our data.
        chunk_set = ds.map(generate_chunks, num_parallel_calls=1)
        label_set = ds.map(generate_labels, num_parallel_calls=1)

        # Break up the chunk sets to individual chunks
        # TODO: Does this improve things?
        chunk_set = chunk_set.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) #pylint: disable=W0108
        label_set = label_set.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) #pylint: disable=W0108

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((chunk_set, label_set))

        # Filter out all chunks with zero (nodata) values
        self._ds = ds.filter(lambda chunk, label: tf.math.equal(tf.math.zero_fraction(chunk), 0))

    def dataset(self):
        return self._ds

    def num_regions(self):
        return self._num_regions
