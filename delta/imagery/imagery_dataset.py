"""
Tools for loading data into the TensorFlow Dataset class.
"""
from abc import ABC, abstractmethod
import functools
import os
import os.path
import math
import tempfile

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import tensorflow as tf

from delta import config
from delta.imagery import image_reader
from delta.imagery import landsat_utils
from delta.imagery import worldview_utils
from delta.imagery import utilities

# TODO: Not currently used, but could be if the TF method of filtering chunks is inefficient.
def parallel_filter_chunks(data, num_threads):
    """Filter out chunks that contain the Landsat nodata value (zero)"""

    (num_chunks, unused_num_bands, width, height) = data.shape()
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
    pool = ThreadPool(num_threads)
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

def horizontal_split(image_size, region, num_splits):
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

def tile_split(image_size, region, num_splits):
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

# TODO: Clean all of this up!
class DeltaImage(ABC):
    NUM_REGIONS = 4
    
    @abstractmethod
    def chunk_image_region(self, roi, chunk_size, chunk_overlap):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def prep(self):
        pass

    def tiles(self):
        s = self.size()
        # TODO: add to config, replace with max buffer size?
        for i in range(self.NUM_REGIONS):
            yield horizontal_split(s, i, self.NUM_REGIONS)

class TiffImage(DeltaImage):
    def __init__(self, path):
        self.path = path

    def chunk_image_region(self, roi, chunk_size, chunk_overlap):
        input_paths = self.prep()

        # Set up the input image handle
        input_reader = image_reader.MultiTiffFileReader()
        input_reader.load_images(input_paths)

        # Load the chunks from inside the ROI
        #print('Loading chunk data from file ' + self.path + ' using ROI: ' + str(roi))
        # TODO: configure number of threads
        chunk_data = input_reader.parallel_load_chunks(roi, chunk_size, chunk_overlap, 1)

        return chunk_data

    def size(self):
        input_paths = self.prep()

        input_reader = image_reader.MultiTiffFileReader()
        input_reader.load_images(input_paths)
        return input_reader.image_size()

class LandsatImage(TiffImage):
    NUM_REGIONS = 4
    def prep(self):
        return landsat_utils.prep_landsat_image(self.path)

class WorldviewImage(TiffImage):
    NUM_REGIONS = 10 # May be too small
    def prep(self):
        return worldview_utils.prep_worldview_image(self.path)

class RGBAImage(TiffImage):
    NUM_REGIONS = 1
    def prep(self):
        """Converts RGBA images to RGB images.
           WARNING: This function does never deletes cached images!
        """
        # Check if we already converted this image
        fname = os.path.basename(self.path)
        output_path = os.path.join(config.cache_dir(), fname)

        if not os.path.exists(output_path):
            # Just remove the alpha band
            cmd = 'gdal_translate -b 1 -b 2 -b 3 ' + self.path + ' ' + output_path
            print(cmd)
            os.system(cmd)
        return [output_path]

IMAGE_CLASSES = {
        'landsat' : LandsatImage,
        'worldview' : WorldviewImage,
        'rgba' : RGBAImage
}

class ImageryDataset:
    # TODO: something better with num_regions, chunk_size
    # TODO: Need to clean up this whole class!
    """
    Create dataset with all files in image_folder with extension ext.

    Cache list of files to list_path, and use caching folder cache_folder.
    """
    def __init__(self, image_type, image_folder=None, chunk_size=256,
                 list_path=None):

        # TODO: Merge with the classes above!
        # TODO: May need to pass in this value!
        num_bands_dict = {'landsat':8, 'worldview':8, 'tif':3, 'rgba':3}
        try:
            num_bands = num_bands_dict[image_type]
        except IndexError:
            raise Exception('Unrecognized input type: ' + image_type)

        # Figure out the image file extension
        ext_dict = {'landsat':'.gz', 'worldview':'.zip', 'tif':'.tif', 'rgba':'.tif'}
        try:
            ext = ext_dict[image_type]
        except IndexError:
            raise Exception('Unrecognized input type: ' + image_type)

        if list_path is None:
            tempfd, list_path = tempfile.mkstemp()
            os.close(tempfd)
            self.__temp_path = list_path
        else:
            self.__temp_path = None

        self._image_type = image_type
        self._chunk_size = chunk_size
        self._chunk_overlap = 0 # TODO: MAKE AN OPTION!

        # Generate a text file list of all the input images, plus region indices.
        self._num_regions, self._num_images = self.__make_image_list(image_folder, list_path, ext)
        assert self._num_regions > 0
        

        # This dataset returns the lines from the text file as entries.
        ds = tf.data.TextLineDataset(list_path)

        # This function generates fake label info for loaded data.
        label_gen_function = functools.partial(self.__load_fake_labels)

        def generate_chunks(lines):
            y = tf.py_function(self.__load_data, [lines], [tf.float64])
            y[0].set_shape((0, num_bands, chunk_size, chunk_size))
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
        chunk_set = chunk_set.flat_map(tf.data.Dataset.from_tensor_slices)
        label_set = label_set.flat_map(tf.data.Dataset.from_tensor_slices)

        # Pair the data and labels in our dataset
        ds = tf.data.Dataset.zip((chunk_set, label_set))

        # Filter out all chunks with zero (nodata) values
        self._ds = ds.filter(lambda chunk, label: tf.math.equal(tf.math.zero_fraction(chunk), 0))

    def __del__(self):
        if self.__temp_path is not None:
            os.remove(self.__temp_path)

    def image_class(self):
        return IMAGE_CLASSES[self._image_type]

    def __load_data(self, text_line):
        text_line = text_line.numpy().decode() # Convert from TF to string type
        parts  = text_line.split(',')
        path   = parts[0].strip()
        roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
                                  int(parts[3].strip()), int(parts[4].strip()))

        image = self.image_class()(path)
        return image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap)

    # TODO: delete this and load actual labels
    def __load_fake_labels(self, text_line):
        """Use to generate fake label data for load_image_region"""
        text_line = text_line.numpy().decode() # Convert from TF to string type
        parts  = text_line.split(',')
        path   = parts[0].strip()
        roi = utilities.Rectangle(int(parts[1].strip()), int(parts[2].strip()),
                                  int(parts[3].strip()), int(parts[4].strip()))

        image = self.image_class()(path)
        chunk_data = image.chunk_image_region(roi, self._chunk_size, self._chunk_overlap)

        # Make a fake label
        full_shape = chunk_data.shape[0]
        chunk_data = np.zeros(full_shape, dtype=np.int32)
        chunk_data[ 0:10] = 1 # Junk labels
        chunk_data[10:20] = 2
        return chunk_data

    def __make_image_list(self, top_folder, output_path, ext):
        '''Write a file listing all of the files in a (recursive) folder
           matching the provided extension.
        '''

        num_entries = 0
        num_images = 0
        with open(output_path, 'w') as f:
            for root, dummy_directories, filenames in os.walk(top_folder):
                for filename in filenames:
                    if os.path.splitext(filename)[1] == ext:
                        path = os.path.join(root, filename)
                        rois = self.image_class()(path).tiles()
                        for r in rois:
                            f.write('%s,%d,%d,%d,%d\n' % (path, r.min_x, r.min_y, r.max_x, r.max_y))
                            num_entries += 1
                    num_images += 1
        return num_entries, num_images

    def dataset(self):
        return self._ds

    def num_images(self):
        return
    def total_num_regions(self):
        return self._num_regions
