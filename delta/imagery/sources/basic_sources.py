"""
Support for TIFF imagery.
"""

from abc import ABC, abstractmethod
import math
from multiprocessing.dummy import Pool as ThreadPool
import os

import numpy as np

from delta import config
from delta.imagery import image_reader
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


class DeltaImage:
    """Base class used for wrapping input images in a way that they can be passed
       to Tensorflow dataset objects.
    """

    # Constants which must be specified for all image types, these are the default values.
    _NUM_REGIONS = 4
    DEFAULT_EXTENSIONS = ['.tif']
    CACHE_CLASS = None

    @abstractmethod
    def chunk_image_region(self, roi, chunk_size, chunk_overlap):
        """Return this portion of the image broken up into small segments.
           The output format is a numpy array of size [N, num_bands, chunk_size, chunk_size]
        """
        pass

    @abstractmethod
    def size(self):
        """Return the size of this image in pixels"""
        pass

    @abstractmethod
    def prep(self):
        """Prepare the file to be opened by other tools (unpack, etc)"""
        pass

    @abstractmethod
    def get_num_bands(self, sample_path):
        """Return the number of bands in the image"""
        pass

    def tiles(self):
        """Generator to yield ROIs for the image."""
        s = self.size()
        # TODO: add to config, replace with max buffer size?
        for i in range(self._NUM_REGIONS):
            yield horizontal_split(s, i, self._NUM_REGIONS)


class TiffImage(DeltaImage):
    """For all versions of DeltaImage that can use our image_reader class"""
  
    DEFAULT_EXTENSIONS = ['.tif']
  
    def __init__(self, path, cache_manager):
        self.path = path
        self.cache_manager = cache_manager

    @abstractmethod
    def prep(self):
        pass

    def get_num_bands(self):
        """Return the number of bands in a prepared file"""
        input_paths = self.prep()

        input_reader = image_reader.MultiTiffFileReader()
        input_reader.load_images(input_paths)
        return input_reader.num_bands()

    def chunk_image_region(self, roi, chunk_size, chunk_overlap):
        # First make sure that the image is unpacked and ready to load
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


class SimpleTiff(TiffImage):
    """A basic image which comes ready to use"""
    _NUM_REGIONS = 1
    DEFAULT_EXTENSIONS = ['.tif']
    CACHE_CLASS = disk_folder_cache.DiskFileCache
    def prep(self):
        return self.path


class RGBAImage(TiffImage):
    """Basic RGBA images where the alpha channel needs to be stripped"""

    _NUM_REGIONS = 1
    DEFAULT_EXTENSIONS = ['.tif']
    CACHE_CLASS = disk_folder_cache.DiskFileCache
    def prep(self):
        """Converts RGBA images to RGB images"""

        # Get the path to the cached image
        fname = os.path.basename(self.path)
        output_path = self.cache_manager.register_item(fname)

        if not os.path.exists(output_path):
            # Just remove the alpha band from the original image
            cmd = 'gdal_translate -b 1 -b 2 -b 3 ' + self.path + ' ' + output_path
            print(cmd)
            os.system(cmd)
        return [output_path]

