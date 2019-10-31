"""
Support for TIFF imagery.
"""

from abc import ABC, abstractmethod
import math
import os

from delta.imagery import image_reader
from delta.imagery import rectangle

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

    return rectangle.Rectangle(min_x, min_y, max_x, max_y)

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

    return rectangle.Rectangle(min_x, min_y, max_x, max_y)


class DeltaImage(ABC):
    """Base class used for wrapping input images in a way that they can be passed
       to Tensorflow dataset objects.
    """

    DEFAULT_EXTENSIONS = ['.tif']

    # Constants which must be specified for all image types, these are the default values.
    def __init__(self, num_regions):
        self._num_regions = num_regions

    @abstractmethod
    def read(self, roi=None):
        """
        Read the image of the given data type. An optional roi specifies the boundaries.
        """

    @abstractmethod
    def size(self):
        """Return the size of this image in pixels"""

    @abstractmethod
    def num_bands(self):
        """Return the number of bands in the image"""

    def tiles(self):
        """Generator to yield ROIs for the image."""
        s = self.size()
        # TODO: add to config, replace with max buffer size?
        for i in range(self._num_regions):
            yield horizontal_split(s, i, self._num_regions)

class TiffImage(DeltaImage):
    """For all versions of DeltaImage that can use our image_reader class"""

    DEFAULT_EXTENSIONS = ['.tif']

    def __init__(self, path, cache_manager, num_regions):
        super(TiffImage, self).__init__(num_regions)
        self.path = path
        self._cache_manager = cache_manager

    def prep(self):
        """Prepare the file to be opened by other tools (unpack, etc)"""
        return [self.path]

    def num_bands(self):
        """Return the number of bands in a prepared file"""
        input_paths = self.prep()

        input_reader = image_reader.MultiTiffFileReader()
        input_reader.load_images(input_paths)
        return input_reader.num_bands()

    def read(self, roi=None):
        input_paths = self.prep()

        # Set up the input image handle
        input_reader = image_reader.MultiTiffFileReader()
        input_reader.load_images(input_paths)
        return input_reader.read_roi(roi)

    def size(self):
        input_paths = self.prep()

        input_reader = image_reader.MultiTiffFileReader()
        input_reader.load_images(input_paths)
        return input_reader.image_size()

class RGBAImage(TiffImage):
    """Basic RGBA images where the alpha channel needs to be stripped"""

    DEFAULT_EXTENSIONS = ['.tif']

    def prep(self):
        """Converts RGBA images to RGB images"""

        # Get the path to the cached image
        fname = os.path.basename(self.path)
        output_path = self._cache_manager.register_item(fname)

        if not os.path.exists(output_path):
            # Just remove the alpha band from the original image
            cmd = 'gdal_translate -b 1 -b 2 -b 3 ' + self.path + ' ' + output_path
            print(cmd)
            os.system(cmd)
        return [output_path]
