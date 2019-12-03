"""
Support for TIFF imagery.
"""

from abc import ABC, abstractmethod
import math

from delta.config import config
from delta.imagery import rectangle

class DeltaImage(ABC):
    """Base class used for wrapping input images in a way that they can be passed
       to Tensorflow dataset objects.
    """
    def __init__(self):
        self.__preprocess_function = None

    def read(self, roi=None, bands=None, buf=None):
        """
        Reads in the requested region of the image.

        If roi is not specified, reads the entire image.
        If buf is specified, writes the image to buf.
        If bands is not specified, reads all bands in [row, col, band] indexing. Otherwise
        only the listed bands are read.
        If bands is a single integer, drops the band dimension.
        """
        if roi is None:
            roi = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        if bands is None:
            bands = range(self.num_bands())
        if isinstance(bands, int):
            result = self._read(roi, [bands], buf)
            result = result[:, :, 0] # reduce dimensions
        else:
            result = self._read(roi, bands, buf)
        if self.__preprocess_function:
            return self.__preprocess_function(result, roi, bands)
        return result

    def set_preprocess(self, callback):
        """
        Set a preproprocessing function callback to be applied to the results of all reads on the image.

        The function takes the arguments callback(image, roi, bands), where image is the numpy array containing
        the read data, roi is the region of interest read, and bands is a list of the bands being read.
        """
        self.__preprocess_function = callback

    @abstractmethod
    def _read(self, roi, bands, buf=None):
        """
        Read the image of the given data type. An optional roi specifies the boundaries.

        This function is intended to be overwritten by subclasses.
        """

    @abstractmethod
    def size(self):
        """Return the size of this image in pixels, as (width, height)."""

    @abstractmethod
    def num_bands(self):
        """Return the number of bands in the image."""

    def width(self):
        """Return the number of columns."""
        return self.size()[0]

    def height(self):
        """Return the number of rows."""
        return self.size()[1]

    def tiles(self):
        """Generator to yield ROIs for the image."""
        max_block_bytes = config.dataset().max_block_size() * 1024 * 1024
        s = self.size()
        # TODO: account for image type
        image_bytes = s[0] * s[1] * self.num_bands() * 4
        num_regions = math.ceil(image_bytes / max_block_bytes)

        input_bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        return input_bounds.make_tile_rois(self.width() // num_regions, self.height(), include_partials=True)
