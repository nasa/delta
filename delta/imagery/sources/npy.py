"""
Functions to support reading data stored in the Binary Numpy format.
"""

import os
import numpy as np

from . import delta_image

class NumpyImage(delta_image.DeltaImage):
    """
    Numpy image data tensorflow dataset wrapper (see imagery_dataset.py).
    Can set either path to load a file, or data to load a numpy array directly.
    """
    def __init__(self, data=None, path=None):
        super(NumpyImage, self).__init__()

        if path:
            assert not data
            assert os.path.exists(path)
            self._data = np.load(path)
            if len(self._data.shape) == 2:
                self._data = np.expand_dims(self._data, axis=2)
        else:
            assert data is not None
            self._data = data

    def _read(self, roi, bands, buf=None):
        """
        Read the image of the given data type. An optional roi specifies the boundaries.

        This function is intended to be overwritten by subclasses.
        """
        if buf is None:
            buf = np.zeros(shape=(roi.width(), roi.height(), self.num_bands() ), dtype=self._data.dtype)
        (min_x, max_x, min_y, max_y) = roi.get_bounds()
        buf = self._data[min_y:max_y,min_x:max_x,:]
        return buf

    def size(self):
        """Return the size of this image in pixels, as (width, height)."""
        return (self._data.shape[1], self._data.shape[0])

    def num_bands(self):
        """Return the number of bands in the image."""
        return self._data.shape[2]
