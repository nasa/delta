# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Read data in numpy arrays.
"""

import os
from typing import Optional

import numpy as np

from delta.imagery import delta_image

class NumpyImage(delta_image.DeltaImage):
    """
    Load a numpy array as an image.
    """
    def __init__(self, data: Optional[np.ndarray]=None, path: Optional[str]=None, nodata_value=None):
        """
        Parameters
        ----------
        data: Optional[numpy.ndarray]
            Loads a numpy array directly.
        path: Optional[str]
            Load a numpy array from a file with `numpy.load`. Only one of data or path should be given.
        nodata_value
            The pixel value representing no data.
        """
        super().__init__(nodata_value)

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
        if buf is None:
            buf = np.zeros(shape=(roi.height(), roi.width(), self.num_bands() ), dtype=self._data.dtype)
        (min_x, max_x, min_y, max_y) = roi.bounds()
        if len(self._data.shape) == 2:
            buf = self._data[min_y:max_y,min_x:max_x]
        else:
            buf = self._data[min_y:max_y,min_x:max_x,:]
        return buf

    def size(self):
        return (self._data.shape[0], self._data.shape[1])

    def num_bands(self):
        if len(self._data.shape) == 2:
            return 1
        return self._data.shape[2]

    def dtype(self):
        return self._data.dtype

class NumpyWriter(delta_image.DeltaImageWriter):
    def __init__(self):
        self._buffer = None

    def initialize(self, size, numpy_dtype, metadata=None, nodata_value=None):
        self._buffer = np.zeros(shape=size, dtype=numpy_dtype)

    def write(self, data, y, x):
        self._buffer[y:y+data.shape[0], x:x+data.shape[1]] = data

    def close(self):
        pass

    def abort(self):
        pass

    def buffer(self):
        return self._buffer
