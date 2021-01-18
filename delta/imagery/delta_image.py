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
Base class for loading images.
"""

from abc import ABC, abstractmethod
import concurrent.futures
import copy
import functools
from typing import Callable, Iterator, List, Tuple

import numpy as np

from delta.imagery import rectangle, utilities

class DeltaImage(ABC):
    """
    Base class used for wrapping input images in a way that they can be passed
    to Tensorflow dataset objects.
    """
    def __init__(self, nodata_value=None):
        self.__preprocess_function = None
        self.__nodata_value = nodata_value

    def read(self, roi: rectangle.Rectangle=None, bands: List[int]=None, buf: np.ndarray=None) -> np.ndarray:
        """
        Reads the image in [row, col, band] indexing.

        If `roi` is not specified, reads the entire image.
        If `buf` is specified, writes the image to buf.
        If `bands` is not specified, reads all bands, otherwise
        only the listed bands are read.
        If bands is a single integer, drops the band dimension.
        """
        if roi is None:
            roi = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        else:
            if roi.min_x < 0 or roi.min_y < 0 or roi.max_x > self.width() or roi.max_y > self.height():
                raise IndexError('Rectangle (%d, %d, %d, %d) outside of bounds (%d, %d).' %
                                 (roi.min_x, roi.min_y, roi.max_x, roi.max_y, self.width(), self.height()))
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

    def set_preprocess(self, callback: Callable[[np.ndarray, rectangle.Rectangle, List[int]], np.ndarray]) -> None:
        """
        Set a preproprocessing function callback to be applied to the results of all reads on the image.

        The function takes the arguments callback(image, roi, bands), where image is the numpy array containing
        the read data, roi is the region of interest read, and bands is a list of the bands being read.
        """
        self.__preprocess_function = callback

    def get_preprocess(self):
        """
        Returns the preprocess function.
        """
        return self.__preprocess_function

    def nodata_value(self):
        """
        Returns the value of pixels to treat as nodata.
        """
        return self.__nodata_value

    @abstractmethod
    def _read(self, roi, bands, buf=None):
        """
        Read the image of the given data type. An optional roi specifies the boundaries.

        This function is intended to be overwritten by subclasses.
        """

    def metadata(self): #pylint:disable=no-self-use
        """
        Returns a dictionary of metadata, in the format used by GDAL.
        """
        return {}

    @abstractmethod
    def size(self) -> Tuple[int, int]:
        """Return the size of this image in pixels, as (width, height)."""

    @abstractmethod
    def num_bands(self) -> int:
        """Return the number of bands in the image."""

    def block_aligned_roi(self, desired_roi: rectangle.Rectangle) -> rectangle.Rectangle:#pylint:disable=no-self-use
        """Return the block-aligned roi containing this image region, if applicable."""
        return desired_roi

    def block_size(self): #pylint: disable=no-self-use
        """Return the preferred block size for efficient reading."""
        return (256, 256)

    def width(self) -> int:
        """Return the number of columns."""
        return self.size()[0]

    def height(self) -> int:
        """Return the number of rows."""
        return self.size()[1]

    def tiles(self, shape, overlap_shape=(0, 0), partials: bool=True, min_shape=(0, 0),
              partials_overlap: bool=False, by_block=False, offset=(0, 0)) -> Iterator[rectangle.Rectangle]:
        """Generator to yield ROIs for the image."""
        input_bounds = rectangle.Rectangle(offset[0], offset[1], max_x=self.width(), max_y=self.height())
        return input_bounds.make_tile_rois(shape, overlap_shape=overlap_shape, include_partials=partials,
                                           min_shape=min_shape, partials_overlap=partials_overlap,
                                           by_block=by_block)

    def roi_generator(self, requested_rois: Iterator[rectangle.Rectangle]) -> Iterator[rectangle.Rectangle]:
        """
        Generator that yields ROIs of blocks in the requested region.
        """
        block_rois = copy.copy(requested_rois)

        whole_bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        for roi in requested_rois:
            if not whole_bounds.contains_rect(roi):
                raise Exception('Roi outside image bounds: ' + str(roi) + str(whole_bounds))

        # gdal doesn't work reading multithreading. But this let's a thread
        # take care of IO input while we do computation.
        jobs = []

        total_rois = len(block_rois)
        while block_rois:
            # For the next (output) block, figure out the (input block) aligned
            # data read that we need to perform to get it.
            read_roi = self.block_aligned_roi(block_rois[0])

            applicable_rois = []

            # Loop through the remaining ROIs and apply the callback function to each
            # ROI that is contained in the section we read in.
            index = 0
            while index < len(block_rois):

                if not read_roi.contains_rect(block_rois[index]):
                    index += 1
                    continue
                applicable_rois.append(block_rois.pop(index))

            jobs.append((read_roi, applicable_rois))

        # only do a few reads ahead since otherwise we will exhaust our memory
        pending = []
        exe = concurrent.futures.ThreadPoolExecutor(1)
        NUM_AHEAD = 2
        for i in range(min(NUM_AHEAD, len(jobs))):
            pending.append(exe.submit(functools.partial(self.read, jobs[i][0])))
        num_remaining = total_rois
        for (i, (read_roi, rois)) in enumerate(jobs):
            buf = pending.pop(0).result()
            for roi in rois:
                x0 = roi.min_x - read_roi.min_x
                y0 = roi.min_y - read_roi.min_y
                num_remaining -= 1
                yield (roi, buf[x0:x0 + roi.width(), y0:y0 + roi.height(), :],
                       (total_rois - num_remaining, total_rois))
            if i + NUM_AHEAD < len(jobs):
                pending.append(exe.submit(functools.partial(self.read, jobs[i + NUM_AHEAD][0])))

    def process_rois(self, requested_rois: Iterator[rectangle.Rectangle],
                     callback_function: Callable[[rectangle.Rectangle, np.ndarray], None],
                     show_progress: bool=False) -> None:
        """
        Process the given region broken up into blocks using the callback function.
        Each block will get the image data from each input image passed into the function.
        Data reading takes place in a separate thread, but the callbacks are executed
        in a consistent order on a single thread.
        """
        for (roi, buf, (i, total)) in self.roi_generator(requested_rois):
            callback_function(roi, buf)
            if show_progress:
                utilities.progress_bar('%d / %d' % (i, total), i / total, prefix='Blocks Processed:')
        if show_progress:
            print()

class DeltaImageWriter(ABC):
    @abstractmethod
    def initialize(self, size, numpy_dtype, metadata=None, nodata_value=None):
        """
        Prepare for writing with the given size and dtype.
        """

    @abstractmethod
    def write(self, data, x, y):
        """
        Writes the data as a rectangular block starting at the given coordinates.
        """

    @abstractmethod
    def close(self):
        """
        Finish writing.
        """

    @abstractmethod
    def abort(self):
        """
        Cancel writing before finished.
        """

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
        return False
