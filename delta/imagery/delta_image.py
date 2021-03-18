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
Base classes for reading and writing images.
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
    Base class used for wrapping input images in DELTA. Can be extended
    to support new data types. A variety of image types are implemented in
    `delta.extensions.sources`.
    """
    def __init__(self, nodata_value=None):
        """
        Parameters
        ----------
        nodata_value: Optional[Any]
            Nodata value for the image, if any.
        """
        self.__preprocess_function = None
        self.__nodata_value = nodata_value

    def read(self, roi: rectangle.Rectangle=None, bands: List[int]=None, buf: np.ndarray=None) -> np.ndarray:
        """
        Reads the image in [row, col, band] indexing.

        Subclasses should generally not overwrite this method--- they will likely want to implement
        `_read`.

        Parameters
        ----------
        roi: `rectangle.Rectangle`
            The bounding box to read from the image. If None, read the entire image.
        bands: List[int]
            Bands to load (zero-indexed). If None, read all bands.
        buf: np.ndarray
            If specified, reads the image into this buffer. Must be sufficiently large.

        Returns
        -------
        np.ndarray:
            A buffer containing the requested part of the image.
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

    def set_preprocess(self, callback: Callable[[np.ndarray, rectangle.Rectangle, List[int]], np.ndarray]):
        """
        Set a preproprocessing function callback to be applied to the results of
        all reads on the image.

        Parameters
        ----------
        callback: Callable[[np.ndarray, rectangle.Rectangle, List[in]], np.ndarray]
            A function to be called on loading image data, callback(image, roi, bands),
            where `image` is the numpy array containing the read data, `roi` is the region of interest read,
            and `bands` is a list of the bands read. Must return a numpy array.
        """
        self.__preprocess_function = callback

    def get_preprocess(self) -> Callable[[np.ndarray, rectangle.Rectangle, List[int]], np.ndarray]:
        """
        Returns
        -------
        Callable[[np.ndarray, rectangle.Rectangle, List[int]], np.ndarray]
            The preprocess function currently set.
        """
        return self.__preprocess_function

    def nodata_value(self):
        """
        Returns
        -------
        The value of pixels to treat as nodata.
        """
        return self.__nodata_value

    @abstractmethod
    def _read(self, roi: rectangle.Rectangle, bands: List[int], buf: np.ndarray=None) -> np.ndarray:
        """
        Read the image.

        Abstract function to be implemented by subclasses. Users should call `read` instead.

        Parameters
        ----------
        roi: rectangle.Rectangle
            Segment of the image to read.
        bands: List[int]
            List of bands to read (zero-indexed).
        buf: np.ndarray
            Buffer to read into. If not specified, a new buffer should be allocated.

        Returns
        -------
        np.ndarray:
            The relevant part of the image as a numpy array.
        """

    def metadata(self): #pylint:disable=no-self-use
        """
        Returns
        -------
        A dictionary of metadata, if any is given for the image type.
        """
        return {}

    @abstractmethod
    def size(self) -> Tuple[int, int]:
        """
        Returns
        -------
        Tuple[int, int]:
            The size of this image in pixels, as (width, height).
        """

    @abstractmethod
    def num_bands(self) -> int:
        """
        Returns
        -------
        int:
            The number of bands in this image.
        """

    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        numpy.dtype:
            The underlying data type of the image.
        """

    def block_aligned_roi(self, desired_roi: rectangle.Rectangle) -> rectangle.Rectangle:#pylint:disable=no-self-use
        """
        Parameters
        ----------
        desired_roi: rectangle.Rectangle
            Original region of interest.

        Returns
        -------
        rectangle.Rectangle:
            The block-aligned roi containing the specified roi.
        """
        return desired_roi

    def block_size(self): #pylint: disable=no-self-use
        """
        Returns
        -------
        (int, int):
            The suggested block size for efficient reading.
        """
        return (256, 256)

    def width(self) -> int:
        """
        Returns
        -------
        int:
            The number of image columns
        """
        return self.size()[0]

    def height(self) -> int:
        """
        Returns
        -------
        int:
            The number of image rows
        """
        return self.size()[1]

    def tiles(self, shape, overlap_shape=(0, 0), partials: bool=True, min_shape=(0, 0),
              partials_overlap: bool=False, by_block=False):
        """
        Splits the image into tiles with the given properties.

        Parameters
        ----------
        shape: (int, int)
            Shape of each tile
        overlap_shape: (int, int)
            Amount to overlap tiles in x and y direction
        partials: bool
            If true, include partial tiles at the edge of the image.
        min_shape: (int, int)
            If true and `partials` is true, keep partial tiles of this minimum size.
        partials_overlap: bool
            If `partials` is false, and this is true, expand partial tiles
            to the desired size. Tiles may overlap in some areas.
        by_block: bool
            If true, changes the returned generator to group tiles by block.
            This is intended to optimize disk reads by reading the entire block at once.

        Returns
        -------
        List[Rectangle] or List[(Rectangle, List[Rectangle])]
            List of ROIs. If `by_block` is true, returns a list of (Rectangle, List[Rectangle])
            instead, where the first rectangle is a larger block containing multiple tiles in a list.
        """
        input_bounds = rectangle.Rectangle(0, 0, max_x=self.width(), max_y=self.height())
        return input_bounds.make_tile_rois(shape, overlap_shape=overlap_shape, include_partials=partials,
                                           min_shape=min_shape, partials_overlap=partials_overlap,
                                           by_block=by_block)

    def roi_generator(self, requested_rois: Iterator[rectangle.Rectangle]) -> \
                  Iterator[Tuple[rectangle.Rectangle, np.ndarray, int, int]]:
        """
        Generator that yields image blocks of the requested rois.

        Parameters
        ----------
        requested_rois: Iterator[Rectangle]
            Regions of interest to read.

        Returns
        -------
        Iterator[Tuple[Rectangle, numpy.ndarray, int, int]]
            A generator with read image regions. In each tuple, the first item
            is the region of interest, the second is a numpy array of the image contents,
            the third is the index of the current region of interest, and the fourth is the total
            number of rois.
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
        Apply a callback function to a list of ROIs.

        Parameters
        ----------
        requested_rois: Iterator[Rectangle]
            Regions of interest to evaluate
        callback_function: Callable[[rectangle.Rectangle, np.ndarray], None]
            A function to apply to each requested region. Pass the bounding box
            of the current region and a numpy array of pixel values as inputs.
        show_progress: bool
            Print a progress bar on the command line if true.
        """
        for (roi, buf, (i, total)) in self.roi_generator(requested_rois):
            callback_function(roi, buf)
            if show_progress:
                utilities.progress_bar('%d / %d' % (i, total), i / total, prefix='Blocks Processed:')
        if show_progress:
            print()

class DeltaImageWriter(ABC):
    """
    Base class for writing images in DELTA.
    """
    @abstractmethod
    def initialize(self, size, numpy_dtype, metadata=None, nodata_value=None):
        """
        Prepare for writing.

        Parameters
        ----------
        size: tuple of ints
            Dimensions of the image to write.
        numpy_dtype: numpy.dtype
            Type of the underling data.
        metadata: dict
            Dictionary of metadata to save with the image.
        nodata_value: numpy_dtype
            Value representing nodata in the image.
        """

    @abstractmethod
    def write(self, data: np.ndarray, x: int, y: int):
        """
        Write a portion of the image.

        Parameters
        ----------
        data: np.ndarray
            A block of image data to write.
        x: int
        y: int
            Top-left coordinates of the block of data to write.
        """

    @abstractmethod
    def close(self):
        """
        Finish writing, perform cleanup.
        """

    @abstractmethod
    def abort(self):
        """
        Cancel writing before finished, perform cleanup.
        """

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
        return False
