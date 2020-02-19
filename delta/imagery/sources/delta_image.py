"""
Support for TIFF imagery.
"""

from abc import ABC, abstractmethod
import concurrent.futures
import copy
import functools

from delta.imagery import rectangle, utilities

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

    def block_aligned_roi(self, desired_roi):#pylint:disable=no-self-use
        """Return the block-aligned roi containing this image region, if applicable."""
        return desired_roi

    def width(self):
        """Return the number of columns."""
        return self.size()[0]

    def height(self):
        """Return the number of rows."""
        return self.size()[1]

    def tiles(self, width, height, min_width=0, min_height=0, overlap=0):
        """Generator to yield ROIs for the image."""
        input_bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        return input_bounds.make_tile_rois(width, height, min_width=min_width, min_height=min_height,
                                           include_partials=True, overlap_amount=overlap)

    def roi_generator(self, requested_rois):
        block_rois = copy.copy(requested_rois)

        whole_bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        for roi in requested_rois:
            if not whole_bounds.contains_rect(roi):
                raise Exception('Roi outside image bounds: ' + str(roi) + str(whole_bounds))

        # gdal doesn't work reading multithreading. But this let's a thread
        # take care of IO input while we do computation.
        exe = concurrent.futures.ThreadPoolExecutor(1)
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

            buf = exe.submit(functools.partial(self.read, read_roi))
            jobs.append((buf, read_roi, applicable_rois))

        num_remaining = total_rois
        for (buf_exe, read_roi, rois) in jobs:
            buf = buf_exe.result()
            for roi in rois:
                x0 = roi.min_x - read_roi.min_x
                y0 = roi.min_y - read_roi.min_y
                num_remaining -= 1
                yield (roi, buf[x0:x0 + roi.width(), y0:y0 + roi.height(), :],
                       (total_rois - num_remaining, total_rois))

    def process_rois(self, requested_rois, callback_function, show_progress=False):
        '''
        Process the given region broken up into blocks using the callback function.
        Each block will get the image data from each input image passed into the function.
        Data reading takes place in a separate thread, but the callbacks are executed
        in a consistent order on a single thread.
        '''
        for (roi, buf, (i, total)) in self.roi_generator(requested_rois):
            callback_function(roi, buf)
            if show_progress:
                utilities.progress_bar('%d / %d' % (i, total), i / total, prefix='Blocks Processed:')
        if show_progress:
            print()
