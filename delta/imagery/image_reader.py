"""
Classes for block-aligned reading from multiple Geotiff files.
"""
import os
import time
import copy
import math
from multiprocessing.dummy import Pool as ThreadPool

import psutil

from osgeo import gdal
import numpy as np

from delta.imagery import utilities

#------------------------------------------------------------------------------


class TiffReader:
    """Wrapper class to help read image data from GeoTiff files"""

    def __init__(self):
        self._handle = None

    def __del__(self):
        self.close_image()


    def open_image(self, path):
        if not os.path.exists(path):
            raise Exception('Image file does not exist: ' + path)
        self._handle = gdal.Open(path)

    def close_image(self):
        self._handle = None

    # TODO: Error checking!!!

    def num_bands(self):
        return self._handle.RasterCount

    def image_size(self):
        return (self._handle.RasterXSize, self._handle.RasterYSize)

    def nodata_value(self, band=1):
        band_handle = self._handle.GetRasterBand(band)
        return band_handle.GetNoDataValue()

    def get_bytes_per_pixel(self, band=1):
        band_handle = self._handle.GetRasterBand(band)
        return utilities.get_num_bytes_from_gdal_type(band_handle.DataType)

    def get_block_info(self, band):
        """Returns ((block height, block width), (num blocks x, num blocks y))"""
        band_handle = self._handle.GetRasterBand(band)
        block_size  = band_handle.GetBlockSize()

        num_blocks_x = int(math.ceil(self._handle.RasterXSize / block_size[0]))
        num_blocks_y = int(math.ceil(self._handle.RasterYSize / block_size[1]))

        return (block_size, (num_blocks_x, num_blocks_y))

    def get_all_metadata(self):
        """Returns all useful image metadata"""
        d = dict()
        d['projection'  ] = self._handle.GetProjection()
        d['geotransform'] = self._handle.GetGeoTransform()
        d['gcps'        ] = self._handle.GetGCPs()
        d['gcpproj'     ] = self._handle.GetGCPProjection()
        d['metadata'    ] = self._handle.GetMetadata()
        return d

    def get_block_aligned_read_roi(self, desired_roi):
        """Returns the block aligned pixel region to read in a Rectangle format
           to get the requested data region while respecting block boundaries.
        """
        size = self.image_size()
        bounds = utilities.Rectangle(0, 0, width=size[0], height=size[1])
        if not bounds.contains_rect(desired_roi):
            raise Exception('desired_roi ' + str(desired_roi)
                            + ' is outside the bounds of image with size' + str(size))

        band = 1
        (block_size, unused_num_blocks) = self.get_block_info(band)
        start_block_x = int(math.floor(desired_roi.min_x     / block_size[0]))
        start_block_y = int(math.floor(desired_roi.min_y     / block_size[1]))
        stop_block_x  = int(math.floor((desired_roi.max_x-1) / block_size[0])) # Rect max is exclusive
        stop_block_y  = int(math.floor((desired_roi.max_y-1) / block_size[1])) # The stops are inclusive

        start_col = start_block_x * block_size[0]
        start_row = start_block_y * block_size[1]
        num_cols  = (stop_block_x - start_block_x + 1) * block_size[0]
        num_rows  = (stop_block_y - start_block_y + 1) * block_size[1]

        # Restrict the output region to the bounding box of the image.
        # - Needed to handle images with partial tiles at the boundaries.
        ans    = utilities.Rectangle(start_col, start_row, width=num_cols, height=num_rows)
        size   = self.image_size()
        bounds = utilities.Rectangle(0, 0, width=size[0], height=size[1])
        return ans.get_intersection(bounds)

    def read_pixels(self, roi, band):
        """Reads in the requested region of the image."""
        band_handle = self._handle.GetRasterBand(band)

        data = band_handle.ReadAsArray(roi.min_x, roi.min_y, roi.width(), roi.height())
        return data



class MultiTiffFileReader():
    """Class to synchronize loading of multiple pixel-matched files.

    User is going to select a region which is bigger than tiles or chunks,
    then will process all chunks centered in that region.

    Need to make sure that each chunk is only used once, minimize tile reloads.

    Each chunk is associated with a single input pixel, there is a spacing between
    'center' pixels which defines the overlap.  When a region is called for, generate
    each chunk inside that region.  Load tiles as needed.
    Record which regions have already been used.

    """

    def __init__(self):
        self._image_handles = []

    def __del__(self):
        self.close()

    def load_images(self, image_path_list):
        """Initialize with multiple image paths."""

        # Create a new TiffReader instance for each file.
        for path in image_path_list:
            new_handle = TiffReader()
            new_handle.open_image(path)
            self._image_handles.append(new_handle)

    def close(self):
        """Close all loaded images."""
        for h in self._image_handles:
            h.close_image()
        self._image_handles = []

    # TODO: Error checking!
    def image_size(self):
        return self._image_handles[0].image_size()

    def num_bands(self):
        """Get the total number of bands across all input images"""
        b = 0
        for image in self._image_handles:
            b += image.num_bands()
        return b

    def nodata_value(self, band=1):
        return self._image_handles[0].nodata_value(band)

    def get_block_info(self, band):
        return self._image_handles[0].get_block_info(band)

    def get_all_metadata(self):
        """All input images should share the same metadata"""
        return self._image_handles[0].get_all_metadata()

    def estimate_memory_usage(self, roi):
        """Estimate the amount of memory (in MB) that will be used to store the
           requested pixel roi.
        """
        total_size = 0
        for image in self._image_handles:
            for band in range(1,image.num_bands()+1):
                total_size += image.num_bands() * roi.area() * image.get_bytes_per_pixel(band)
        return total_size / utilities.BYTES_PER_MB

    def sleep_until_mem_free_for_roi(self, roi):
        """Sleep until enough free memory exists to load this ROI"""
        # TODO: This will have to be more sophisticated.
        WAIT_TIME_SECS = 5
        mb_needed = self.estimate_memory_usage(roi)
        mb_free   = 0
        while mb_free < mb_needed:
            mb_free = psutil.virtual_memory().free / utilities.BYTES_PER_MB
            if mb_free < mb_needed:
                print('Need %d MB to load the next ROI, but only have %d MB free. Sleep for %d seconds...'
                      % (mb_needed, mb_free, WAIT_TIME_SECS))
                time.sleep(WAIT_TIME_SECS)

    def _get_band_index(self, image_index):
        """Return the absolute band index of the first band in the given image index"""
        b = 0
        for i in range(0,image_index):
            b += self._image_handles[i].num_bands()
        return b

    def parallel_load_chunks(self, roi, chunk_size, chunk_overlap, num_threads=1):
        """Uses multiple threads to populate a numpy data structure with
           image chunks spanning the given roi, formatted for Tensorflow to load.
        """

        # Get image chunk centers
        chunk_info = utilities.generate_chunk_info(chunk_size, chunk_overlap)
        (chunk_center_list, chunk_roi) = \
            utilities.get_chunk_center_list_in_region(roi, chunk_info[0], chunk_info[1], chunk_size)
        if not chunk_center_list:
            raise ValueError('Unable to load any chunks from this ROI!')
            #print('Initial num chunks = ' + str(len(chunk_center_list)))
        #print('Initial chunk ROI = ' + str(chunk_roi))

        # Throw out any partial chunks.
        image_size = self.image_size()
        whole_image_roi = utilities.Rectangle(0,0,width=image_size[0],height=image_size[1])
        (chunk_center_list, chunk_roi) = \
                utilities.restrict_chunk_list_to_roi(chunk_center_list, chunk_size, whole_image_roi)

        num_chunks = len(chunk_center_list)
        if not num_chunks:
            raise Exception('Failed to load any chunks from this ROI!')
        #print('Computed chunks = ' + str(chunk_center_list))
        #print('Final num chunks = ' + str(len(chunk_center_list)))
        #print('Final chunk ROI = ' + str(chunk_roi))

        # Get the block-aligned read ROI (for the larger chunk containing ROI)
        read_roi = self._image_handles[0].get_block_aligned_read_roi(chunk_roi)
        #print('Read ROI = ' + str(read_roi))

        #raise Exception('DEBUG')

        # Adjust centers to be relative to the read ROI.
        offset_centers = [(c[0]-read_roi.min_x,c[1]-read_roi.min_y)
                          for c in chunk_center_list]

        # Allocate the output data structure
        output_shape = (num_chunks, self.num_bands(), chunk_size, chunk_size)
        data_store = np.zeros(shape=output_shape)


        # Internal function to copy all chunks from all bands of one image handle to data_store
        def process_one_image(image_index):
            image      = self._image_handles[image_index]
            band_index = self._get_band_index(image_index) # The first index of one or more sequential bands
            for band in range(1,image.num_bands()+1):
                #print('Data read from image ' + str(image) + ', band ' + str(band))
                this_data = image.read_pixels(read_roi, band)

                # Copy each of the data chunks into the output.
                chunk_index = 0
                for center in offset_centers:
                    rect = utilities.rect_from_chunk_center(center, chunk_size)
                    data_store[chunk_index, band_index, :, :] = this_data[rect.min_y:rect.max_y,
                                                                          rect.min_x:rect.max_x]
                    chunk_index += 1
                band_index += 1

        # Call process_one_image in parallel using a thread pool
        pool = ThreadPool(num_threads)
        pool.map(process_one_image, range(0,len(self._image_handles)))
        pool.close()
        pool.join()
        return data_store


    def process_rois(self, requested_rois, callback_function):
        """Process the given region broken up into blocks using the callback function.
           Each block will get the image data from each input image passed into the function.
           Function definition TBD!
           Blocks that go over the image boundary will be passed as partial image blocks.
           All data reading and function calling takes place in the current thread, to use
           multiple threads you need to hand off the work in the callback function.
        """

        if not self._image_handles:
            raise Exception('Cannot process region, no images loaded!')
        first_image = self._image_handles[0]

        block_rois = copy.copy(requested_rois)

        print('Ready to process ' + str(len(requested_rois)) +' tiles.')

        image_size = self.image_size()
        whole_bounds = utilities.Rectangle(0, 0, width=image_size[0], height=image_size[1])
        for roi in block_rois:
            if not whole_bounds.contains_rect(roi):
                raise Exception('Roi outside image bounds: ' + str(roi))

        # Loop until we have processed all of the blocks.
        while block_rois:
            # For the next (output) block, figure out the (input block) aligned
            # data read that we need to perform to get it.
            read_roi = first_image.get_block_aligned_read_roi(block_rois[0])
            #print('Want to process ROI: ' + str(block_rois[0]))
            #print('Reading input ROI: ' + str(read_roi))
            #sys.stdout.flush()

            # If we don't have enough memory to load the ROI, wait here until we do.
            self.sleep_until_mem_free_for_roi(read_roi)

            # TODO: We need a way to make sure the bands for certain image types always end up
            #       in the same order!
            # Read in all of the data for this region, appending all bands for
            # each input image.
            data_vec = []
            for image in self._image_handles:
                for band in range(1,image.num_bands()+1):
                    #print('Data read from image ' + str(image) + ', band ' + str(band))
                    data_vec.append(image.read_pixels(read_roi, band))

            # Loop through the remaining ROIs and apply the callback function to each
            # ROI that is contained in the section we read in.
            index = 0
            num_processed = 0
            while index < len(block_rois):

                roi = block_rois[index]
                if not read_roi.contains_rect(roi):
                    #print(read_roi + ' does not contain ' + )
                    index += 1
                    continue

                # We pass the read roi to the function rather than doing
                # any kind of cropping here.

                # Execute the callback function with the data vector.
                callback_function(roi, read_roi, data_vec)

                # Instead of advancing the index, remove the current ROI from the list.
                block_rois.pop(index)
                num_processed += 1

            #print('From the read ROI, was able to process ' + str(num_processed) +' tiles.')

        print('Finished processing tiles!')
