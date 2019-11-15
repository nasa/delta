"""
Classes for block-aligned reading from multiple Geotiff files.
"""
import os
import copy
import math

from osgeo import gdal
import numpy as np

from delta.config import config
from delta.imagery import rectangle, utilities

from . import basic_sources

class TiffImage(basic_sources.DeltaImage):
    """For geotiffs."""

    def __init__(self, path):
        super(TiffImage, self).__init__(path)
        self.path = path

    def prep(self):
        """Prepare the file to be opened by other tools (unpack, etc)"""
        return [self.path]

    def num_bands(self):
        """Return the number of bands in a prepared file"""
        input_paths = self.prep()

        input_reader = MultiTiffFileReader()
        input_reader.load_images(input_paths)
        return input_reader.num_bands()

    def read(self, roi=None):
        input_paths = self.prep()

        # Set up the input image handle
        input_reader = MultiTiffFileReader()
        input_reader.load_images(input_paths)
        return input_reader.read_roi(roi)

    def size(self):
        input_paths = self.prep()

        input_reader = MultiTiffFileReader()
        input_reader.load_images(input_paths)
        return input_reader.image_size()

class RGBAImage(TiffImage):
    """Basic RGBA images where the alpha channel needs to be stripped"""

    def prep(self):
        """Converts RGBA images to RGB images"""

        # Get the path to the cached image
        fname = os.path.basename(self.path)
        output_path = config.cache_manager().register_item(fname)

        if not os.path.exists(output_path):
            # Just remove the alpha band from the original image
            cmd = 'gdal_translate -b 1 -b 2 -b 3 ' + self.path + ' ' + output_path
            print(cmd)
            os.system(cmd)
        return [output_path]

#------------------------------------------------------------------------------

class TiffReader:
    """Wrapper class to help read image data from GeoTiff files"""

    def __init__(self):
        self._handle = None

    def __del__(self):
        self.close_image()


    def open_image(self, path):
        '''
        Opens an image if it exists
        '''
        if not os.path.exists(path):
            raise Exception('Image file does not exist: ' + path)
        self._handle = gdal.Open(path)

    def close_image(self):
        '''
        Closes an image
        '''
        self._handle = None

    # TODO: Error checking!!!

    def num_bands(self):
        '''
        Gets number of bands.
        '''
        return self._handle.RasterCount

    def image_size(self):
        '''
        Gets image size
        '''
        return (self._handle.RasterXSize, self._handle.RasterYSize)

    def nodata_value(self, band=1):
        '''
        returns the value that indicates no data is present in a pixel for a band.
        '''
        band_handle = self._handle.GetRasterBand(band)
        return band_handle.GetNoDataValue()

    def data_type(self, band=1):
        """Returns the GDAL data type of the image"""
        band_handle = self._handle.GetRasterBand(band)
        return band_handle.DataType

    def get_bytes_per_pixel(self, band=1):
        '''
        returns the number of bytes per pixel
        '''
        band_handle = self._handle.GetRasterBand(band)
        return utilities.get_num_bytes_from_gdal_type(band_handle.DataType)

    def get_block_info(self, band):
        """Returns ((block height, block width), (num blocks x, num blocks y))"""
        band_handle = self._handle.GetRasterBand(band)
        block_size = band_handle.GetBlockSize()

        num_blocks_x = int(math.ceil(self._handle.RasterXSize / block_size[0]))
        num_blocks_y = int(math.ceil(self._handle.RasterYSize / block_size[1]))

        return (block_size, (num_blocks_x, num_blocks_y))

    def get_all_metadata(self):
        """Returns all useful image metadata"""
        data = dict()
        data['projection'] = self._handle.GetProjection()
        data['geotransform'] = self._handle.GetGeoTransform()
        data['gcps'] = self._handle.GetGCPs()
        data['gcpproj'] = self._handle.GetGCPProjection()
        data['metadata'] = self._handle.GetMetadata()
        return data

    def get_block_aligned_read_roi(self, desired_roi):
        """Returns the block aligned pixel region to read in a Rectangle format
           to get the requested data region while respecting block boundaries.
        """
        size = self.image_size()
        bounds = rectangle.Rectangle(0, 0, width=size[0], height=size[1])
        if not bounds.contains_rect(desired_roi):
            raise Exception('desired_roi ' + str(desired_roi)
                            + ' is outside the bounds of image with size' + str(size))

        band = 1
        (block_size, unused_num_blocks) = self.get_block_info(band)
        start_block_x = int(math.floor(desired_roi.min_x     / block_size[0]))
        start_block_y = int(math.floor(desired_roi.min_y     / block_size[1]))
        # Rect max is exclusive
        stop_block_x = int(math.floor((desired_roi.max_x-1) / block_size[0]))
        # The stops are inclusive
        stop_block_y = int(math.floor((desired_roi.max_y-1) / block_size[1]))

        start_col = start_block_x * block_size[0]
        start_row = start_block_y * block_size[1]
        num_cols = (stop_block_x - start_block_x + 1) * block_size[0]
        num_rows = (stop_block_y - start_block_y + 1) * block_size[1]

        # Restrict the output region to the bounding box of the image.
        # - Needed to handle images with partial tiles at the boundaries.
        ans = rectangle.Rectangle(start_col, start_row, width=num_cols, height=num_rows)
        size = self.image_size()
        bounds = rectangle.Rectangle(0, 0, width=size[0], height=size[1])
        return ans.get_intersection(bounds)

    def read_pixels(self, roi, band, buf=None):
        """Reads in the requested region of the image."""
        band_handle = self._handle.GetRasterBand(band)

        if buf is None:
            data = band_handle.ReadAsArray(roi.min_x, roi.min_y, roi.width(), roi.height())
            return data
        band_handle.ReadAsArray(roi.min_x, roi.min_y, roi.width(), roi.height(), buf_obj=buf)
        return buf

def write_simple_image(output_path, data, data_type=gdal.GDT_Byte, metadata=None):
    """Just dump 2D numpy data to a single channel image file"""

    num_cols  = data.shape[1]
    num_rows  = data.shape[0]
    num_bands = 1

    driver = gdal.GetDriverByName('GTiff')
    handle = driver.Create(output_path, num_cols, num_rows, num_bands, data_type)
    if metadata:
        handle.SetProjection  (metadata['projection'  ])
        handle.SetGeoTransform(metadata['geotransform'])
        handle.SetMetadata    (metadata['metadata'    ])
        handle.SetGCPs        (metadata['gcps'], metadata['gcpproj'])

    band   = handle.GetRasterBand(1)
    band.WriteArray(data)
    band.FlushCache()

def write_multiband_image(output_path, data, data_type=gdal.GDT_Byte):
    """Dump 3D numpy data to a multi channel image file"""

    num_cols  = data.shape[2]
    num_rows  = data.shape[1]
    num_bands = data.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    handle = driver.Create(output_path, num_cols, num_rows, num_bands, data_type)
    for b in range(0,num_bands):
        band   = handle.GetRasterBand(b+1)
        band.WriteArray(data[b,:,:])
    band.FlushCache()


class TiffWriter:
    """Class to manage block writes to a Geotiff file.
    """
    def __init__(self, path, num_rows, num_cols, num_bands=1, data_type=gdal.GDT_Byte, #pylint:disable=too-many-arguments
                 tile_width=256, tile_height=256, no_data_value=None, metadata=None):
        self._width  = num_cols
        self._height = num_rows
        self._tile_height = tile_height
        self._tile_width  = tile_width

        # Constants
        options = ['COMPRESS=LZW', 'BigTIFF=IF_SAFER', 'INTERLEAVE=BAND']
        options += ['BLOCKXSIZE='+str(self._tile_width),
                    'BLOCKYSIZE='+str(self._tile_height)]
        MIN_SIZE_FOR_TILES=100
        if num_cols > MIN_SIZE_FOR_TILES or num_rows > MIN_SIZE_FOR_TILES:
            options += ['TILED=YES']

        driver = gdal.GetDriverByName('GTiff')
        self._handle = driver.Create(path, num_cols, num_rows, num_bands, data_type, options)
        if not self._handle:
            raise Exception('Failed to create output file: ' + path)

        if no_data_value is not None:
            for i in range(1,num_bands+1):
                self._handle.GetRasterBand(i).SetNoDataValue(no_data_value)

        # Set the metadata values used in TiffReader
        # TODO: May need to adjust the order here to work with some files
        if metadata:
            self._handle.SetProjection  (metadata['projection'  ])
            self._handle.SetGeoTransform(metadata['geotransform'])
            self._handle.SetMetadata    (metadata['metadata'    ])
            self._handle.SetGCPs        (metadata['gcps'], metadata['gcpproj'])

    def __del__(self):
        self._handle.FlushCache()
        self._handle = None

    def get_size(self):
        return (self._width, self._height)

    def get_tile_size(self):
        return (self._tile_width, self._tile_height)

    def get_num_tiles(self):
        num_x = int(math.ceil(self._width  / self._tile_width))
        num_y = int(math.ceil(self._height / self._tile_height))
        return (num_x, num_y)

    def write_block(self, data, block_col, block_row, band=0):
        '''Add a tile write command to the queue.
           Partial tiles are allowed at the right at bottom edges.
        '''

        # Check that the tile position is valid
        num_tiles = self.get_num_tiles()
        if (block_col >= num_tiles[0]) or (block_row >= num_tiles[1]):
            raise Exception('Block position ' + str((block_col, block_row))
                            + ' is outside the tile count: ' + str(num_tiles))
        is_edge_block = ((block_col == num_tiles[0]-1) or
                         (block_row == num_tiles[1]-1))

        if is_edge_block: # Data must fit inside the image size
            max_col = block_col*self._tile_width  + data.shape[1]
            max_row = block_row*self._tile_height + data.shape[0]
            if ( (max_col > self._width ) or
                 (max_row > self._height)   ):
                raise Exception('Error: Data block max position '
                                + str((max_col, max_row))
                                + ' falls outside the image bounds: '
                                + str((self._width, self._height)))
        else: # Shape must be exactly one tile
            if ( (data.shape[0] != self._tile_height) or
                 (data.shape[1] != self._tile_width )  ):
                raise Exception('Error: Data block size is ' + str(data.shape)
                                + ', output file block size is '
                                + str((self._tile_width, self._tile_height)))

        gdal_band = self._handle.GetRasterBand(band+1)

        bSize = gdal_band.GetBlockSize()
        gdal_band.WriteArray(data, block_col*bSize[0], block_row*bSize[1])


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

    def __init__(self, image_path_list=None):
        self._image_handles = []

        if image_path_list:
            self.load_images(image_path_list)

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
        for handle in self._image_handles:
            handle.close_image()
        self._image_handles = []

    # TODO: Error checking!
    def image_size(self):
        '''
        Returns image size!
        '''
        return self._image_handles[0].image_size()

    def data_type(self, band=1):
        '''
        Returns band data type!
        '''
        return self._image_handles[0].data_type(band)

    def num_bands(self):
        """Get the total number of bands across all input images"""
        num_bands = 0
        for image in self._image_handles:
            num_bands += image.num_bands()
        return num_bands

    def nodata_value(self, band=1):
        '''Returns the bands nodata value'''
        return self._image_handles[0].nodata_value(band)

    def get_block_info(self, band):
        '''Returns the bands blockinfo'''
        return self._image_handles[0].get_block_info(band)

    def get_all_metadata(self):
        """All input images should share the same metadata"""
        return self._image_handles[0].get_all_metadata()

    def _get_band_index(self, image_index):
        """Return the absolute band index of the first band in the given image index"""
        band_index = 0
        for i in range(0, image_index):
            band_index += self._image_handles[i].num_bands()
        return band_index

    def read_roi(self, roi=None):
        if roi is None:
            s = self.image_size()
            roi = rectangle.Rectangle(0, 0, s[0], s[1])
        result = np.zeros(shape=(roi.height(), roi.width(), self.num_bands()),
                          dtype=utilities.gdal_dtype_to_numpy_type(self.data_type()))

        for (i, image) in enumerate(self._image_handles):
            band_index = self._get_band_index(i) # The first index of one or more sequential bands
            for band in range(1,image.num_bands()+1):
                result[:, :, band_index] = image.read_pixels(roi, band)
                band_index += 1

        return result

    def process_rois(self, requested_rois, callback_function, strict_order=False, show_progress=False):
        """Process the given region broken up into blocks using the callback function.
           Each block will get the image data from each input image passed into the function.
           Function definition TBD!
           Blocks that go over the image boundary will be passed as partial image blocks.
           All data reading and function calling takes place in the current thread, to use
           multiple threads you need to hand off the work in the callback function.
           If strict_order is passed in the ROIs will be processed in exactly the input order.
        """

        if not self._image_handles:
            raise Exception('Cannot process region, no images loaded!')
        first_image = self._image_handles[0]

        block_rois = copy.copy(requested_rois)

        image_size = self.image_size()
        whole_bounds = rectangle.Rectangle(0, 0, width=image_size[0], height=image_size[1])
        for roi in requested_rois:
            if not whole_bounds.contains_rect(roi):
                raise Exception('Roi outside image bounds: ' + str(roi))

        buf = np.zeros(shape=(self.num_bands(), 1, 1),
                       dtype=utilities.gdal_dtype_to_numpy_type(self.data_type()))

        total_rois = len(block_rois)
        num_remaining = total_rois
        # Loop until we have processed all of the blocks.
        while block_rois:
            # For the next (output) block, figure out the (input block) aligned
            # data read that we need to perform to get it.
            read_roi = first_image.get_block_aligned_read_roi(block_rois[0])

            if read_roi.width() > buf.shape[2] or read_roi.height() > buf.shape[1]: # pylint: disable=E1136
                new_height, new_width = (max(buf.shape[1], read_roi.height()), max(buf.shape[2], read_roi.width())) # pylint: disable=E1136
                buf = np.zeros(shape=(self.num_bands(), new_height, new_width),
                               dtype=utilities.gdal_dtype_to_numpy_type(self.data_type()))

            # TODO: We need a way to make sure the bands for certain image types always end up
            #       in the same order!
            # Read in all of the data for this region, appending all bands for
            # each input image.
            for image in self._image_handles:
                for band in range(1, image.num_bands()+1):
                    image.read_pixels(read_roi, band, buf=buf[band-1])

            # Loop through the remaining ROIs and apply the callback function to each
            # ROI that is contained in the section we read in.
            index = 0
            while index < len(block_rois):

                roi = block_rois[index]
                if not read_roi.contains_rect(roi):
                    index += 1
                    if strict_order:
                        break
                    continue

                x0 = roi.min_x - read_roi.min_x
                y0 = roi.min_y - read_roi.min_y
                callback_function(roi, buf[:, y0:y0 + roi.height(), x0:x0 + roi.width()])

                # Instead of advancing the index, remove the current ROI from the list.
                block_rois.pop(index)
                num_remaining -= 1
                if show_progress:
                    utilities.progress_bar('%d / %d' % (total_rois - num_remaining, total_rois),
                                           (total_rois - num_remaining) / total_rois, prefix='Blocks Processed:')
        if show_progress:
            print()
