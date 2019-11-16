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

        input_reader = TiffReader(input_paths)
        return input_reader.num_bands()

    def read(self, roi=None):
        input_paths = self.prep()

        # Set up the input image handle
        input_reader = TiffReader(input_paths)
        return input_reader.read(roi=roi)

    def size(self):
        input_paths = self.prep()

        input_reader = TiffReader(input_paths)
        return input_reader.size()

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

class TiffReader:
    """Wrapper class to help read image data from GeoTiff files"""

    def __init__(self, paths):
        '''
        Opens a geotiff for reading. paths can be either a singlefilename or a list.
        For a list, the images are opened in order as a multi-band image, assumed to overlap.
        '''
        if isinstance(paths, str):
            paths = [paths]
        self._paths = paths
        self._handles = []
        for path in paths:
            if not os.path.exists(path):
                raise Exception('Image file does not exist: ' + path)
            self._handles.append(gdal.Open(path))
        self._band_map = []
        for i, h in enumerate(self._handles):
            if h.RasterXSize != self._handles[0].RasterXSize or h.RasterYSize != self._handles[0].RasterYSize:
                raise Exception('Images %s and %s have different sizes!' % (self._paths[0], self._paths[i]))
            for j in range(h.RasterCount):
                self._band_map.append((i, j + 1)) # gdal uses 1-based band indexing

    def __del__(self):
        self.close()

    def __asert_open(self):
        if self._handles is None:
            raise IOError('Operating on an image that has been closed.')

    def close(self):
        self._handles = None # gdal doesn't have a close function for some reason
        self._band_map = None
        self._paths = None

    def num_bands(self):
        self.__asert_open()
        return len(self._band_map)

    def width(self):
        self.__asert_open()
        return self._handles[0].RasterYSize

    def height(self):
        self.__asert_open()
        return self._handles[0].RasterXSize

    def size(self):
        self.__asert_open()
        return (self.height(), self.width())

    def _gdal_band(self, band):
        (h, b) = self._band_map[band]
        return self._handles[h].GetRasterBand(b)

    def nodata_value(self, band=0):
        '''
        Returns the value that indicates no data is present in a pixel for the specified band.
        '''
        self.__asert_open()
        return self._gdal_band(band).GetNoDataValue()

    def data_type(self, band=0):
        '''
        Returns the GDAL data type of the image.
        '''
        self.__asert_open()
        return self._gdal_band(band).DataType

    def numpy_type(self, band=0):
        self.__asert_open()
        return utilities.gdal_dtype_to_numpy_type(self.data_type(band))

    def bytes_per_pixel(self, band=0):
        '''
        Returns the number of bytes per pixel
        '''
        self.__asert_open()
        return utilities.get_num_bytes_from_gdal_type(self.data_type(band))

    def block_info(self, band=0):
        """Returns ((block height, block width), (num blocks x, num blocks y))"""
        self.__asert_open()
        band_handle = self._gdal_band(band)
        block_size = band_handle.GetBlockSize()

        num_blocks_x = int(math.ceil(self.height() / block_size[0]))
        num_blocks_y = int(math.ceil(self.width() / block_size[1]))

        return (block_size, (num_blocks_x, num_blocks_y))

    def metadata(self):
        '''
        Returns all useful image metadata.

        If multiple images were specified, returns the information from the first.
        '''
        self.__asert_open()
        data = dict()
        h = self._handles[0]
        data['projection'] = h.GetProjection()
        data['geotransform'] = h.GetGeoTransform()
        data['gcps'] = h.GetGCPs()
        data['gcpproj'] = h.GetGCPProjection()
        data['metadata'] = h.GetMetadata()
        return data

    def get_block_aligned_read_roi(self, desired_roi):
        '''
        Returns the block aligned pixel region to read in a Rectangle format
        to get the requested data region while respecting block boundaries.
        '''
        self.__asert_open()
        size = self.size()
        bounds = rectangle.Rectangle(0, 0, width=size[0], height=size[1])
        if not bounds.contains_rect(desired_roi):
            raise Exception('desired_roi ' + str(desired_roi)
                            + ' is outside the bounds of image with size' + str(size))

        (block_size, unused_num_blocks) = self.block_info(0)
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
        size = self.size()
        bounds = rectangle.Rectangle(0, 0, width=size[0], height=size[1])
        return ans.get_intersection(bounds)

    def read(self, roi=None, band=None, buf=None):
        '''
        Reads in the requested region of the image.

        If roi is not specified, reads the entire image.
        If buf is specified, writes the image to buf.
        If band is not specified, reads all bands in [band, row, col] indexing.
        '''
        self.__asert_open()
        if roi is None:
            roi = rectangle.Rectangle(0, 0, self.height(), self.width())

        if band is not None:
            band_handle = self._gdal_band(band)
            return band_handle.ReadAsArray(roi.min_x, roi.min_y, roi.width(), roi.height(), buf_obj=buf)

        if buf is None:
            buf = np.zeros(shape=(self.num_bands(), roi.height(), roi.width()), dtype=self.numpy_type())

        for i in range(self.num_bands()):
            self.read(roi=roi, band=i, buf=buf[i, :, :])
        return buf

    def process_rois(self, requested_rois, callback_function, strict_order=False, show_progress=False):
        '''
        Process the given region broken up into blocks using the callback function.
        Each block will get the image data from each input image passed into the function.
        Blocks that go over the image boundary will be passed as partial image blocks.
        All data reading and function calling takes place in the current thread, to use
        multiple threads you need to hand off the work in the callback function.
        If strict_order is passed in the ROIs will be processed in exactly the input order.
        '''

        self.__asert_open()

        block_rois = copy.copy(requested_rois)

        whole_bounds = rectangle.Rectangle(0, 0, self.height(), self.width())
        for roi in requested_rois:
            if not whole_bounds.contains_rect(roi):
                raise Exception('Roi outside image bounds: ' + str(roi))

        buf = np.zeros(shape=(self.num_bands(), 1, 1), dtype=self.numpy_type())

        total_rois = len(block_rois)
        num_remaining = total_rois
        while block_rois:
            # For the next (output) block, figure out the (input block) aligned
            # data read that we need to perform to get it.
            read_roi = self.get_block_aligned_read_roi(block_rois[0])

            if read_roi.width() > buf.shape[2] or read_roi.height() > buf.shape[1]: # pylint: disable=E1136
                new_height, new_width = (max(buf.shape[1], read_roi.height()), max(buf.shape[2], read_roi.width())) # pylint: disable=E1136
                buf = np.zeros(shape=(self.num_bands(), new_height, new_width), dtype=self.numpy_type())

            self.read(roi=read_roi, buf=buf)

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
