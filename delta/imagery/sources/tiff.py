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
        '''
        Opens a geotiff for reading. paths can be either a single filename or a list.
        For a list, the images are opened in order as a multi-band image, assumed to overlap.
        '''
        super(TiffImage, self).__init__()
        paths = self._prep(path)

        self._paths = paths
        self._handles = []
        for p in paths:
            if not os.path.exists(p):
                raise Exception('Image file does not exist: ' + p)
            self._handles.append(gdal.Open(p))
        self._band_map = []
        for i, h in enumerate(self._handles):
            if h.RasterXSize != self._handles[0].RasterXSize or h.RasterYSize != self._handles[0].RasterYSize:
                raise Exception('Images %s and %s have different sizes!' % (self._paths[0], self._paths[i]))
            for j in range(h.RasterCount):
                self._band_map.append((i, j + 1)) # gdal uses 1-based band indexing

    def __del__(self):
        self.close()

    def _prep(self, paths): #pylint:disable=no-self-use
        """
        Prepare the file to be opened by other tools (unpack, etc).

        Returns a list of underlying files to load instead of the original path.
        This is intended to be overwritten by subclasses.
        """
        if isinstance(paths, str):
            return [paths]
        return paths

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

    def size(self):
        self.__asert_open()
        return (self._handles[0].RasterYSize, self._handles[0].RasterXSize)

    def _read(self, roi, bands, buf=None):
        self.__asert_open()

        if buf is None:
            buf = np.zeros(shape=(self.num_bands(), roi.width(), roi.height()), dtype=self.numpy_type())
        for i, b in enumerate(bands):
            band_handle = self._gdal_band(b)
            s = buf[i, :, :].shape
            if s != (roi.width(), roi.height()):
                raise IOError('Buffer shape should be (%d, %d) but is (%d, %d)!' %
                              (roi.width(), roi.height(), s[0], s[1]))
            band_handle.ReadAsArray(roi.min_y, roi.min_x, roi.height(), roi.width(), buf_obj=buf[i, :, :])
        return np.transpose(buf, [1, 2, 0])

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
        bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        if not bounds.contains_rect(desired_roi):
            raise Exception('desired_roi ' + str(desired_roi)
                            + ' is outside the bounds of image with size' + str(self.size()))

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
        bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        return ans.get_intersection(bounds)

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

        whole_bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        for roi in requested_rois:
            if not whole_bounds.contains_rect(roi):
                raise Exception('Roi outside image bounds: ' + str(roi) + str(whole_bounds))

        buf = np.zeros(shape=(self.num_bands(), 1, 1), dtype=self.numpy_type())

        total_rois = len(block_rois)
        num_remaining = total_rois
        while block_rois:
            # For the next (output) block, figure out the (input block) aligned
            # data read that we need to perform to get it.
            read_roi = self.get_block_aligned_read_roi(block_rois[0])

            if read_roi.width() > buf.shape[1] or read_roi.height() > buf.shape[2]: # pylint: disable=E1136
                new_width, new_height = (max(buf.shape[1], read_roi.width()), max(buf.shape[2], read_roi.height())) # pylint: disable=E1136
                buf = np.zeros(shape=(self.num_bands(), new_width, new_height), dtype=self.numpy_type())

            self.read(roi=read_roi, buf=buf[:, :read_roi.width(), :read_roi.height()])

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

                callback_function(roi, np.transpose(buf[:, x0:x0 + roi.width(), y0:y0 + roi.height()], [1, 2, 0]))

                # Instead of advancing the index, remove the current ROI from the list.
                block_rois.pop(index)
                num_remaining -= 1
                if show_progress:
                    utilities.progress_bar('%d / %d' % (total_rois - num_remaining, total_rois),
                                           (total_rois - num_remaining) / total_rois, prefix='Blocks Processed:')
        if show_progress:
            print()

    def save(self, path, tile_size=(0,0), show_progress=False):
        """
        Save a TiffImage to the file output_path, optionally overwriting the tile_size.
        """

        # Use the input tile size for the block size unless the user specified one.
        (bs, _) = self.block_info()
        block_size_x = bs[0]
        block_size_y = bs[1]
        if tile_size[0] > 0:
            block_size_x = tile_size[0]
        if tile_size[1] > 0:
            block_size_y = tile_size[1]

        # Set up the output image
        with TiffWriter(path, self.width(), self.height(), self.num_bands(),
                        self.data_type(), block_size_x, block_size_y,
                        self.nodata_value(), self.metadata()) as writer:
            input_bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
            output_rois = input_bounds.make_tile_rois(block_size_x, block_size_y, include_partials=True)

            def callback_function(output_roi, data):
                """Callback function to write the first channel to the output file."""

                # Figure out some ROI positioning values
                block_x = output_roi.min_x / block_size_x
                block_y = output_roi.min_y / block_size_y

                # Loop on bands
                for band in range(data.shape[2]):
                    writer.write_block(data[:, :, band], block_x, block_y, band)

            self.process_rois(output_rois, callback_function, show_progress=show_progress)

class RGBAImage(TiffImage):
    """Basic RGBA images where the alpha channel needs to be stripped"""

    def _prep(self, paths):
        """Converts RGBA images to RGB images"""

        # Get the path to the cached image
        fname = os.path.basename(paths)
        output_path = config.cache_manager().register_item(fname)

        if not os.path.exists(output_path):
            # Just remove the alpha band from the original image
            cmd = 'gdal_translate -b 1 -b 2 -b 3 ' + paths + ' ' + output_path
            os.system(cmd)
        return [output_path]

def write_tiff(output_path, data, data_type=gdal.GDT_Byte, metadata=None):
    """Just dump 2D numpy data to a single channel image file"""

    if len(data.shape) < 3:
        num_bands = 1
    else:
        num_bands = data.shape[2]

    with TiffWriter(output_path, data.shape[0], data.shape[1], num_bands=num_bands,
                    data_type=data_type, metadata=metadata, tile_width=data.shape[0],
                    tile_height=data.shape[1]) as writer:
        if len(data.shape) < 3:
            writer.write_block(data[:, :], 0, 0, 0)
        else:
            for b in range(num_bands):
                writer.write_block(data[:, :, b], 0, 0, b)

class TiffWriter:
    """Class to manage block writes to a Geotiff file.
    """
    def __init__(self, path, width, height, num_bands=1, data_type=gdal.GDT_Byte, #pylint:disable=too-many-arguments
                 tile_width=256, tile_height=256, no_data_value=None, metadata=None):
        self._width  = width
        self._height = height
        self._tile_height = tile_height
        self._tile_width  = tile_width

        # Constants
        options = ['COMPRESS=LZW', 'BigTIFF=IF_SAFER', 'INTERLEAVE=BAND']
        options += ['BLOCKXSIZE='+str(self._tile_width),
                    'BLOCKYSIZE='+str(self._tile_height)]
        MIN_SIZE_FOR_TILES=100
        if width > MIN_SIZE_FOR_TILES or height > MIN_SIZE_FOR_TILES:
            options += ['TILED=YES']

        driver = gdal.GetDriverByName('GTiff')
        self._handle = driver.Create(path, height, width, num_bands, data_type, options)
        if not self._handle:
            raise Exception('Failed to create output file: ' + path)

        if no_data_value is not None:
            for i in range(1,num_bands+1):
                self._handle.GetRasterBand(i).SetNoDataValue(no_data_value)

        # TODO: May need to adjust the order here to work with some files
        if metadata:
            self._handle.SetProjection  (metadata['projection'  ])
            self._handle.SetGeoTransform(metadata['geotransform'])
            self._handle.SetMetadata    (metadata['metadata'    ])
            self._handle.SetGCPs        (metadata['gcps'], metadata['gcpproj'])

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
        return False

    def close(self):
        if self._handle is not None:
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

    def write_block(self, data, block_x, block_y, band=0):
        '''Add a tile write command to the queue.
           Partial tiles are allowed at the right at bottom edges.
        '''

        # Check that the tile position is valid
        num_tiles = self.get_num_tiles()
        if (block_x >= num_tiles[0]) or (block_y >= num_tiles[1]):
            raise Exception('Block position ' + str((block_x, block_y))
                            + ' is outside the tile count: ' + str(num_tiles))
        is_edge_block = ((block_x == num_tiles[0]-1) or
                         (block_y == num_tiles[1]-1))

        if is_edge_block: # Data must fit inside the image size
            max_x = block_x * self._tile_width  + data.shape[0]
            max_y = block_y * self._tile_height + data.shape[1]
            if max_x > self._width or max_y > self._height:
                raise Exception('Error: Data block max position '
                                + str((max_x, max_y))
                                + ' falls outside the image bounds: '
                                + str((self._width, self._height)))
        else: # Shape must be exactly one tile
            if ( (data.shape[0] != self._tile_width) or
                 (data.shape[1] != self._tile_height)  ):
                raise Exception('Error: Data block size is ' + str(data.shape)
                                + ', output file block size is '
                                + str((self._tile_width, self._tile_height)))

        gdal_band = self._handle.GetRasterBand(band+1)

        bSize = gdal_band.GetBlockSize()
        gdal_band.WriteArray(data, block_y * bSize[1], block_x * bSize[0])
