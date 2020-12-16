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
Block-aligned reading from multiple Geotiff files.
"""
import os
import math

import numpy as np
from osgeo import gdal

from delta.config import config
from delta.imagery import delta_image, rectangle


# Suppress GDAL warnings, errors become exceptions so we get them
gdal.SetConfigOption('CPL_LOG', '/dev/null')
gdal.UseExceptions()



class TiffImage(delta_image.DeltaImage):
    """For geotiffs."""

    def __init__(self, path, nodata_value=None):
        '''
        Opens a geotiff for reading. paths can be either a single filename or a list.
        For a list, the images are opened in order as a multi-band image, assumed to overlap.
        '''
        super().__init__(nodata_value)
        paths = self._prep(path)

        self._paths = paths
        self._handles = []
        for p in paths:
            if not os.path.exists(p):
                raise Exception('Image file does not exist: ' + p)
            result = gdal.Open(p)
            if result is None:
                raise Exception('Failed to open tiff file %s.' % (p))
            self._handles.append(result)
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
        num_bands = len(bands) if bands else self.num_bands()

        if buf is None:
            buf = np.zeros(shape=(num_bands, roi.width(), roi.height()), dtype=self.numpy_type())
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
        ret = self._handles[h].GetRasterBand(b)
        assert ret
        return ret

    # using custom nodata from config TODO: use both
    #def nodata_value(self, band=0):
    #    '''
    #    Returns the value that indicates no data is present in a pixel for the specified band.
    #    '''
    #    self.__asert_open()
    #    return self._gdal_band(band).GetNoDataValue()

    def data_type(self, band=0):
        '''
        Returns the GDAL data type of the image.
        '''
        self.__asert_open()
        return self._gdal_band(band).DataType

    def numpy_type(self, band=0):
        self.__asert_open()
        dtype = self.data_type(band)
        if dtype == gdal.GDT_Byte:
            return np.uint8
        if dtype == gdal.GDT_UInt16:
            return np.uint16
        if dtype == gdal.GDT_UInt32:
            return np.uint32
        if dtype == gdal.GDT_Float32:
            return np.float32
        if dtype == gdal.GDT_Float64:
            return np.float64
        raise Exception('Unrecognized gdal data type: ' + str(dtype))

    def bytes_per_pixel(self, band=0):
        '''
        Returns the number of bytes per pixel
        '''
        self.__asert_open()
        results = {
            gdal.GDT_Byte:    1,
            gdal.GDT_UInt16:  2,
            gdal.GDT_UInt32:  4,
            gdal.GDT_Float32: 4,
            gdal.GDT_Float64: 8
        }
        return results.get(self.data_type(band))

    def block_size(self):
        """Returns (block height, block width)"""
        (bs, _) = self.block_info()
        return bs

    def block_info(self, band=0):
        """Returns ((block height, block width), (num blocks x, num blocks y))"""
        self.__asert_open()
        band_handle = self._gdal_band(band)
        block_size = band_handle.GetBlockSize()

        num_blocks_x = int(math.ceil(self.width()  / block_size[0]))
        num_blocks_y = int(math.ceil(self.height() / block_size[1]))

        # we are backwards from gdal I think
        return ((block_size[1], block_size[0]), (num_blocks_x, num_blocks_y))

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

    def block_aligned_roi(self, desired_roi):
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

        start_x = start_block_x * block_size[0]
        start_y = start_block_y * block_size[1]
        w = (stop_block_x - start_block_x + 1) * block_size[0]
        h = (stop_block_y - start_block_y + 1) * block_size[1]

        # Restrict the output region to the bounding box of the image.
        # - Needed to handle images with partial tiles at the boundaries.
        ans = rectangle.Rectangle(start_x, start_y, width=w, height=h)
        bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        return ans.get_intersection(bounds)

    def save(self, path, tile_size=(0,0), nodata_value=None, show_progress=False):
        """
        Save a TiffImage to the file output_path, optionally overwriting the tile_size.
        Input tile size is (width, height)
        """

        if nodata_value is None:
            nodata_value = self.nodata_value()
        # Use the input tile size for the block size unless the user specified one.
        block_size_y, block_size_x = self.block_size()
        if tile_size[0] > 0:
            block_size_x = tile_size[0]
        if tile_size[1] > 0:
            block_size_y = tile_size[1]

        # Set up the output image
        with _TiffWriter(path, self.width(), self.height(), self.num_bands(),
                         self.data_type(), block_size_x, block_size_y,
                         nodata_value, self.metadata()) as writer:
            input_bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
            output_rois = input_bounds.make_tile_rois((block_size_x, block_size_y), include_partials=True)

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
        output_path = config.io.cache.manager().register_item(fname)

        if not os.path.exists(output_path):
            # Just remove the alpha band from the original image
            cmd = 'gdal_translate -b 1 -b 2 -b 3 ' + paths + ' ' + output_path
            os.system(cmd)
        return [output_path]

def numpy_dtype_to_gdal_type(dtype): #pylint: disable=R0911
    if dtype == np.uint8:
        return gdal.GDT_Byte
    if dtype == np.uint16:
        return gdal.GDT_UInt16
    if dtype == np.uint32:
        return gdal.GDT_UInt32
    if dtype == np.int16:
        return gdal.GDT_Int16
    if dtype == np.int32:
        return gdal.GDT_Int32
    if dtype == np.float32:
        return gdal.GDT_Float32
    if dtype == np.float64:
        return gdal.GDT_Float64
    raise Exception('Unrecognized numpy data type: ' + str(dtype))

def write_tiff(output_path, data, metadata=None):
    """Try to write a tiff file"""

    if len(data.shape) < 3:
        num_bands = 1
    else:
        num_bands = data.shape[2]
    data_type = numpy_dtype_to_gdal_type(data.dtype)

    TILE_SIZE=256

    with _TiffWriter(output_path, data.shape[0], data.shape[1], num_bands=num_bands,
                     data_type=data_type, metadata=metadata, tile_width=min(TILE_SIZE, data.shape[0]),
                     tile_height=min(TILE_SIZE, data.shape[1])) as writer:
        for x in range(0, data.shape[0], TILE_SIZE):
            for y in range(0, data.shape[1], TILE_SIZE):
                block = (x // TILE_SIZE, y // TILE_SIZE)
                if len(data.shape) < 3:
                    writer.write_block(data[x:x+TILE_SIZE, y:y+TILE_SIZE], block[0], block[1], 0)
                else:
                    for b in range(num_bands):
                        writer.write_block(data[x:x+TILE_SIZE, y:y+TILE_SIZE, b], block[0], block[1], b)

class _TiffWriter:
    """
    Class to manage block writes to a Geotiff file. Internal helper class.
    """
    def __init__(self, path, width, height, num_bands=1, data_type=gdal.GDT_Byte, #pylint:disable=too-many-arguments
                 tile_width=256, tile_height=256, nodata_value=None, metadata=None):
        self._width  = width
        self._height = height
        self._tile_height = tile_height
        self._tile_width  = tile_width
        self._handle = None

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

        if nodata_value is not None:
            for i in range(1,num_bands+1):
                self._handle.GetRasterBand(i).SetNoDataValue(nodata_value)

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
        assert gdal_band

        gdal_band.WriteArray(data, block_y * self._tile_height, block_x * self._tile_width)

    def write_region(self, data, x, y):
        assert 0 <= y < self._height
        assert 0 <= x < self._width

        if len(data.shape) < 3:
            gdal_band = self._handle.GetRasterBand(1)
            assert gdal_band
            gdal_band.WriteArray(data, y, x)
            return

        for band in range(data.shape[2]):
            gdal_band = self._handle.GetRasterBand(band+1)
            assert gdal_band
            gdal_band.WriteArray(data[:, :, band], y, x)

class TiffWriter(delta_image.DeltaImageWriter):
    def __init__(self, filename):
        self._filename = filename
        self._tiff_w = None

    def initialize(self, size, numpy_dtype, metadata=None, nodata_value=None):
        """
        Prepare for writing with the given size and dtype.
        """
        assert (len(size) == 3), ('Error: len(size) of '+str(size)+' != 3')
        TILE_SIZE = 256
        self._tiff_w = _TiffWriter(self._filename, size[0], size[1], num_bands=size[2],
                                   data_type=numpy_dtype_to_gdal_type(numpy_dtype), metadata=metadata,
                                   nodata_value=nodata_value,
                                   tile_width=min(TILE_SIZE, size[0]), tile_height=min(TILE_SIZE, size[1]))

    def write(self, data, x, y):
        """
        Writes the data as a rectangular block starting at the given coordinates.
        """
        self._tiff_w.write_region(data, x, y)

    def close(self):
        """
        Finish writing.
        """
        if self._tiff_w is not None:
            self._tiff_w.close()

    def abort(self):
        self.close()
        try:
            os.remove(self._filename)
        except OSError:
            pass
