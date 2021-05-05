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

from delta.imagery import delta_image, rectangle
from delta.extensions.sources.npy import NumpyImage


# Suppress GDAL warnings, errors become exceptions so we get them
gdal.SetConfigOption('CPL_LOG', '/dev/null')
gdal.UseExceptions()

_GDAL_TO_NUMPY_TYPES = {
    gdal.GDT_Byte:    np.dtype(np.uint8),
    gdal.GDT_UInt16:  np.dtype(np.uint16),
    gdal.GDT_UInt32:  np.dtype(np.uint32),
    gdal.GDT_Float32: np.dtype(np.float32),
    gdal.GDT_Float64: np.dtype(np.float64)
}
_NUMPY_TO_GDAL_TYPES = {v: k for k, v in _GDAL_TO_NUMPY_TYPES.items()}

class TiffImage(delta_image.DeltaImage):
    """Images supported by GDAL."""

    def __init__(self, path, nodata_value=None):
        """
        Opens a geotiff for reading.

        Parameters
        ----------
        paths: str or List[str]
            Either a single filename or a list.
            For a list, the images are opened in order as a multi-band image, assumed to overlap.
        nodata_value: dtype of image
            Value representing no data.
        """
        super().__init__(nodata_value)
        paths = self._prep(path)

        self._path = path
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

        This can be overwritten by subclasses to, for example,
        unpack a zip file to a cache directory.

        Parameters
        ----------
        paths: str or List[str]
            Paths passed to constructor

        Returns
        -------
        Returns a list of underlying files to load instead of the original paths.
        """
        if isinstance(paths, str):
            return [paths]
        return paths

    def __asert_open(self):
        if self._handles is None:
            raise IOError('Operating on an image that has been closed.')

    def close(self):
        """
        Close the image.
        """
        self._handles = None # gdal doesn't have a close function for some reason
        self._band_map = None
        self._paths = None

    def path(self):
        """
        Returns the paths returned by `_prep`.
        """
        return self._path

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
            buf = np.zeros(shape=(num_bands, roi.width(), roi.height()), dtype=self.dtype())
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

    def _gdal_type(self, band=0):
        """
        Returns the GDAL data type of the image.
        """
        self.__asert_open()
        return self._gdal_band(band).DataType

    def dtype(self):
        self.__asert_open()
        dtype = self._gdal_type(0)
        if dtype in _GDAL_TO_NUMPY_TYPES:
            return _GDAL_TO_NUMPY_TYPES[dtype]
        raise Exception('Unrecognized gdal data type: ' + str(dtype))

    def bytes_per_pixel(self, band=0):
        """
        Returns
        -------
        int:
            the number of bytes per pixel
        """
        self.__asert_open()
        return gdal.GetDataTypeSize(self._gdal_type(band)) // 8

    def block_size(self):
        """
        Returns
        -------
        (int, int):
            block height, block width
        """
        self.__asert_open()
        band_handle = self._gdal_band(0)
        block_size = band_handle.GetBlockSize()
        return (block_size[0], block_size[1])

    def metadata(self):
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
        self.__asert_open()
        bounds = rectangle.Rectangle(0, 0, width=self.width(), height=self.height())
        if not bounds.contains_rect(desired_roi):
            raise Exception('desired_roi ' + str(desired_roi)
                            + ' is outside the bounds of image with size' + str(self.size()))

        block_size = self.block_size()
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

    def save(self, path, tile_size=None, nodata_value=None, show_progress=False):
        """
        Save to file, with preprocessing applied.

        Parameters
        ----------
        path: str
            Filename to save to.
        tile_size: (int, int)
            If specified, overwrite block size
        nodata_value: image dtype
            If specified, overwrite nodata value
        show_progress: bool
            Write progress bar to stdout
        """

        write_tiff(path, image=self, nodata=nodata_value, block_size=tile_size,
                   show_progress=show_progress)

def _numpy_dtype_to_gdal_type(dtype): #pylint: disable=R0911
    if dtype in _NUMPY_TO_GDAL_TYPES:
        return _NUMPY_TO_GDAL_TYPES[dtype]
    raise Exception('Unrecognized numpy data type: ' + str(dtype))

def write_tiff(output_path: str, data: np.ndarray=None, image: delta_image.DeltaImage=None,
               nodata=None, metadata: dict=None, block_size=None, show_progress: bool=False):
    """
    Write a numpy array to a file as a tiff.

    Parameters
    ----------
    output_path: str
        Filename to save tiff file to
    data: numpy.ndarray
        Image data to save.
    image: delta_image.DeltaImage
        Image data to save (specify one of this or data).
    nodata: Any
        Nodata value.
    metadata: dict
        Optional metadata to include.
    block_size: Tuple[int]
        Optionally override block size for writing.
    show_progress: bool
        Display command line progress bar.
    """

    if data is not None:
        assert image is None, 'Must specify one of data or image.'
        image = NumpyImage(data, nodata_value=nodata)
    assert image is not None, 'Must specify either data or image.'
    num_bands = image.num_bands()
    data_type = _numpy_dtype_to_gdal_type(image.dtype())
    size = (image.width(), image.height())
    # gdal requires block size of 16 when writing...
    ts = image.block_size()
    if block_size:
        ts = block_size
    if nodata is None:
        nodata = image.nodata_value()
    if metadata is None:
        metadata = image.metadata()

    with _TiffWriter(output_path, size[0], size[1], num_bands=num_bands,
                     data_type=data_type, metadata=metadata, nodata_value=nodata,
                     tile_width=ts[0], tile_height=ts[1]) as writer:
        input_bounds = rectangle.Rectangle(0, 0, width=size[0], height=size[1])
        ts = writer.tile_shape()
        output_rois = input_bounds.make_tile_rois(ts, include_partials=True)
        def callback_function(output_roi, data):
            """Callback function to write the first channel to the output file."""

            # Figure out some ROI positioning values
            block_x = output_roi.min_x // ts[0]
            block_y = output_roi.min_y // ts[1]

            # Loop on bands
            if len(data.shape) == 2:
                writer.write_block(data[:, :], block_x, block_y, 0)
            else:
                for band in range(num_bands):
                    writer.write_block(data[:, :, band], block_x, block_y, band)

        image.process_rois(output_rois, callback_function, show_progress=show_progress)

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

        self.__initialize(path, num_bands, data_type, nodata_value, metadata)

    def __initialize(self, path, num_bands, data_type, nodata_value, metadata):
        options = ['BigTIFF=IF_SAFER', 'INTERLEAVE=BAND']
        if data_type not in (gdal.GDT_Float32, gdal.GDT_Float64):
            options += ['COMPRESS=LZW']

        MIN_SIZE_FOR_TILES=100
        if self._width > MIN_SIZE_FOR_TILES or self._height > MIN_SIZE_FOR_TILES:
            options += ['TILED=YES']
            # requires 16 byte alignment in tiled mode
            self._tile_width = min(max(1, self._tile_width // 16) * 16, 1024)
            self._tile_height = min(max(1, self._tile_height // 16) * 16, 1024)

        options += ['BLOCKXSIZE='+str(self._tile_width),
                    'BLOCKYSIZE='+str(self._tile_height)]

        driver = gdal.GetDriverByName('GTiff')
        self._handle = driver.Create(path, self._height, self._width, num_bands, data_type, options)
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

    def tile_shape(self):
        return (self._tile_width, self._tile_height)

    def close(self):
        if self._handle is not None:
            self._handle.FlushCache()
            self._handle = None

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
    """
    Write a geotiff to a file.
    """
    def __init__(self, filename):
        self._filename = filename
        self._tiff_w = None

    def initialize(self, size, numpy_dtype, metadata=None, nodata_value=None):
        assert (len(size) == 3), ('Error: len(size) of '+str(size)+' != 3')
        TILE_SIZE = 256
        self._tiff_w = _TiffWriter(self._filename, size[0], size[1], num_bands=size[2],
                                   data_type=_numpy_dtype_to_gdal_type(numpy_dtype), metadata=metadata,
                                   nodata_value=nodata_value,
                                   tile_width=min(TILE_SIZE, size[0]), tile_height=min(TILE_SIZE, size[1]))

    def write(self, data, x, y):
        self._tiff_w.write_region(data, x, y)

    def close(self):
        if self._tiff_w is not None:
            self._tiff_w.close()

    def abort(self):
        self.close()
        try:
            os.remove(self._filename)
        except OSError:
            pass
