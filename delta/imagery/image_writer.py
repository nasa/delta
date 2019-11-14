"""
Geotiff writing class using GDAL with dedicated writer thread.
"""
import math

from osgeo import gdal

#=============================================================================
# Image writer class

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

        # Set the metadata values used in image_reader.py
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
