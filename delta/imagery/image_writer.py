"""
Geotiff writing class using GDAL with dedicated writer thread.
"""
import threading
import time
import math

from osgeo import gdal

#=============================================================================
# Image writer class

def write_simple_image(output_path, data, data_type=gdal.GDT_Byte):
    """Just dump 2D numpy data to a single channel image file"""

    num_cols  = data.shape[1]
    num_rows  = data.shape[0]
    num_bands = 1

    driver = gdal.GetDriverByName('GTiff')
    handle = driver.Create(output_path, num_cols, num_rows, num_bands, data_type)
    band   = handle.GetRasterBand(1)
    band.WriteArray(data)
    band.FlushCache()


class TiffWriter:

    """Class to manage block writes to a Geotiff file.
       Currently only uint8 output is supported.
       TODO: Make sure everything works with opening and closing files in sequence!
    """

    def __init__(self):
        self._handle         = None
        self._shutdown       = False # Turned on in __del__()
        self._writeQueue     = []
        self._writeQueueLock = threading.Lock()

        print('Launching write thread...')
        self._writerThread = threading.Thread(target=self._internal_writer)

        self._writerThread.start()

        self._width  = 0
        self._height = 0
        self._tile_width  = 256
        self._tile_height = 256

    def __del__(self):
        '''Try this just in case, but it may not work!'''
        self.cleanup()

    def cleanup(self):
        print('cleanup called.')
        self.finish_writing_geotiff()

        # Shut down the writing thread, giving it ten minutes to finish.
        TIMEOUT = 60*10
        self._shutdown = True
        print('Terminating write thread...')
        self._writerThread.join(TIMEOUT)

    def get_size(self):
        return (self._width, self._height)

    def get_tile_size(self):
        return (self._tile_width, self._tile_height)

    def get_num_tiles(self):
        num_x = int(math.ceil(self._width  / self._tile_width))
        num_y = int(math.ceil(self._height / self._tile_height))
        return (num_x, num_y)

    def _internal_writer(self):
        '''Internal thread that writes blocks as they become available.'''

        SLEEP_TIME = 0.2

        blockCounter = 0
        print('Starting block writing loop')
        while True:

            # Check the write queue
            with self._writeQueueLock:
                noTiles = not self._writeQueue
                # If there is a tile, grab it while we have tho lock.
                if not noTiles:
                    parts = self._writeQueue.pop(0)

            # Wait until a tile is ready to be written
            if noTiles:
                if self._shutdown:
                    print('Writing thread is shutting down')
                    break # Shutdown was commanded
                else:
                    #print('Writing thread is sleeping')
                    time.sleep(SLEEP_TIME) # Wait
                    continue

            if not self._handle:
                print('ERROR: Trying to write image block before initialization!')
                break

            # Write out the block
            try:
                self._write_geotiff_block_internal(parts[0], parts[1], parts[2], parts[3])
                blockCounter += 1
            except Exception as e: #pylint: disable=W0703
                print(str(e))
                print('Caught exception writing: ' + str(parts[1] +', '+ str(parts[2])))

        print('Write thread ended after writing ' + str(blockCounter) +' blocks.')


    def _write_geotiff_block_internal(self, data, block_col, block_row, band):
        '''Write a single block to disk in a geotiff file'''

        if not self._handle:
            raise Exception('Failed to write block: output file not initialized!')

        band = self._handle.GetRasterBand(band+1)
        #stuff = dir(band)
        #for s in stuff:
        #    print(s)
        #band.WriteBlock(block_col, block_row, data) This function is not supported!

        bSize = band.GetBlockSize()
        band.WriteArray(data, block_col*bSize[0], block_row*bSize[1])

        #band.FlushBlock(block_col, block_row) This function is not supported!
        band.FlushCache() # TODO: Call after every tile?


    def init_output_geotiff(self, path, num_rows, num_cols, noDataValue, #pylint: disable=R0913
                            tile_width=256, tile_height=256, metadata=None,
                            data_type=gdal.GDT_Byte, num_bands=1):
        '''Set up a geotiff file for writing and return the handle.'''
        # TODO: Copy metadata from the source file.

        self._width  = num_cols
        self._height = num_rows
        self._tile_height = tile_height
        self._tile_width  = tile_width

        # Constants
        options = ['COMPRESS=LZW', 'BigTIFF=IF_SAFER', 'INTERLEAVE=BAND']
        # TODO: Some tile sizes fail to write if copied from the input image!!
        options += ['TILED=YES', 'BLOCKXSIZE='+str(self._tile_width),
                    'BLOCKYSIZE='+str(self._tile_height)]

        print('Starting TIFF driver...')
        driver = gdal.GetDriverByName('GTiff')
        self._handle = driver.Create(path, num_cols, num_rows, num_bands, data_type, options)
        if not self._handle:
            raise Exception('Failed to create output file: ' + path)

        if noDataValue is not None:
            for i in range(1,num_bands+1):
                self._handle.GetRasterBand(i).SetNoDataValue(noDataValue)

        # Set the metadata values used in image_reader.py
        # TODO: May need to adjust the order here to work with some files
        if metadata:
            self._handle.SetProjection  (metadata['projection'  ])
            self._handle.SetGeoTransform(metadata['geotransform'])
            self._handle.SetMetadata    (metadata['metadata'    ])
            self._handle.SetGCPs        (metadata['gcps'], metadata['gcpproj'])

    def write_geotiff_block(self, data, block_col, block_row, band=0):
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

        if not self._handle:
            time.sleep(0.5) # Sleep a short time then check again in case of race conditions.
            if not self._handle:
                raise Exception('Error: Data block added before initialization!')

        # Grab the lock and append to it.
        with self._writeQueueLock:
            self._writeQueue.append((data, block_col, block_row, band))

    def finish_writing_geotiff(self):
        '''Call when we have finished writing a geotiff file.'''

        if not self._handle:
            return

        MAX_WAIT_TIME = 20  # Wait up to this long to finish writing tiles.
        SLEEP_TIME    = 0.5 # Wait interval
        totalWait     = 0.0
        print('Finishing TIFF writing...')
        while True:
            with self._writeQueueLock:
                numTiles = len(self._writeQueue)
            if numTiles == 0:
                print('All tiles have been written!')
                break
            else:
                if totalWait >= MAX_WAIT_TIME:
                    print('Waited too long to finish, forcing shutdown!')
                    break # Waited long enough, force shutdown.
                else:
                    print('Waiting on '+str(numTiles)+' writes to finish...')
                    totalWait += SLEEP_TIME # Wait a bit longer
                    time.sleep(SLEEP_TIME)

        print('Clearing the write queue...')

        # Make sure that the write queue is empty.
        with self._writeQueueLock:
            self._writeQueue = []

        # Finish whatever we are writing, then close the handle.
        print('Closing TIFF handle...')
        band = self._handle.GetRasterBand(1)
        band.FlushCache()
        self._handle = None
        print('TIFF handle is gone.')
