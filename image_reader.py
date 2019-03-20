#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __BEGIN_LICENSE__
#  Copyright (c) 2009-2013, United States Government as represented by the
#  Administrator of the National Aeronautics and Space Administration. All
#  rights reserved.
#
#  The NGT platform is licensed under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance with the
#  License. You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# __END_LICENSE__

"""
Testing for Land Classifier
"""
import sys, os, subprocess
import string, argparse, threading, time
import copy
import psutil
import math

from osgeo import gdal
import numpy as np

from utilities import Rectangle

#------------------------------------------------------------------------------

BYTES_PER_MB = 1024*1024

def get_num_bytes_from_gdal_type(gdal_type):
    """Return the number of bytes for one pixel (one band) in a GDAL type."""
    results = {
        gdal.GDT_Byte:    1,
        gdal.GDT_UInt16:  2,
        gdal.GDT_UInt32:  4,
        gdal.GDT_Float32: 4
    }
    return results.get(gdal_type)


#=============================================================================
# Image writer class -> Move to another file!

class TiffWriter:

    """Class to manage block writes to a Geotiff file.
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
        
        self._tileWidth  = 256
        self._tileHeight = 256

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


    def _internal_writer(self):
        '''Internal thread that writes blocks as they become available.'''

        SLEEP_TIME = 0.2

        blockCounter = 0
        print('Starting block writing loop')
        while True:

            # Check the write queue
            with self._writeQueueLock:
                noTiles = (len(self._writeQueue) == 0)
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
                self._write_geotiff_block_internal(parts[0], parts[1], parts[2])
                blockCounter += 1
            except Exception as e:
                print(str(e))
                print('Caught exception writing: ' + str(parts[1] +', '+ str(parts[2])))
                
        print('Write thread ended after writing ' + str(blockCounter) +' blocks.')


    def _write_geotiff_block_internal(self, data, blockCol, blockRow):
        '''Write a single block to disk in a geotiff file'''

        if not self._handle:
            raise Exception('Failed to write block: output file not initialized!')

        band = self._handle.GetRasterBand(1)
        #stuff = dir(band)
        #for s in stuff:
        #    print(s)
        #band.WriteBlock(blockCol, blockRow, data) This function is not supported!
        
        bSize = band.GetBlockSize()
        band.WriteArray(data, blockCol*bSize[0], blockRow*bSize[1])
        
        #band.FlushBlock(blockCol, blockRow) This function is not supported!
        band.FlushCache() # TODO: Call after every tile?


    def init_output_geotiff(self, path, numRows, numCols, noDataValue,
                          tileWidth=256, tileHeight=256):
        '''Set up a geotiff file for writing and return the handle.'''
        # TODO: Copy metadata from the source file.

        self._tileHeight = tileHeight
        self._tileWidth  = tileWidth

        # Constants
        dataType = gdal.GDT_Byte
        numBands = 1
        # TODO: The block size must by synced up with the NN size!
        options = ['COMPRESS=LZW', 'BigTIFF=IF_SAFER', 'TILED=YES',
                  'BLOCKXSIZE='+str(self._tileWidth),
                  'BLOCKYSIZE='+str(self._tileHeight)]

        print('Starting TIFF driver...')
        driver = gdal.GetDriverByName('GTiff')
        self._handle = driver.Create( path, numCols, numRows, numBands, dataType, options)

        #handle.SetGeoTransform(GeoT)
        #handle.SetProjection( Projection.ExportToWkt() )
        if (noDataValue != None):
            self._handle.GetRasterBand(1).SetNoDataValue(noDataValue)

    def get_tile_size(self):
        return (self._tileWidth, self._tileHeight)

    def write_geotiff_block(self, data, blockCol, blockRow):
        '''Add a tile write command to the queue.'''

        if ( (data.shape[0] != self._tileHeight) or 
             (data.shape[1] != self._tileWidth )  ):
            raise Exception('Error: Data block size is ' + str(data.shape) +
                            ', file block size is ' + str((self._tileWidth, self._tileHeight)))

        if not self._handle:
            time.sleep(0.5) # Sleep a short time then check again in case of race conditions.
            if not self._handle:
                raise Exception('Error: Data block added before initialization!')

        # Grab the lock and append to it.
        with self._writeQueueLock:
              self._writeQueue.append((data, blockCol, blockRow))

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
            self._writeQueue.clear()

        # Finish whatever we are writing, then close the handle.
        print('Closing TIFF handle...')
        band = self._handle.GetRasterBand(1)
        band.FlushCache()
        self._handle = None
        print('TIFF handle is gone.')

#=============================================================================

class TiffReader:
    """Wrapper class to help read image data from GeoTiff files"""

    def __init__(self):
        self._handle = None

    def __del__(self):
        self.close_image()


    def open_image(self, path):

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
      return get_num_bytes_from_gdal_type(band_handle.DataType)

    def get_block_info(self, band):
        """Returns ((block height, block width), (num blocks x, num blocks y))"""
        band_handle = self._handle.GetRasterBand(band)
        block_size  = band_handle.GetBlockSize()
        
        # TODO: How to handle fractional values?
        num_blocks_x = int(self._handle.RasterXSize / block_size[0])
        num_blocks_y = int(self._handle.RasterYSize / block_size[1])
        
        return (block_size, (num_blocks_x, num_blocks_y))

    def get_block_aligned_read_roi(self, desired_roi):
        """Returns the block aligned pixel region to read in a Rectangle format
           to get the requested data region while respecting block boundaries.
        """
        band = 1
        (block_size, num_blocks) = self.get_block_info(band)
        start_block_x = math.floor(desired_roi.min_x / block_size[0])
        start_block_y = math.floor(desired_roi.min_y / block_size[1])
        stop_block_x  = math.floor(desired_roi.max_x / block_size[0])
        stop_block_y  = math.floor(desired_roi.max_y / block_size[1])

        start_col = start_block_x * block_size[0]
        start_row = start_block_y * block_size[1]
        num_cols  = (stop_block_x - start_block_x) * block_size[0]
        num_rows  = (stop_block_y - start_block_y) * block_size[1]
        
        return Rectangle(start_col, start_row, width=num_cols, height=num_rows)

    def read_pixels(self, roi, band):
        """Reads in the requested region of the image."""
        band_handle = self._handle.GetRasterBand(band)
        data = band_handle.ReadAsArray(roi.min_x, roi.min_y, roi.width(), roi.height())
        return data

def dummy_callback(arg1, arg2, arg3):
    print('Callback executed for: ' + str(arg1))

class MultiTiffFileReader():
    """Class to synchronize loading of multiple pixel-matched files"""
  
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
        """Coles all loaded images."""
        for h in self._image_handles:
            h.close_image()
        self._image_handles = []
    
    # TODO: Error checking!
    def image_size(self):
        return self._image_handles[0].image_size()
    
    def estimate_memory_usage(self, roi):
        """Estimate the amount of memory (in MB) that will be used to store the
           requested pixel roi.
        """
        total_size = 0
        for image in self._image_handles:
            for band in range(1,image.num_bands()+1):
                total_size += image.num_bands() * roi.area() * image.get_bytes_per_pixel(band)
        return total_size / BYTES_PER_MB

    def sleep_until_mem_free_for_roi(self, roi):
        '''Sleep until enough free memory exists to load this ROI'''
        # TODO: This will have to be more sophisticated.
        WAIT_TIME_SECS = 5
        mb_needed = self.estimate_memory_usage(roi)
        mb_free   = 0
        while mb_free < mb_needed:
          mb_free = psutil.virtual_memory().free / BYTES_PER_MB
          if mb_free < mb_needed:
              print('Need %d MB to load the next ROI, but only have %d MB free. Sleep for %d seconds...'
                    % (mb_needed, mb_free, WAIT_TIME_SECS))
              time.sleep(WAIT_TIME_SECS)

    def process_rois(self, requested_rois, callback_function):
        """Process the given region broken up into blocks using the callback function.
           Each block will get the image data from each input image passed into the function.
           Function definition TBD!
           Blocks that go over the image boundary will be passed as partial image blocks.
        """

        if not self._image_handles:
            raise Exception('Cannot process region, no images loaded!')
        first_image = self._image_handles[0]

        ## For each block in row-major order, record the ROI in the image.
        #block_rois = []
        #for r in range(0,num_block_rows):
        #    for c in range(0,num_block_cols):
        #        this_col = col + c*block_width
        #        this_row = row + r*block_width
        #        this_roi = Rectangle(this_col, this_row,
        #                             width=block_width, height=block_height)
        #        block_rois.append(this_roi)
        block_rois = copy.copy(requested_rois)

        print('Ready to process ' + str(len(requested_rois)) +' tiles.')

        # Loop until we have processed all of the blocks.
        while len(block_rois) > 0:

            # For the next (output) block, figure out the (input block) aligned
            # data read that we need to perform to get it.
            read_roi = first_image.get_block_aligned_read_roi(block_rois[0])
            print('Want to process ROI: ' + str(block_rois[0]))
            print('Reading input ROI: ' + str(read_roi))

            # If we don't have enough memory to load the ROI, wait here until we do.
            self.sleep_until_mem_free_for_roi(read_roi)

            # TODO: We need a way to make sure the bands for certain image types always end up
            #       in the same order!
            # Read in all of the data for this region, appending all bands for
            # each input image.
            data_vec = []
            for image in self._image_handles:
                for band in range(1,image.num_bands()+1):
                    print('Data read from image ' + str(image) + ', band ' + str(band))
                    data_vec.append(image.read_pixels(read_roi, band))

            # Loop through the remaining ROIs and apply the callback function to each
            # ROI that is contained in the section we read in.
            index = 0
            num_processed = 0
            while index < len(block_rois):
                roi = block_rois[index]
                if not read_roi.contains(roi):
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

            print('From the read ROI, was able to process ' + str(num_processed) +' tiles.')

        print('Finished processing tiles!')


#=============================================================================



def main(argsIn):

    try:

        # Use parser that ignores unknown options
        usage  = "usage: image_reader [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-path", dest="inputPath", default=None,
                                              help="Input path")

        parser.add_argument("--output-path", dest="outputPath", default=None,
                                              help="Output path.")


        # This call handles all the parallel_mapproject specific options.
        options = parser.parse_args(argsIn)

        # Check the required positional arguments.

    except argparse.ArgumentError as msg:
        raise Usage(msg)


    image = TiffReader()
    image.open_image(options.inputPath)

    band = 1
    (nCols, nRows) = image.image_size()
    (bSize, (numBlocksX, numBlocksY)) = image.get_block_info(band)
    noData = image.nodata_value()

    #print('nBands = %d, nRows = %d, nCols = %d' % (nBands, nRows, nCols))
    #print('noData = %s, dType = %d, bSize = %d, %d' % (str(noData), dType, bSize[0], bSize[1]))

    

    input_reader = MultiTiffFileReader()
    input_reader.load_images([options.inputPath])
    (nCols, nRows) = input_reader.image_size()

    #print('Num blocks = %f, %f' % (numBlocksX, numBlocksY))

    # TODO: Will we be faster using this method? Or ReadAsArray? Or ReadRaster?
    #data = band.ReadBlock(0,0) # Reads in as 'bytes' or raw data
    #print(type(data))
    #print('len(data) = ' + str(len(data)))
    
    #data = band.ReadAsArray(0, 0, bSize[0], bSize[1]) # Reads as numpy array
    ##np.array()
    #print(type(data))
    ##print('len(data) = ' + str(len(data)))
    #print('data.shape = ' + str(data.shape))
    

    output_tile_width = bSize[0]
    output_tile_height = 32

    # Make a list of output ROIs
    numBlocksX = 1
    numBlocksY = int(3744 / output_tile_height)

    #stuff = dir(band)
    #for s in stuff:
    #    print(s)

    print('Testing image duplication!')
    writer = TiffWriter()
    writer.init_output_geotiff(options.outputPath, nRows, nCols, noData,
                             tileWidth=output_tile_width, tileHeight=output_tile_height)

    # Setting up output ROIs
    output_rois = []
    for r in range(0,numBlocksY):
        for c in range(0,numBlocksX):
            
            roi = Rectangle(c*output_tile_width, r*output_tile_height,
                            width=output_tile_width, height=output_tile_height)
            output_rois.append(roi)
            #print(roi)
            #print(band)
            #data = image.read_pixels(roi, band)
            #writer.write_geotiff_block(data, c, r)
            
    
    def callback_function(output_roi, read_roi, data_vec):
      
        print('For output roi: ' + str(output_roi) +' got read_roi ' + str(read_roi))
        print(data_vec[0].shape)
        
        col = output_roi.min_x / 5616 # Hack for testing!
        row = output_roi.min_y / 32
        writer.write_geotiff_block(data_vec[0], col, row)
        
            
    print('Writing TIFF blocks...')
    input_reader.process_rois(output_rois, callback_function)
            
                        
            
    print('Done sending in blocks!')
    writer.finish_writing_geotiff()
    print('Done duplicating the image!')

    time.sleep(2)
    print('Cleaning up the writer!')
    writer.cleanup()

    image = None # Close the image

    print('Script is finished.')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    
    
    
    
    
    
    
    
    
    
    
    
