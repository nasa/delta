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

'''
Testing for Land Classifier
'''
import sys, os, subprocess
import string, argparse, threading, time

from osgeo import gdal
import numpy as np

#------------------------------------------------------------------------------


#=============================================================================
# Image writer class -> Move to another file!

class TiffWriter:

    '''Class to manage block writes to a Geotiff file.
       TODO: Make sure everything works with opening and closing files in sequence!
    '''
    

    def __init__(self):
        self._handle         = None
        self._shutdown       = False # Turned on in __del__()
        self._writeQueue     = []
        self._writeQueueLock = threading.Lock()

        print('Launching write thread...')
        self._writerThread = threading.Thread(target=self._internalWriter)
        
        self._writerThread.start()
        
        self._tileWidth  = 256
        self._tileHeight = 256

    def __del__(self):
        '''Try this just in case, but it may not work!'''
        self.cleanup()

    def cleanup(self):
        print('cleanup called.')
        self.finishWritingGeotiff()

        # Shut down the writing thread, giving it ten minutes to finish.
        TIMEOUT = 60*10
        self._shutdown = True
        print('Terminating write thread...')
        self._writerThread.join(TIMEOUT)


    def _internalWriter(self):
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
                    print('Writing thread is sleeping')
                    time.sleep(SLEEP_TIME) # Wait
                    continue

            if not self._handle:
                print('ERROR: Trying to write image block before initialization!')
                break

            # Write out the block
            try:
                self._writeGeotiffBlockInternal(parts[0], parts[1], parts[2])
                blockCounter += 1
            except Exception as e:
                print(str(e))
                print('Caught exception writing: ' + str(parts[1] +', '+ str(parts[2])))
                
        print('Write thread ended after writing ' + str(blockCounter) +' blocks.')


    def _writeGeotiffBlockInternal(self, data, blockCol, blockRow):
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


    def initOutputGeotiff(self, path, numRows, numCols, noDataValue,
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

    def getTileSize(self):
        return (self._tileWidth, self._tileHeight)

    def writeGeotiffBlock(self, data, blockCol, blockRow):
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

    def finishWritingGeotiff(self):
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


    image  = gdal.Open(options.inputPath)

    nBands = image.RasterCount      # how many bands, to help you loop
    nRows  = image.RasterYSize      # how many rows
    nCols  = image.RasterXSize      # how many columns
    
    
    print('nBands = %d, nRows = %d, nCols = %d' % (nBands, nRows, nCols))
    
    band   = image.GetRasterBand(1) # 1 based, for this example only the first
    noData = band.GetNoDataValue()  # this might be important later
    dType  = band.DataType          # the datatype for this band
    bSize  = band.GetBlockSize()

    print('noData = %s, dType = %d, bSize = %d, %d' % (str(noData), dType, bSize[0], bSize[1]))

    # TODO: How to handle fractional values?
    numBlocksX = int(nCols / bSize[0])
    numBlocksY = int(nRows / bSize[1])
    print('Num blocks = %f, %f' % (numBlocksX, numBlocksY))

    # TODO: Will we be faster using this method? Or ReadAsArray? Or ReadRaster?
    #data = band.ReadBlock(0,0) # Reads in as 'bytes' or raw data
    #print(type(data))
    #print('len(data) = ' + str(len(data)))
    
    #data = band.ReadAsArray(0, 0, bSize[0], bSize[1]) # Reads as numpy array
    ##np.array()
    #print(type(data))
    ##print('len(data) = ' + str(len(data)))
    #print('data.shape = ' + str(data.shape))
    

    #stuff = dir(band)
    #for s in stuff:
    #    print(s)

    print('Testing image duplication!')
    writer = TiffWriter()
    writer.initOutputGeotiff(options.outputPath, nRows, nCols, noData,
                             tileWidth=5616, tileHeight=16)

    print('Writing TIFF blocks...')
    for r in range(0,numBlocksY):
        for c in range(0,numBlocksX):
            #print(r)
            #print(c)
            data = band.ReadAsArray(c*bSize[0], r*bSize[1],bSize[0], bSize[1])
            #print(type(data))
            #print('data.shape = ' + str(data.shape))
            writer.writeGeotiffBlock(data, c, r)
    print('Done sending in blocks!')
    writer.finishWritingGeotiff()
    print('Done duplicating the image!')

    time.sleep(2)
    print('Cleaning up the writer!')
    writer.cleanup()

    image = None # Close the image

    print('Script is finished.')

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
    
    
    
    
    
    
    
    
    
    
    
    
