"""
Test for GDAL I/O classes.
"""
import sys
import os
import argparse
import math

# TODO: Clean this up
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../delta')))

from imagery.image_reader import * #pylint: disable=W0614,W0401,C0413
from imagery.image_writer import * #pylint: disable=W0614,W0401,C0413

#------------------------------------------------------------------------------

# TODO: Make unit tests!

def main(argsIn):

    try:

        # Use parser that ignores unknown options
        usage  = "usage: image_reader [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument('input_paths', metavar='N', type=str, nargs='+',
                            help='Input files')

        #parser.add_argument("--input-path", dest="inputPath", default=None,
        #                                      help="Input path")

        parser.add_argument("--output-path", dest="outputPath", default=None,
                            help="Output path.")


        parser.add_argument("--tile-size", nargs=2, metavar=('tile_width', 'tile_height'),
                            dest='tile_size', default=[0,0], type=int,
                            help="Specify the output tile size")

        # This call handles all the parallel_mapproject specific options.
        options = parser.parse_args(argsIn)

        # Check the required positional arguments.

    except argparse.ArgumentError:
        print(usage)
        return -1

    #image = TiffReader()
    #image.open_image(options.inputPath)

    #band = 1
    #(nCols, nRows) = image.image_size()
    #(bSize, (numBlocksX, numBlocksY)) = image.get_block_info(band)
    #noData = image.nodata_value()

    #print('nBands = %d, nRows = %d, nCols = %d' % (nBands, nRows, nCols))
    #print('noData = %s, dType = %d, bSize = %d, %d' % (str(noData), dType, bSize[0], bSize[1]))

    band = 1
    input_reader = MultiTiffFileReader()
    input_reader.load_images(options.input_paths)
    (nCols, nRows) = input_reader.image_size()
    noData = input_reader.nodata_value()
    (bSize, (numBlocksX, numBlocksY)) = input_reader.get_block_info(band)

    input_bounds = Rectangle(0, 0, width=nCols, height=nRows)

    input_metadata = input_reader.get_all_metadata()
    #print('Read metadata: ' + str(input_metadata))

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

    # Use the input tile size unless the user specified one.
    output_tile_width  = bSize[0]
    output_tile_height = bSize[1]
    if options.tile_size[0] > 0:
        output_tile_width = options.tile_size[0]
    if options.tile_size[1] > 0:
        output_tile_height = options.tile_size[1]

    print('Using output tile size ' + str(output_tile_width) + ' by ' + str(output_tile_height))

    # Make a list of output ROIs
    numBlocksX = int(math.ceil(nCols / output_tile_width))
    numBlocksY = int(math.ceil(nRows / output_tile_height))

    #stuff = dir(band)
    #for s in stuff:
    #    print(s)

    print('Testing image duplication!')
    writer = TiffWriter()
    writer.init_output_geotiff(options.outputPath, nRows, nCols, noData,
                               tile_width=output_tile_width,
                               tile_height=output_tile_height,
                               metadata=input_metadata)

    # Setting up output ROIs
    output_rois = []
    for r in range(0,numBlocksY):
        for c in range(0,numBlocksX):

            # Get the ROI for the block, cropped to fit the image size.
            roi = Rectangle(c*output_tile_width, r*output_tile_height,
                            width=output_tile_width, height=output_tile_height)
            roi = roi.get_intersection(input_bounds)

            output_rois.append(roi)
            #print(roi)
            #print(band)
            #data = image.read_pixels(roi, band)
            #writer.write_geotiff_block(data, c, r)

    def callback_function(output_roi, read_roi, data_vec):
        """Callback function to write the first channel to the output file."""

        #print('For output roi: ' + str(output_roi) +' got read_roi ' + str(read_roi))
        #print('Data shape = ' + str(data_vec[0].shape))

        # Figure out the output block
        col = output_roi.min_x / output_tile_width
        row = output_roi.min_y / output_tile_height

        data = data_vec[0] # Just for testing

        # Figure out where the desired output data falls in read_roi
        x0 = output_roi.min_x - read_roi.min_x
        y0 = output_roi.min_y - read_roi.min_y
        x1 = x0 + output_roi.width()
        y1 = y0 + output_roi.height()

        # Crop the desired data portion and write it out.
        output_data = data[y0:y1, x0:x1]
        writer.write_geotiff_block(output_data, col, row)


    print('Writing TIFF blocks...')
    input_reader.process_rois(output_rois, callback_function)



    print('Done sending in blocks!')
    writer.finish_writing_geotiff()
    print('Done duplicating the image!')

    time.sleep(2)
    print('Cleaning up the writer!')
    writer.cleanup()

    print('Script is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
