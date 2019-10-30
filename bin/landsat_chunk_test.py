#!/usr/bin/python
"""
Script test out the image chunk generation calls.
"""
# pylint: disable=unsubscriptable-object
import os
import sys
import argparse

from delta.imagery import rectangle
from delta.imagery import utilities
from delta.imagery.sources import landsat
from delta.imagery.image_reader import MultiTiffFileReader
from delta.imagery.image_writer import write_simple_image

#------------------------------------------------------------------------------

def main(argsIn):

    try:

        usage  = "usage: landsat_chunk_test [options]"
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--mtl-path", dest="mtl_path", default=None,
                            help="Path to the MTL file in the same folder as Landsat image band files.")

        parser.add_argument("--image-path", dest="image_path", default=None,
                            help="Instead of using an MTL file, just load this one image.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Write output chunk files to this folder.")

        parser.add_argument("--output-band", dest="output_band", type=int, default=0,
                            help="Only chunks from this band are written to disk.")

        parser.add_argument("--num-threads", dest="num_threads", type=int, default=1,
                            help="Number of threads to use for parallel image loading.")

        parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=1024,
                            help="The length of each side of the output image chunks.")

        parser.add_argument("--chunk-overlap", dest="chunk_overlap", type=int, default=0,
                            help="The amount of overlap of the image chunks.")

        options = parser.parse_args(argsIn)

    except argparse.ArgumentError:
        print(usage)
        return -1

    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)

    if options.mtl_path:

        # Get all of the TOA coefficients and input file names
        data = landsat.parse_mtl_file(options.mtl_path)

        input_folder = os.path.dirname(options.mtl_path)

        input_paths = []
        for fname in data['FILE_NAME']:
            input_path = os.path.join(input_folder, fname)
            input_paths.append(input_path)

    else: # Just load the specified image
        input_paths = [options.image_path]

    # Open the input image and get information about it
    input_reader = MultiTiffFileReader()
    input_reader.load_images(input_paths)
    (num_cols, num_rows) = input_reader.image_size()

    # Process the entire input image(s) into chunks at once.
    roi = rectangle.Rectangle(0,0,width=num_cols,height=num_rows)
    chunk_data = input_reader.parallel_load_chunks(roi, options.chunk_size,
                                                   options.chunk_overlap, options.num_threads)

    # For debug output, write each individual chunk to disk from a single band
    shape = chunk_data.shape
    num_chunks = shape[0]
    num_bands  = shape[1]
    print('num_chunks = ' + str(num_chunks))
    print('num_bands = ' + str(num_bands))

    for chunk in range(0,num_chunks):
        data = chunk_data[chunk,options.output_band,:,:]
        #print('data.shape = ' + str(data.shape))

        # Dump to disk
        output_path = os.path.join(options.output_folder, 'chunk_'+str(chunk) + '.tif')
        write_simple_image(output_path, data, data_type=utilities.numpy_dtype_to_gdal_type(chunk_data.dtype))

        #raise Exception('DEBUG')

    print('Landsat chunker is finished.')
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
