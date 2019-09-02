"""
Higher level functions for dealing with large images.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta.imagery import rectangle #pylint: disable=C0413
from delta.imagery import utilities #pylint: disable=C0413
from delta.imagery.image_reader import * #pylint: disable=W0614,W0401,C0413
from delta.imagery.image_writer import * #pylint: disable=W0614,W0401,C0413


def apply_function_to_file(input_path, output_path, user_function, tile_size=(0,0), nodata_out=0):
    """Apply the given function to the entire input image and write the
       result into the output path.  The function is applied to each tile of data.
       The function should either take just a data argument (for single band images)
       or take the data argument and a band index (for multi-band images).
    """

    # Set up an image reader and get information from it
    input_reader = MultiTiffFileReader([input_path])
    (num_cols, num_rows) = input_reader.image_size()
    num_bands  = input_reader.num_bands()
    (block_size_in, num_blocks) = input_reader.get_block_info(band=1) #pylint: disable=W0612
    input_metadata = input_reader.get_all_metadata()

    # Use the input tile size for the block size unless the user specified one.
    X, Y = (0, 1)
    block_size_out = block_size_in
    if tile_size[X] > 0:
        block_size_out[X] = int(tile_size[X])
    if tile_size[Y] > 0:
        block_size_out[Y] = int(tile_size[Y])

    # Set up the output image
    writer = TiffWriter()
    writer.init_output_geotiff(output_path, num_rows, num_cols, nodata_out,
                               block_size_out[X], block_size_out[Y],
                               input_metadata,
                               utilities.get_gdal_data_type('float'),
                               num_bands)

    input_bounds = rectangle.Rectangle(0, 0, width=num_cols, height=num_rows)
    output_rois = input_bounds.make_tile_rois(block_size_out[X], block_size_out[Y], include_partials=True)

    def callback_function(output_roi, read_roi, data_vec):
        """Callback function to write the first channel to the output file."""

        # Figure out some ROI positioning values
        ((col, row), (x0, y0, x1, y1)) = get_block_and_roi(output_roi, read_roi, block_size_out)

        # Loop on bands
        for band in range(0,num_bands):
            data = data_vec[band]

            # Crop the desired data portion and apply the user function.
            if num_bands == 1:
                output_data = user_function(data[y0:y1, x0:x1])
            else:
                output_data = user_function(data[y0:y1, x0:x1], band)

            # Write out the result
            writer.write_geotiff_block(output_data, col, row, band)

    print('Writing TIFF blocks...')
    input_reader.process_rois(output_rois, callback_function)

    writer.finish_writing_geotiff()

    time.sleep(2)
    writer.cleanup()

    print('Done writing: ' + output_path)

    image = None # Close the image  #pylint: disable=W0612
