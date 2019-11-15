"""
Higher level functions for dealing with large images.
"""

from delta.imagery import rectangle, utilities
from delta.imagery.sources import tiff


def apply_function_to_file(input_path, output_path, user_function, tile_size=(0,0), nodata_out=0):
    """Apply the given function to the entire input image and write the
       result into the output path.  The function is applied to each tile of data.
       The function should either take just a data argument (for single band images)
       or take the data argument and a band index (for multi-band images).
    """

    # Set up an image reader and get information from it
    input_reader = tiff.MultiTiffFileReader([input_path])
    (num_cols, num_rows) = input_reader.image_size()
    num_bands  = input_reader.num_bands()
    (block_size_in, _) = input_reader.get_block_info(band=1)
    input_metadata = input_reader.get_all_metadata()

    # Use the input tile size for the block size unless the user specified one.
    X, Y = (0, 1)
    block_size_out = block_size_in
    if tile_size[X] > 0:
        block_size_out[X] = int(tile_size[X])
    if tile_size[Y] > 0:
        block_size_out[Y] = int(tile_size[Y])

    # Set up the output image
    writer = tiff.TiffWriter(output_path, num_rows, num_cols, num_bands,
                             utilities.get_gdal_data_type('float'),
                             block_size_out[X], block_size_out[Y],
                             nodata_out, input_metadata)

    input_bounds = rectangle.Rectangle(0, 0, width=num_cols, height=num_rows)
    output_rois = input_bounds.make_tile_rois(block_size_out[X], block_size_out[Y], include_partials=True)

    def callback_function(output_roi, data):
        """Callback function to write the first channel to the output file."""

        # Figure out some ROI positioning values
        block_col = output_roi.min_x / block_size_out[0]
        block_row = output_roi.min_y / block_size_out[1]

        # Loop on bands
        for band in range(0,num_bands):
            # Crop the desired data portion and apply the user function.
            if num_bands == 1:
                output_data = user_function(data[band, :, :])
            else:
                output_data = user_function(data[band, :, :], band)

            # Write out the result
            writer.write_block(output_data, block_col, block_row, band)

    input_reader.process_rois(output_rois, callback_function, show_progress=True)

    del writer
