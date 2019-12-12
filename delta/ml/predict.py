import numpy as np
from numpy.lib.stride_tricks import as_strided

from delta.imagery import rectangle

def predict(model, cs, image, input_bounds=None, show_progress=False):
    block_size_x = 256
    block_size_y = 256

    # Set up the output image
    if not input_bounds:
        input_bounds = rectangle.Rectangle(0, 0, width=image.width(), height=image.height())

    result = np.zeros((input_bounds.width() - cs + 1, input_bounds.height() - cs + 1), dtype=np.uint8)

    def callback_function(roi, data):
        """Callback function to write the first channel to the output file."""

        # Figure out some ROI positioning values
        block_x = (roi.min_x - input_bounds.min_x) // block_size_x
        block_y = (roi.min_y - input_bounds.min_y) // block_size_y
        out_shape = (data.shape[0] - cs + 1, data.shape[1] - cs + 1)
        chunks = as_strided(data, shape=(out_shape[0], out_shape[1], cs, cs, data.shape[2]),
                            strides=(data.strides[0], data.strides[1], data.strides[0],
                                     data.strides[1], data.strides[2]),
                            writeable=False)
        chunks = np.reshape(chunks, (-1, cs, cs, data.shape[2]))
        predictions = model.predict(chunks, verbose=0)
        best = np.argmax(predictions, axis=1)
        image = np.reshape(best, (out_shape[0], out_shape[1]))
        result[block_x * block_size_x : block_x * block_size_x + out_shape[0],
               block_y * block_size_y : block_y * block_size_y + out_shape[1]] = image

    output_rois = input_bounds.make_tile_rois(block_size_x + cs - 1, block_size_y + cs - 1,
                                              include_partials=True, overlap_amount=cs - 1)

    image.process_rois(output_rois, callback_function, show_progress=show_progress)
    return result
