"""
Script test out the image chunk generation calls.
"""
import sys
import argparse

import numpy as np
import tensorflow as tf
from osgeo import gdal

from delta.config import config
from delta.imagery.sources.worldview import WorldviewImage
from delta.imagery.sources.tiff import TiffWriter
from delta.imagery import rectangle
from delta.ml.train import load_keras_model

def main(argsIn):
    parser = argparse.ArgumentParser(usage='classify.py [options]')

    parser.add_argument("--keras-model", dest="keras_model", required=True,
                        help="Path to the saved Keras model.")

    parser.add_argument("--input-image", dest="input_image", required=True,
                        help="Path to the input image file.")

    options = config.parse_args(parser, argsIn)

    try:
        model = load_keras_model(options.keras_model, config.num_gpus())
    except AttributeError:
        model = tf.keras.models.load_model(options.keras_model)
    image = WorldviewImage(options.input_image)

    block_size_x = 1024
    block_size_y = 1024
    cs = config.chunk_size()

    # Set up the output image
    input_bounds = rectangle.Rectangle(3000, 5000, width=2000, height=2000)

    with TiffWriter('out.tiff', input_bounds.width() - cs + 1, input_bounds.height() - cs + 1, 3,
                    gdal.GDT_Byte, block_size_x, block_size_y,
                    0, image.metadata()) as writer:
        output_rois = input_bounds.make_tile_rois(block_size_x + cs - 1, block_size_y + cs - 1,
                                                  include_partials=True, overlap_amount=cs - 1)

        def callback_function(roi, data):
            """Callback function to write the first channel to the output file."""

            # Figure out some ROI positioning values
            block_x = (roi.min_x - input_bounds.min_x) // block_size_x
            block_y = (roi.min_y - input_bounds.min_y) // block_size_y
            out_shape = (data.shape[0] - cs + 1, data.shape[1] - cs + 1)
            print(data.shape)
            chunks = np.lib.stride_tricks.as_strided(data, shape=(out_shape[0], out_shape[1], cs, cs, data.shape[2]),
                                                     strides=(data.strides[0], data.strides[1], data.strides[0],
                                                              data.strides[1], data.strides[2]),
                                                     writeable=False)
            print(chunks.shape)
            chunks = np.reshape(chunks, (-1, cs, cs, data.shape[2]))
            print(chunks.shape)
            predictions = model.predict(chunks, verbose=0)
            print(predictions.shape)
            best = np.argmax(predictions, axis=1)
            image = np.reshape(best, (out_shape[0], out_shape[1]))
            print(np.unique(image))

            # Loop on bands
            for band in range(3):
                writer.write_block(image * 75, block_x, block_y, band)

        image.process_rois(output_rois, callback_function, show_progress=True)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
