"""
Script test out the image chunk generation calls.
"""
import os.path
import sys
import argparse

import tensorflow as tf
import numpy as np

from delta.config import config
from delta.imagery.sources import tiff
from delta.imagery.sources import loader
from delta.ml import predict

def main(args):
    parser = argparse.ArgumentParser(usage='classify.py [options]')

    parser.add_argument("--model", dest="model", required=True,
                        help="Path to the saved Keras model.")
    parser.add_argument("--validate", dest="validate", required=False, default=False,
                        action="store_true", help="Compare to specified labels.")

    options = config.parse_args(parser, args, labels=True, ml=True)

    model = tf.keras.models.load_model(options.model)

    colors = np.array([[0x0, 0x0, 0x0],
                       [0x67, 0xa9, 0xcf],
                       [0xf6, 0xef, 0xf7],
                       [0xbd, 0xc9, 0xe1],
                       [0x02, 0x81, 0x8a]], dtype=np.uint8)
    error_colors = np.array([[0x0, 0x0, 0x0],
                             [0xFF, 0x00, 0x00]], dtype=np.uint8)

    cs = config.chunk_size()
    ds_config = config.dataset()
    for i in range(ds_config.num_images()):
        image = loader.load_image(ds_config, i)
        base_name = os.path.splitext(os.path.basename(ds_config.image(i)))[0]
        if options.validate:
            label = loader.load_label(ds_config, i)
            (result, error_image, accuracy) = predict.predict_validate(model, cs, image, label, show_progress=True)
            print('%.2g%% Correct: %s' % (accuracy * 100, ds_config.image(i)))
            colored_error = error_colors[error_image.astype('uint8')]
            tiff.write_tiff(base_name + '_errors.tiff', colored_error, metadata=image.metadata())
        else:
            result = predict.predict(model, cs, image, show_progress=True)
        tiff.write_tiff(base_name + '_predicted.tiff', colors[result], metadata=image.metadata())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
