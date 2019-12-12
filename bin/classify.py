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
from delta.ml.train import load_keras_model
from delta.ml.predict import predict

def main(args):
    parser = argparse.ArgumentParser(usage='classify.py [options]')

    parser.add_argument("--model", dest="model", required=True,
                        help="Path to the saved Keras model.")

    options = config.parse_args(parser, args, labels=False, ml=True)

    try:
        model = load_keras_model(options.model, config.num_gpus())
    except AttributeError:
        model = tf.keras.models.load_model(options.model)

    colors = np.array([[0x0, 0x0, 0x0],
                       [0xf6, 0xef, 0xf7],
                       [0x67, 0xa9, 0xcf],
                       [0xbd, 0xc9, 0xe1],
                       [0x02, 0x81, 0x8a]], dtype=np.uint8)

    ds_config = config.dataset()
    for i in range(ds_config.num_images()):
        image = loader.load_image(ds_config, i)
        out_name = os.path.splitext(os.path.basename(ds_config.image(i)))[0] + '_predicted.tiff'
        result = predict(model, config.chunk_size(), image, show_progress=True)
        tiff.write_tiff(out_name, colors[result], metadata=image.metadata())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
