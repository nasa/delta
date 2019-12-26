"""
Classify input images given a model.
"""
import os.path

import tensorflow as tf
import numpy as np

from delta.config import config
from delta.imagery.sources import tiff
from delta.imagery.sources import loader
from delta.ml import predict

def setup_parser(subparsers):
    sub = subparsers.add_parser('classify', description='Classify images given a model.')

    sub.add_argument('model', help='File to save the network to.')
    sub.add_argument("--model", dest="model", required=True,
                     help="Path to the saved Keras model.")
    sub.add_argument("--validate", dest="validate", required=False, default=False,
                     action="store_true", help="Compare to specified labels.")

    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, labels=True, ml=True)

def main(options):
    model = tf.keras.models.load_model(options.model)

    colors = np.array([[0x0, 0x0, 0x0],
                       [0x67, 0xa9, 0xcf],
                       [0xf6, 0xef, 0xf7],
                       [0xbd, 0xc9, 0xe1],
                       [0x02, 0x81, 0x8a]], dtype=np.uint8)
    error_colors = np.array([[0x0, 0x0, 0x0],
                             [0xFF, 0x00, 0x00]], dtype=np.uint8)

    cs = config.chunk_size()
    images = config.images()
    for (path, i) in enumerate(images):
        image = loader.load_image(images, i)
        base_name = os.path.splitext(os.path.basename(path))[0]
        if options.validate:
            label = loader.load_image(config.labels(), i)
            (result, error_image, accuracy) = predict.predict_validate(model, cs, image, label, show_progress=True)
            print('%.2g%% Correct: %s' % (accuracy * 100, path))
            colored_error = error_colors[error_image.astype('uint8')]
            tiff.write_tiff(base_name + '_errors.tiff', colored_error, metadata=image.metadata())
        else:
            result = predict.predict(model, cs, image, show_progress=True)
        tiff.write_tiff(base_name + '_predicted.tiff', colors[result], metadata=image.metadata())
    return 0
