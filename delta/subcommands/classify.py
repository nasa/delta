"""
Classify input images given a model.
"""
import os.path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from delta.config import config
from delta.imagery.sources import tiff
from delta.imagery.sources import loader
from delta.ml import predict

def setup_parser(subparsers):
    sub = subparsers.add_parser('classify', help='Classify images given a model.')

    sub.add_argument('--prob', dest='prob', action='store_true', help='Save image of class probabilities.')
    sub.add_argument('model', help='File to save the network to.')

    sub.set_defaults(function=main)
    # TODO: move chunk_size into model somehow
    config.setup_arg_parser(sub, labels=True, train=True)

def save_confusion(cm, filename):
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    image = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('inferno'))
    ax.set_title('Confusion Matrix')
    f.colorbar(image)
    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xlim(-0.5, cm.shape[0]-0.5)
    ax.set_ylim(-0.5, cm.shape[0]-0.5)
    m = cm.max()
    total = cm.sum()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, '%d\n%.2g%%' % (cm[i, j], cm[i, j] / total * 100), horizontalalignment='center',
                    color='white' if cm[i, j] < m / 2 else 'black')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicated Label')
    f.savefig(filename)

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
    labels = config.labels()
    for (i, path) in enumerate(images):
        image = loader.load_image(images, i)
        base_name = os.path.splitext(os.path.basename(path))[0]
        if labels:
            label = loader.load_image(config.labels(), i)
            (result, error_image, cm) = predict.predict_validate(model, cs, image, label,
                                                                 config.classes(), probabilities=options.prob,
                                                                 show_progress=True)
            print('%.2g%% Correct: %s' % (np.sum(np.diag(cm)) / np.sum(cm) * 100, path))
            colored_error = error_colors[error_image.astype('uint8')]
            tiff.write_tiff(base_name + '_errors.tiff', colored_error, metadata=image.metadata())
            save_confusion(cm, base_name + '_confusion.pdf')
        else:
            result = predict.predict(model, cs, image, config.classes(), probabilities=options.prob, show_progress=True)
        if options.prob:
            tiff.write_tiff(base_name + '_prob.tiff', result, metadata=image.metadata())
        else:
            tiff.write_tiff(base_name + '_predicted.tiff', colors[result], metadata=image.metadata())
    return 0
