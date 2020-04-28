# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Classify input images given a model.
"""
import os.path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from delta.config import config
from delta.imagery.sources import tiff
from delta.imagery.sources import loader
from delta.ml import predict
import delta.ml.layers
import delta.imagery.imagery_config
import delta.ml.ml_config

def setup_parser(subparsers):
    sub = subparsers.add_parser('classify', help='Classify images given a model.')
    config.setup_arg_parser(sub, ['general', 'io', 'dataset'])

    sub.add_argument('--prob', dest='prob', action='store_true', help='Save image of class probabilities.')
    sub.add_argument('--autoencoder', dest='autoencoder', action='store_true', help='Classify with the autoencoder.')
    sub.add_argument('model', help='File to save the network to.')

    sub.set_defaults(function=main)

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

def ae_convert(data):
    return (data[:, :, [4, 2, 1]] * 256.0).astype(np.uint8)

def main(options):
    model = tf.keras.models.load_model(options.model, custom_objects=delta.ml.layers.ALL_LAYERS)

    colors = np.array([[0x0, 0x0, 0x0],
                       [0x67, 0xa9, 0xcf],
                       [0xf6, 0xef, 0xf7],
                       [0xbd, 0xc9, 0xe1],
                       [0x02, 0x81, 0x8a]], dtype=np.uint8)
    error_colors = np.array([[0x0, 0x0, 0x0],
                             [0xFF, 0x00, 0x00]], dtype=np.uint8)

    images = config.dataset.images()
    labels = config.dataset.labels()

    if options.autoencoder:
        labels = None
    for (i, path) in enumerate(images):
        image = loader.load_image(images, i)
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_image = tiff.DeltaTiffWriter('predicted_' + base_name + '.tiff')
        prob_image = None
        if options.prob:
            prob_image = tiff.DeltaTiffWriter('prob_' + base_name + '.tiff')
        error_image = None
        if labels:
            error_image = tiff.DeltaTiffWriter('errors_' + base_name + '.tiff')

        label = None
        if labels:
            label = loader.load_image(config.dataset.labels(), i)
        if options.autoencoder:
            label = image
            predictor = predict.ImagePredictor(model, output_image, True, (ae_convert, np.uint8, 3))
        else:
            predictor = predict.LabelPredictor(model, output_image, True, colormap=colors, prob_image=prob_image,
                                               error_image=error_image, error_colors=error_colors)

        try:
            predictor.predict(image, label)
        except KeyboardInterrupt:
            print('\nAborted.')
            return 0

        if labels:
            cm = predictor.confusion_matrix()
            print('%.2g%% Correct: %s' % (np.sum(np.diag(cm)) / np.sum(cm) * 100, path))
            save_confusion(cm, 'confusion_' + base_name + '.pdf')

        if options.autoencoder:
            tiff.write_tiff('orig_' + base_name + '.tiff', ae_convert(image.read()),
                            metadata=image.metadata())
    return 0
