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
Visualize the training data.
"""

import sys

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.config_parser import config_model, config_augmentation

done_plot = False
img_plots = []
lab_plots = []
figure = None

def plot_images(images, labels):
    global done_plot, figure #pylint: disable=global-statement
    if figure is None:
        figure, axarr = plt.subplots(len(images), images[0].shape[2] + 1)
        label_norm = matplotlib.colors.Normalize(0, len(config.dataset.classes) + 1)
        if len(axarr.shape) < 2:
            axarr = np.expand_dims(axarr, axis=0)
        for (j, image) in enumerate(images):
            img_plots.append([])
            lab_plots.append([])
            label = labels[j]
            for i in range(image.shape[2]):
                img = axarr[j, i].imshow(image[:, :, i], norm=matplotlib.colors.Normalize(0.0, 1.0))
                img_plots[j].append(img)
                lab = axarr[j, i].imshow(label, cmap='inferno', alpha=0.1, norm=label_norm)
                lab_plots[j].append(lab)
            lab = axarr[j, -1].imshow(label, cmap='inferno', norm=label_norm)
            lab_plots[j].append(lab)

        plt.subplots_adjust(bottom=0.15, right=0.90)
        axslide = plt.axes([0.15, 0.05, 0.70, 0.03])
        axcolor = plt.axes([0.92, 0.2, 0.06, 0.6])
        figure.colorbar(img_plots[0][0], cax=axcolor)
        slide = matplotlib.widgets.Slider(axslide, 'Label Alpha', 0.0, 1.0, valinit=0.0, valstep=0.05)

        def update_alpha(alpha):
            for row in lab_plots:
                for l in row[:-1]:
                    l.set_alpha(alpha)

        slide.on_changed(update_alpha)
        def on_press(event):
            global done_plot #pylint: disable=global-statement
            if event.key == 'q':
                sys.exit(0)
            done_plot = True
        figure.canvas.mpl_connect('key_press_event', on_press)
    else:
        for i in range(len(img_plots)): #pylint: disable=consider-using-enumerate
            image = images[i]
            for j in range(image.shape[2]):
                img_plots[i][j].set_data(image[:, :, j])
        for i in range(len(lab_plots)): #pylint: disable=consider-using-enumerate
            for l in lab_plots[i]:
                l.set_data(labels[i])

    done_plot = False
    while not done_plot:
        plt.waitforbuttonpress()
        if not plt.get_fignums():
            sys.exit(0)

def main(options):
    images = config.dataset.images()
    if not images:
        print('No images specified.', file=sys.stderr)
        return 1
    img = images.load(0)
    model = config_model(img.num_bands())
    temp_model = model()

    tile_size = config.io.tile_size()
    tile_overlap = None
    stride = config.train.spec().stride

    # compute input and output sizes
    if temp_model.input_shape[1] is None:
        in_shape = None
        out_shape = temp_model.compute_output_shape((0, tile_size[0], tile_size[1], temp_model.input_shape[3]))
        out_shape = out_shape[1:3]
        tile_overlap = (tile_size[0] - out_shape[0], tile_size[1] - out_shape[1])
    else:
        in_shape = temp_model.input_shape[1:3]
        out_shape = temp_model.output_shape[1:3]

    if options.autoencoder:
        ids = imagery_dataset.AutoencoderDataset(images, in_shape, tile_shape=tile_size,
                                                 tile_overlap=tile_overlap, stride=stride)
    else:
        labels = config.dataset.labels()
        if not labels:
            print('No labels specified.', file=sys.stderr)
            return 1
        ids = imagery_dataset.ImageryDataset(images, labels, out_shape, in_shape,
                                             tile_shape=tile_size, tile_overlap=tile_overlap,
                                             stride=stride)

    assert temp_model.input_shape[1] == temp_model.input_shape[2], 'Must have square chunks in model.'
    assert temp_model.input_shape[3] == ids.num_bands(), 'Model takes wrong number of bands.'
    tf.keras.backend.clear_session()

    #colormap = np.zeros(dtype=np.uint8, shape=(len(config.dataset.classes), 3))
    #for c in config.dataset.classes:
    #    print(len(config.dataset.classes), c.value)
    #    colormap[c.value][0] = (c.color >> 32) & 0xFF
    #    colormap[c.value][1] = (c.color >> 16) & 0xFF
    #    colormap[c.value][2] = c.color & 0xFF

    images = []
    labels = []
    PLOT_AT_ONCE=7

    for result in ids.dataset(config.dataset.classes.weights(), config_augmentation()):
        image = result[0].numpy()
        label = result[1].numpy()
        pw = (image.shape[0] - label.shape[0]) // 2
        ph = (image.shape[1] - label.shape[1]) // 2
        label = np.pad(label, ((pw, pw), (ph, ph), (0, 0)))
        images.append(image)
        labels.append(label)
        if len(images) == PLOT_AT_ONCE:
            plot_images(images, labels)
            images = []
            labels = []

    return 0
