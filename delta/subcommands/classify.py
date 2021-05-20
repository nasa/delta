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

import sys
import time
import numpy as np
import matplotlib
import tensorflow as tf

from delta.config import config
from delta.config.extensions import image_writer
from delta.ml import predict
from delta.ml.io import load_model

matplotlib.use('Agg')
import matplotlib.pyplot as plt #pylint: disable=wrong-import-order,wrong-import-position,ungrouped-imports

def save_confusion(cm, class_labels, filename):
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    image = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('inferno'))
    ax.set_title('Confusion Matrix')
    f.colorbar(image)
    ax.set_xlim(-0.5, cm.shape[0] - 0.5)
    ax.set_ylim(-0.5, cm.shape[0] - 0.5)
    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    m = cm.max()
    total = cm.sum()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, '%d\n%.2g%%' % (cm[i, j], cm[i, j] / total * 100), horizontalalignment='center',
                    color='white' if cm[i, j] < m / 2 else 'black')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    f.savefig(filename)

def ae_convert(data):
    r = np.clip((data[:, :, [4, 2, 1]]  * np.float32(100.0)), 0.0, 255.0).astype(np.uint8)
    return r

def print_classes(cm):
    for i in range(cm.shape[0]):
        name = config.dataset.classes[i].name if \
               len(config.dataset.classes) == cm.shape[0] else ('Class %d' % (i))
        with np.errstate(invalid='ignore'):
            print('%s--- Precision: %6.2f%%    Recall: %6.2f%%        Pixels: %d / %d' % \
                    (name.ljust(20),
                     np.nan_to_num(cm[i,i] / np.sum(cm[:, i]) * 100),
                     np.nan_to_num(cm[i,i] / np.sum(cm[i, :]) * 100),
                     int(np.sum(cm[i, :])), int(np.sum(cm))))
    print('%6.2f%% Correct' % (float(np.sum(np.diag(cm)) / np.sum(cm) * 100)))

def classify_image(model, image, label, path, net_name, options):
    out_path, base_name = os.path.split(path)
    base_name = os.path.splitext(base_name)[0]
    base_out = (options.outprefix if options.outprefix else net_name + '_') + base_name + '.tiff'


    # check if is subdirectory
    if options.basedir and os.path.abspath(options.basedir) == \
            os.path.commonpath([os.path.abspath(options.basedir), os.path.abspath(out_path)]):
        out_path = os.path.relpath(out_path, options.basedir)
    else:
        out_path = ''
    if options.outdir:
        out_path = os.path.join(options.outdir, out_path)
    if out_path:
        os.makedirs(out_path, exist_ok=True)

    writer = image_writer('tiff')

    # DONE: need to implement prob_error_image writer
    error_image = None
    continuous_error_image = None
    continuous_abs_error_image = None
    if label:
        assert image.size() == label.size(), 'Image and label do not match.'
        if options.errors:
            error_image = writer(os.path.join(out_path, 'errors_' + base_out))
        if options.continuous_error:
            continuous_error_image = writer(os.path.join(out_path, 'continuous_errors_' + base_out))
        if options.continuous_abs_error:
            continuous_abs_error_image = writer(os.path.join(out_path, 'continuous_abs_errors_' + base_out))


    prob_image = writer(os.path.join(out_path, base_out)) if options.prob else None
    output_image = writer(os.path.join(out_path, base_out)) if not options.prob else None


    ts = config.io.tile_size()
    if options.autoencoder:
        label = image
        predictor = predict.ImagePredictor(model, ts, output_image, True, base_name,
                                           None if options.noColormap else (ae_convert, np.uint8, 3))
    else:
        colors = list(map(lambda x: x.color, config.dataset.classes))
        error_colors = np.array([[0x0, 0x0, 0x0],
                                 [0xFF, 0x00, 0x00]], dtype=np.uint8)
        if options.noColormap:
            colors=None # Forces raw one channel output
        # DONE: need to add prob_error_image
        predictor = predict.LabelPredictor(model, ts, output_image, True, base_name, colormap=colors,
                                           prob_image=prob_image, error_image=error_image,
                                           error_colors=error_colors, continuous_error_image=continuous_error_image, continuous_abs_error_image=continuous_abs_error_image)

    overlap = (options.overlap, options.overlap)
    try:
        if config.general.gpus() == 0:
            with tf.device('/cpu:0'):
                predictor.predict(image, label, overlap=overlap)
        else:
            predictor.predict(image, label, overlap=overlap)
    except KeyboardInterrupt:
        print('\nAborted.')
        sys.exit(0)

    #if options.autoencoder:
    #    write_tiff('orig_' + net_name + '_' + base_name + '.tiff',
    #               image.read() if options.noColormap else ae_convert(image.read()),
    #               metadata=image.metadata())

    if label:
        cm = predictor.confusion_matrix()
        class_names = list(map(lambda x: x.name, config.dataset.classes))
        print_classes(cm)
        if len(config.dataset.classes) != cm.shape[0]:
            class_names = list(map(lambda x: 'Class %d' % (x), range(cm.shape[0])))
        if options.confusion:
            save_confusion(cm, class_names,
                           os.path.join(out_path, 'confusion_' + os.path.splitext(base_out)[0] + '.pdf'))
        return cm
    return None

def main(options):

    model = load_model(options.model)

    start_time = time.time()
    images = config.dataset.images()
    labels = config.dataset.labels()
    net_name = os.path.splitext(os.path.basename(options.model))[0]

    if len(images) == 0:
        print('No images specified.')
        return 0

    full_cm = None
    if options.autoencoder:
        labels = None
    for (i, path) in enumerate(images):
        cm = classify_image(model, images.load(i), labels.load(i) if labels else None, path, net_name, options)
        if cm is not None:
            if full_cm is None:
                full_cm = np.copy(cm).astype(np.int64)
            else:
                full_cm += cm
    stop_time = time.time()
    if labels:
        print('Overall:')
        print_classes(full_cm)
    print('Elapsed time = ', stop_time - start_time)
    return 0
