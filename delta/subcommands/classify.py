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
import math
import csv

import shapely
from shapely import wkt
from osgeo import gdal

import numpy as np
import matplotlib
from packaging import version
import tensorflow as tf
import tensorflow.keras.metrics #pylint: disable=no-name-in-module

from delta.config import config
from delta.config.extensions import image_writer
from delta.imagery.rectangle import Rectangle
from delta.ml import predict
from delta.ml.io import load_model
from delta.ml.config_parser import metric_from_dict


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

def print_classes(output_file, cm, metrics, comment):

    if output_file is not None:
        file_handle = open(output_file, 'a')

    print(comment)
    if output_file is not None:
        file_handle.write(comment + '\n')
    for i in range(cm.shape[0]):
        name = config.dataset.classes[i].name if \
               len(config.dataset.classes) == cm.shape[0] else ('Class %d' % (i))
        with np.errstate(invalid='ignore'):
            precision_percent = np.nan_to_num(cm[i,i] / np.sum(cm[:, i]) * 100) # Column = predictions
            recall_percent = np.nan_to_num(cm[i,i] / np.sum(cm[i, :]) * 100) # Row = actual values
            s = ('%s--- Precision: %6.2f%%    Recall: %6.2f%%        Frequency: %6.2f%%' %
                 (name.ljust(20), precision_percent, recall_percent,
                  float(np.sum(cm[i, :] / np.sum(cm)) * 100)))
            print(s)
            if output_file is not None:
                file_handle.write(s + '\n')
    s = '%6.2f%% Accuracy' % (float(np.sum(np.diag(cm)) / np.sum(cm) * 100))
    print(s)
    if output_file is not None:
        file_handle.write(s + '\n')

    s = ''
    for m in metrics:
        s += '%s = %8.4f ' % (m.name, float(m.result()))
    print(s)
    if output_file is not None:
        file_handle.write(s + '\n')
        file_handle.close()


class LossToMetricWrapper(tensorflow.keras.metrics.Metric):
    """Wrap a Loss object to make it behave like a Metric object"""
    def __init__(self, loss_object):
        super().__init__(name=loss_object.name)
        self._loss_object = loss_object
        self._moving_average = 0.0
        self._total_count = 0

    def update_state(self, y_true, y_pred, sample_weight=None): #pylint: disable=unused-argument, arguments-differ
        this_loss = self._loss_object.call(y_true, y_pred)
        if isinstance(this_loss, tf.Tensor):
            this_loss = this_loss.numpy()
        elif not isinstance(this_loss, np.ndarray):
            this_loss = np.ndarray(this_loss)
        self._total_count += y_true.size
        self._moving_average += (this_loss.mean() - self._moving_average) * (y_true.size / self._total_count)

    def result(self):
        return self._moving_average

def get_metrics():
    """Returns a list of specified metrics, wrapping up losses as metrics"""
    if config.classify.metrics() is None:
        return []
    metrics = [metric_from_dict(m) for m in config.classify.metrics()]
    metrics = [LossToMetricWrapper(m) if issubclass(type(m), tf.keras.losses.Loss) else m for m in metrics]
    return metrics

def classify_image(model, image, label, path, net_name, options,
                   shapes=None, persistent_metrics=None):
    '''Classify an image and return the confusion matrix and metrics if labels were provided'''
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

    error_image = None
    if label:
        assert image.size() == label.size(), 'Image and label do not match.'
        if options.error_abs:
            error_image = writer(os.path.join(out_path, 'error_abs_' + base_out))
        elif options.error:
            error_image = writer(os.path.join(out_path, 'error_' + base_out))

    prob_image = writer(os.path.join(out_path, base_out)) if config.classify.prob_image() else None
    output_image = writer(os.path.join(out_path, base_out)) if not config.classify.prob_image() else None

    ts = config.io.tile_size()
    roi = get_roi_containing_shapes(shapes)

    metrics = None
    num_temp_metrics = 0
    if options.autoencoder:
        label = image
        predictor = predict.ImagePredictor(model, ts, output_image, True, base_name,
                                           None if options.noColormap else (ae_convert, np.uint8, 3))
    else:
        colors = list(map(lambda x: x.color, config.dataset.classes))
        if options.noColormap:
            colors=None # Forces raw one channel output
        if label:
            metrics = get_metrics() # For single images
            num_temp_metrics = len(metrics)
            if persistent_metrics is not None: # Persistent metrics over all images
                metrics = metrics + persistent_metrics
        predictor = predict.LabelPredictor(model, ts, output_image, True, base_name, colormap=colors,
                                           prob_image=prob_image, error_image=error_image, error_abs=options.error_abs,
                                           metrics=metrics)

    overlap = (config.classify.overlap(), config.classify.overlap())
    try:
        if config.general.gpus() == 0:
            with tf.device('/cpu:0'):
                predictor.predict(image, label, overlap=overlap, input_bounds=roi, roi_shapes=shapes)
        else:
            predictor.predict(image, label, overlap=overlap, input_bounds=roi, roi_shapes=shapes)
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
        if len(config.dataset.classes) != cm.shape[0]:
            class_names = list(map(lambda x: 'Class %d' % (x), range(cm.shape[0])))
        if options.confusion and (not shapes):
            save_confusion(cm, class_names,
                           os.path.join(out_path, 'confusion_' + os.path.splitext(base_out)[0] + '.pdf'))
        metrics = predictor.metrics()
        metrics, persistent_metrics = (metrics[0:num_temp_metrics], metrics[num_temp_metrics:])
        return cm, metrics, persistent_metrics
    return None, None, None

def get_wkt_path(image_path, wkt_folder=None):
    '''Return the path to where the WKT file for an image should be'''
    WKT_EXTENSION = '.wkt.csv'
    if wkt_folder:
        p = os.path.join(wkt_folder, os.path.basename(image_path))
        path = os.path.splitext(p)[0] + WKT_EXTENSION
    else:
        path = os.path.splitext(image_path)[0] + WKT_EXTENSION
    return path

def load_shapes_matching_tag(wkt_path, tag):
    '''Returns a list of all the shapes defined for this tag in the WKT file.
       If tag is None, return untagged regions'''
    shapes = []
    with open(wkt_path, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            raw_line = ', '.join(row)
            if ('POLYGON' not in raw_line) or (len(row) != 2):
                continue
            names = row[1].strip().split(',')
            if not tag and (names == ['']): # Match non-tagged regions
                s = wkt.loads(row[0])
                shapes.append(s)
            else:
                for n in names: # Check each tag of the region
                    if tag == n.strip():
                        s = wkt.loads(row[0])
                        shapes.append(s)
                        continue
    return shapes

# MOVE
def get_roi_containing_shapes(shapes) -> Rectangle:
    '''Return a Rectangle containing all the shapes or None if none were passed in'''
    if not shapes:
        return None
    roi = Rectangle(*shapes[0].bounds)
    for i in range(1,len(shapes)):
        new_roi = Rectangle(*shapes[i].bounds)
        roi.expand_to_contain_rect(new_roi)
    # Convert to integer values
    return Rectangle(int(math.floor(roi.min_x)),
                     int(math.floor(roi.min_y)),
                     int(math.ceil(roi.max_x)),
                     int(math.ceil(roi.max_y)))

def shapes_to_pixel_coordinates(shapes, image_path):
    '''Convert any shapes not in pixel coordinates to pixel coordinates.
       Always returns a list of Polygon objects, even if the input list
       contains MultiPolygon objects.'''
    handle = gdal.Open(image_path)
    transform = handle.GetGeoTransform()

    def apply_transform(coord, transform):
        return ((coord[0] - transform[0]) / transform[1],
                (coord[1] - transform[3]) / transform[5])

    output_shapes = []
    for s in shapes:
        if s.geom_type == 'Polygon':
            coord_list = [apply_transform(c, transform) for c in s.exterior.coords]
            interior_coord_lists = []
            for i in s.interiors:
                these_coords = [apply_transform(c, transform) for c in i.coords]
                interior_coord_lists.append(these_coords)
            output_shapes.append(shapely.geometry.Polygon(coord_list, interior_coord_lists))
            continue
        if s.geom_type == 'MultiPolygon':
            for g in s.geoms:
                coord_list = [apply_transform(c, transform) for c in g.exterior.coords]
                interior_coord_lists = []
                for i in g.interiors:
                    these_coords = [apply_transform(c, transform) for c in i.coords]
                    interior_coord_lists.append(these_coords)
                output_shapes.append(shapely.geometry.Polygon(coord_list, interior_coord_lists))
            continue
        raise Exception('Unrecognized shape type: ' + s.geom_type)
    return output_shapes

def load_wkt_shapes(wkt_path, image_path, region_name):
    '''Loads shapes (in image coordinates) from an image's WKT file'''

    if os.path.isfile(wkt_path):
        geo_shapes = load_shapes_matching_tag(wkt_path, region_name)
        if geo_shapes:
            return shapes_to_pixel_coordinates(geo_shapes, image_path)
    return []

def main(options): #pylint: disable=R0912
    if version.parse(tf.__version__) < version.parse('2.2'): # eager execution not default
        tf.config.experimental_run_functions_eagerly(True)

    model = load_model(options.model)

    start_time = time.time()
    images = config.dataset.images()
    labels = config.dataset.labels()
    net_name = os.path.splitext(os.path.basename(options.model))[0]

    if len(images) == 0:
        print('No images specified.')
        return 0

    result_file = net_name + '_results.txt'
    if result_file and os.path.exists(result_file):
        os.remove(result_file)

    if options.autoencoder or not labels:
        labels = None
        regions = ['all'] # Each whole image
    else:
        regions = ['all', 'no_label'] # Also individual regions without labels
        specified_regions = config.classify.regions()
        if specified_regions:
            regions += specified_regions

    for region_name in regions:
        # If there are multiple images we need to maintain separate metric objects
        # to keep summary statistics over all the images
        full_cm = None
        full_metrics = get_metrics() if len(images) > 1 else None
        for (i, image_path) in enumerate(images):
            this_image = images.load(i)
            wkt_path = get_wkt_path(image_path, config.classify.wkt_dir())
            shapes = None

            if region_name == 'no_label':
                # Individually compute untagged regions
                shapes = load_wkt_shapes(wkt_path, image_path, None)
                for s in shapes:
                    cm, metrics, full_metrics = classify_image(model, this_image, labels.load(i),
                                                               image_path, net_name, options,
                                                               shapes=[s], persistent_metrics=None)
                    print_classes(result_file, cm, metrics, 'For image ' + image_path + ',  shape: ' + str(s))
                continue

            # Load shapes from file if they are specified for this image/region pair
            if region_name != 'all':
                shapes = load_wkt_shapes(wkt_path, image_path, region_name)
                if not shapes:
                    continue # This region name not specified for this image

            cm, metrics, full_metrics = classify_image(model, this_image,
                                                       labels.load(i) if labels else None,
                                                       image_path, net_name, options,
                                                       shapes, persistent_metrics=full_metrics)
            if cm is not None:
                if (region_name == 'all') and (len(images) > 1):
                    print_classes(result_file, cm, metrics, 'Image: %s' % (image_path))
                if full_cm is None:
                    full_cm = np.copy(cm).astype(np.int64)
                else:
                    full_cm += cm
                if len(images) == 1: # So the statistics are printed properly outside the loop
                    full_metrics = metrics
        if labels and (full_cm is not None):
            print_classes(result_file, full_cm, full_metrics, 'Overall:' if region_name == 'all' else region_name + ':')
    stop_time = time.time()
    print('Elapsed time = ', stop_time - start_time)
    return 0
