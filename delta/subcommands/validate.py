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
Check if the input data is valid.
"""

import sys
import os

import numpy as np
from osgeo import gdal

from delta.config import config

def get_image_stats(path):
    '''Return a list of image band statistics like [[min, max, mean, stddev], ...]'''
    tif_handle = gdal.Open(path)
    num_bands = tif_handle.RasterCount

    output = []
    for b in range(0,num_bands):
        band = tif_handle.GetRasterBand(b+1)
        stats = band.GetStatistics(False, True)
        output.append(stats)

    return output


def get_class_dict():
    '''Populate dictionary with class names by index number'''
    d = {}
    for c in config.dataset.classes:
        d[c.end_value] = c.name
    if config.dataset.labels().nodata_value():
        d[len(config.dataset.classes)] = 'nodata'
    return d

def classes_string(classes, values, image_name):
    '''Generate a formatted string out of strings or numbers.
       "classes" must come from get_class_dict()'''
    s = '%-20s   ' % (image_name)
    is_integer = np.issubdtype(type(values[0]), np.integer)
    is_float   = isinstance(values[0], float)
    if is_integer:
        total = sum(values.values())
        nodata_class = None
        if config.dataset.labels().nodata_value():
            nodata_class = len(config.dataset.classes)
            total -= values[nodata_class]
    for (j, name) in classes.items():
        if name == 'nodata':
            continue
        if is_integer:
            v = values[j] if j in values else 0
            s += '%12.2f%% ' % (v / total * 100, )
        else:
            if is_float:
                s += '%12.2f ' % (values[j])
            else:
                s += '%12s  ' % (values[j], )
    return s



def check_image(images, measures, total_counts, i):
    '''Accumulate total_counts and print out image statistics'''

    # Find min, max, mean, std
    stats = get_image_stats(images[i])

    # Accumulate statistics
    if not total_counts:
        for band in stats: #pylint: disable=W0612
            total_counts.append({'min'   : 0.0,
                                 'max'   : 0.0,
                                 'mean'  : 0.0,
                                 'stddev': 0.0})

    for (b, bandstats) in enumerate(stats):
        total_counts[b]['min'   ] += bandstats[0]
        total_counts[b]['max'   ] += bandstats[1]
        total_counts[b]['mean'  ] += bandstats[2]
        total_counts[b]['stddev'] += bandstats[3]
        name = ''
        if b == 0:
            name = os.path.basename(images[i])
        print(classes_string(measures, bandstats, name))

    return ''

def print_image_totals(images, measures, total_counts):
    '''Convert from source image stat totals to averages and print'''
    num_images = len(images)
    num_bands  = len(total_counts)
    for b in range(0,num_bands):
        values = []
        for m in range(0,len(measures)): #pylint: disable=C0200
            values.append(total_counts[b][measures[m]]/num_images)
        name = ''
        if b == 0:
            name = 'Total'
        print(classes_string(measures, values, name))

def check_label(images, labels, classes, total_counts, i):
    '''Accumulate total_counts and print out image statistics'''
    img   = images.load(i)
    label = labels.load(i)
    if label.size() != img.size():
        return 'Error: size mismatch for %s and %s.\n' % (images[i], labels[i])
    # Count number of times each label appears in image
    v, counts = np.unique(label.read(), return_counts=True)

    # Load the label counts into dictionary and accumulate total_counts
    values = { k:0 for (k, _) in classes.items() }
    for (j, value) in enumerate(v):
        values[value] = counts[j]
        if value not in total_counts:
            total_counts[value] = 0
        total_counts[value] += counts[j]
    # Print out display line with percentages
    print(classes_string(classes, values, labels[i].split('/')[-1]))
    return ''

def evaluate_images(images, labels):
    '''Print class statistics for a set of images with matching labels'''
    errors = ''
    classes = get_class_dict()

    # Evaluate labels first
    counts = {}
    header = classes_string(classes, classes, 'Label')
    print(header)
    print('-' * len(header))
    for i in range(len(labels)):
        errors += check_label(images, labels, classes, counts, i)
    print('-' * len(header))
    print(classes_string(classes, counts, 'Total'))
    print()

    if config.dataset.labels().nodata_value():
        nodata_c = counts[len(config.dataset.classes)]
        total = sum(counts.values())
        print('Nodata is %6.2f%% of the data. Total Pixels: %.2f million.' % \
              (nodata_c / total * 100, (total - nodata_c) / 1000000))

    # Now evaluate source images
    counts = []
    print()
    measures = {0:'min', 1:'max', 2:'mean', 3:'stddev'}
    header = classes_string(classes, measures, 'Image')
    print(header)
    print('-' * len(header))
    for i in range(len(images)):
        errors += check_image(images, measures, counts, i)
    print('-' * len(header))
    print_image_totals(images, measures, counts)
    print()


    return errors

def main(_):
    images = config.dataset.images() # Get all image paths based on config values
    labels = config.dataset.labels() # Get all label paths based on config pathn
    if not images:
        print('No images specified.', file=sys.stderr)
        return 1
    if not labels:
        print('No labels specified.', file=sys.stderr)
    else:
        assert len(images) == len(labels)
    print('Validating %d images.' % (len(images)))
    errors = evaluate_images(images, labels)
    tc = config.train.spec()
    if tc.validation.images:
        print('Validating %d validation images.' % (len(tc.validation.images)))
        errors += evaluate_images(tc.validation.images, tc.validation.labels)
    if errors:
        print(errors, file=sys.stderr)
        return -1

    print('Validation successful.')
    return 0
