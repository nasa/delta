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

import numpy as np

from delta.config import config

def get_class_dict():
    d = {}
    for c in config.dataset.classes:
        d[c.end_value] = c.name
    if config.dataset.labels().nodata_value():
        d[len(config.dataset.classes)] = 'nodata'
    return d

def classes_string(classes, values, image_name):
    s = '%-20s   ' % (image_name)
    is_numeric = np.issubdtype(type(values[0]), np.integer)
    if is_numeric:
        total = sum(values.values())
        nodata_class = None
        if config.dataset.labels().nodata_value():
            nodata_class = len(config.dataset.classes)
            total -= values[nodata_class]
    for (j, name) in classes.items():
        if name == 'nodata':
            continue
        if is_numeric:
            v = values[j] if j in values else 0
            s += '%12.2f%% ' % (v / total * 100, )
        else:
            s += '%12s  ' % (values[j], )
    return s

def check_image(images, labels, classes, total_counts, i):
    img = images.load(i)
    if labels:
        label = labels.load(i)
        if label.size() != img.size():
            return 'Error: size mismatch for %s and %s.\n' % (images[i], labels[i])
        v, counts = np.unique(label.read(), return_counts=True)

        values = { k:0 for (k, _) in classes.items() }
        for (j, value) in enumerate(v):
            values[value] = counts[j]
            if value not in total_counts:
                total_counts[value] = 0
            total_counts[value] += counts[j]
        print(classes_string(classes, values, labels[i].split('/')[-1]))
    return ''

def evaluate_images(images, labels):
    errors = ''
    counts = {}
    classes = get_class_dict()
    header = classes_string(classes, classes, 'Image')
    print(header)
    print('-' * len(header))
    for i in range(len(images)):
        errors += check_image(images, labels, classes, counts, i)
    print('-' * len(header))
    print(classes_string(classes, counts, 'Total'))
    print()
    if config.dataset.labels().nodata_value():
        nodata_c = counts[len(config.dataset.classes)]
        total = sum(counts.values())
        print('Nodata is %6.2f%% of the data. Total Pixels: %.2f million.' % \
              (nodata_c / total * 100, (total - nodata_c) / 1000000))
    return errors

def main(_):
    images = config.dataset.images()
    labels = config.dataset.labels()
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
