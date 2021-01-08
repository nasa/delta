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
    classes = config.dataset.classes
    for c in config.dataset.classes:
        d[c.end_value] = c.name
    if config.dataset.labels().nodata_value():
        d[len(config.dataset.classes)] = 'nodata'
    return d

def check_image(images, labels, i):
    img = images.load(i)
    if labels:
        label = labels.load(i)
        if label.size() != img.size():
            return 'Error: size mismatch for %s and %s.\n' % (images[i], labels[i])
        v, counts = np.unique(label.read(), return_counts=True)
        total = sum(counts)

        d = get_class_dict()
        values = { k:0 for (k, _) in d.items() }
        for j in range(len(v)):
            values[v[j]] = counts[j]
        s = ''
        for (j, name) in d.items():
            s += '%s: %6.2f%%     ' % (name, values[j] / total * 100)
        print(s + labels[i].split('/')[-1])
        #print('Values for %s: ' % (labels[i]), np.unique(label.read()))
    return ''

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
    errors = ''
    for i in range(len(images)):
        errors += check_image(images, labels, i)
    tc = config.train.spec()
    images = tc.validation.images
    labels = tc.validation.labels
    if images:
        print('Validating %d validation images.' % (len(images)))
        for i in range(len(images)):
            errors += check_image(images, labels, i)
    if not errors:
        errors = 'Validation successful.'
    print(errors, file=sys.stderr)

    return 0
