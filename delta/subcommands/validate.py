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

from delta.config import config

def check_image(images, labels, i):
    img = images.load(i)
    if labels:
        label = labels.load(i)
        if label.size() != img.size():
            return 'Error: size mismatch for %s and %s.\n' % (images[i], labels[i])
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
