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
Train a neural network.
"""

import sys
import time
import os

#import logging
#logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train
from delta.ml.model_parser import config_model
from delta.ml.layers import ALL_LAYERS

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

def main(options):

    log_folder = config.dataset.log_folder()
    if log_folder:
        if not options.resume: # Start fresh and clear the read logs
            os.system('rm ' + log_folder + '/*')
            print('Dataset progress recording in: ' + log_folder)
        else:
            print('Resuming dataset progress recorded in: ' + log_folder)

    start_time = time.time()
    images = config.dataset.images()
    if not images:
        print('No images specified.', file=sys.stderr)
        return 1
    tc = config.train.spec()
    if options.autoencoder:
        ids = imagery_dataset.AutoencoderDataset(images, config.train.network.chunk_size(),
                                                 tc.chunk_stride, resume_mode=options.resume,
                                                 log_folder=log_folder)
    else:
        labels = config.dataset.labels()
        if not labels:
            print('No labels specified.', file=sys.stderr)
            return 1
        ids = imagery_dataset.ImageryDataset(images, labels, config.train.network.chunk_size(),
                                             config.train.network.output_size(), tc.chunk_stride,
                                             resume_mode=options.resume,
                                             log_folder=log_folder)

    try:
        if options.resume is not None:
            model = tf.keras.models.load_model(options.resume, custom_objects=ALL_LAYERS)
        else:
            model = config_model(ids.num_bands())
        model, _ = train(model, ids, tc)

        if options.model is not None:
            model.save(options.model)
    except KeyboardInterrupt:
        print()
        print('Training cancelled.')

    stop_time = time.time()
    print('Elapsed time = ', stop_time-start_time)
    return 0
