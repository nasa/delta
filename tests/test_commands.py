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

#pylint: disable=redefined-outer-name

import os
import shutil
import tempfile

import pytest
import tensorflow as tf

from delta.subcommands.main import main

@pytest.fixture(scope="session")
def doubling_config(doubling_tiff_filenames):
    tmpdir = tempfile.mkdtemp()

    config_path = os.path.join(tmpdir, 'dataset.yaml')
    with open(config_path, 'w') as f:
        f.write('dataset:\n')
        f.write('  images:\n')
        f.write('    nodata_value: 10.0\n')
        f.write('    files:\n')
        for fn in doubling_tiff_filenames[0]:
            f.write('      - %s\n' % (fn))
        f.write('  labels:\n')
        f.write('    nodata_value: 10.0\n')
        f.write('    files:\n')
        for fn in doubling_tiff_filenames[1]:
            f.write('      - %s\n' % (fn))

    yield config_path

    shutil.rmtree(tmpdir)

def test_predict(doubling_config, tmp_path):
    model_path = tmp_path / 'model.h5'
    print(open(doubling_config, 'r').read())
    inputs = tf.keras.layers.Input((32, 32, 1))
    out = tf.keras.layers.Add()([inputs, inputs])
    tf.keras.Model(inputs, out).save(model_path)
    #with open(config_path, 'w') as f:
    #    f.write("""train:
    #                 network:
    #                   model:
    #                     layers:
    #                       - Input:
    #                           shape: [~, ~, num_bands]
    #                           name: input_layer
    #                       - Add:
    #                           inputs: [input_layer, input_layer]
    #                 epochs: 1
    #            """)
    args = 'delta classify --config %s %s' % (doubling_config, model_path)
    main(args.split())
