#!/usr/bin/env python

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

#pylint:disable=redefined-outer-name,wrong-import-position
import os
import random
import shutil
import sys
import tempfile
import warnings
import zipfile

import numpy as np
import pytest

# conftest.py loaded before pytest.ini warning filters apparently
warnings.filterwarnings('ignore', category=DeprecationWarning, module='osgeo')

from delta.config import config
from delta.extensions.sources import tiff

import delta.config.modules
delta.config.modules.register_all()

assert 'tensorflow' not in sys.modules, 'For speed of command line tool, tensorflow should not be imported by config!'

from delta.imagery import imagery_dataset #pylint: disable=wrong-import-position

import tensorflow as tf  #pylint: disable=wrong-import-position, wrong-import-order
tf.get_logger().setLevel('ERROR')

def config_reset():
    """
    Rests the configuration with useful default options for testing.
    """
    config.reset() # don't load any user files
    config.load(yaml_str=
                '''
                mlflow:
                  enabled: false
                ''')

def generate_tile(width=32, height=32, blocks=50):
    """Generate a widthXheightX3 image, with blocks pixels surrounded by ones and the rest zeros in band 0"""
    image = np.zeros((width, height, 1), np.float32)
    label = np.zeros((width, height), np.uint8)
    for _ in range(blocks):
        x = random.randint(2, width - 3)
        y = random.randint(2, height - 3)
        # ensure non-overlapping
        while (image[x + 2, y - 2 : y + 2, 0] > 0.0).any() or (image[x + 2, y - 2 : y + 2, 0] > 0.0).any() or \
              (image[x - 1 : x + 1, y - 2, 0] > 0.0).any() or (image[x - 1 : x + 1, y + 2, 0] > 0.0).any():
            x = random.randint(2, width - 3)
            y = random.randint(2, height - 3)
        image[x + 1, y - 1, 0] = 1.0
        image[x    , y - 1, 0] = 1.0
        image[x - 1, y - 1, 0] = 1.0
        image[x + 1, y    , 0] = 1.0
        image[x - 1, y    , 0] = 1.0
        image[x + 1, y + 1, 0] = 1.0
        image[x    , y + 1, 0] = 1.0
        image[x - 1, y + 1, 0] = 1.0
        label[x, y] = 1
    return (image, label)

@pytest.fixture(scope="session")
def original_file():
    tmpdir = tempfile.mkdtemp()
    image_path = os.path.join(tmpdir, 'image.tiff')
    label_path = os.path.join(tmpdir, 'label.tiff')

    (image, label) = generate_tile(64, 32, 50)
    tiff.write_tiff(image_path, image)
    tiff.write_tiff(label_path, label)
    yield (image_path, label_path)

    shutil.rmtree(tmpdir)

@pytest.fixture(scope="session")
def worldview_filenames(original_file):
    tmpdir = tempfile.mkdtemp()
    image_name = 'WV02N42_939570W073_2520792013040400000000MS00_GU004003002'
    imd_name = '19MAY13164205-M2AS-503204071020_01_P003.IMD'
    zip_path = os.path.join(tmpdir, image_name + '.zip')
    label_path = os.path.join(tmpdir, image_name + '_label.tiff')
    image_dir = os.path.join(tmpdir, 'image')
    image_path = os.path.join(image_dir, image_name + '.tif')
    vendor_dir = os.path.join(image_dir, 'vendor_metadata')
    imd_path = os.path.join(vendor_dir, imd_name)
    os.mkdir(image_dir)
    os.mkdir(vendor_dir)
    # not really a valid file but this is all we need, only one band in image
    with open(imd_path, 'a') as f:
        f.write('absCalFactor = 9.295654e-03\n')
        f.write('effectiveBandwidth = 4.730000e-02\n')

    tiff.TiffImage(original_file[0]).save(image_path)
    tiff.TiffImage(original_file[1]).save(label_path)

    z = zipfile.ZipFile(zip_path, mode='x')
    z.write(image_path, arcname=image_name + '.tif')
    z.write(imd_path, arcname=os.path.join('vendor_metadata', imd_name))
    z.close()

    yield (zip_path, label_path)

    shutil.rmtree(tmpdir)

@pytest.fixture(scope="session")
def landsat_filenames(original_file):
    tmpdir = tempfile.mkdtemp()
    image_name = 'L1_IGNORE_AAABBB_DATE'
    mtl_name = image_name + '_MTL.txt'
    mtl_path = os.path.join(tmpdir, mtl_name)
    zip_path = os.path.join(tmpdir, image_name + '.zip')
    # not really a valid file but this is all we need, only one band in image
    with open(mtl_path, 'a') as f:
        f.write('SPACECRAFT_ID = LANDSAT_1\n')
        f.write('SUN_ELEVATION = 5.8\n')
        f.write('FILE_NAME_BAND_1 = 1.tiff\n')
        f.write('RADIANCE_MULT_BAND_1 = 2.0\n')
        f.write('RADIANCE_ADD_BAND_1 = 2.0\n')
        f.write('REFLECTANCE_MULT_BAND_1 = 2.0\n')
        f.write('REFLECTANCE_ADD_BAND_1 = 2.0\n')
        f.write('K1_CONSTANT_BAND_1 = 2.0\n')
        f.write('K2_CONSTANT_BAND_1 = 2.0\n')

    image_path = os.path.join(tmpdir, '1.tiff')
    tiff.TiffImage(original_file[0]).save(image_path)

    z = zipfile.ZipFile(zip_path, mode='x')
    z.write(image_path, arcname='1.tiff')
    z.write(mtl_path, arcname=mtl_name)
    z.close()

    label_path = os.path.join(tmpdir, image_name + '_label.tiff')
    tiff.TiffImage(original_file[1]).save(label_path)

    yield (zip_path, label_path)

    shutil.rmtree(tmpdir)

NUM_SOURCES = 1
@pytest.fixture(scope="session")
def all_sources(worldview_filenames):
    return [(worldview_filenames, '.zip', 'worldview', '_label.tiff', 'tiff')]

def load_dataset(source, output_size, chunk_size=3, autoencoder=False):
    config_reset()
    (image_path, label_path) = source[0]
    config.load(yaml_str=
                '''
                io:
                  cache:
                    dir: %s
                dataset:
                  images:
                    type: %s
                    directory: %s
                    extension: %s
                    preprocess: ~
                  labels:
                    type: %s
                    directory: %s
                    extension: %s
                    preprocess: ~''' %
                (os.path.dirname(image_path), source[2], os.path.dirname(image_path), source[1],
                 source[4], os.path.dirname(label_path), source[3]))

    if autoencoder:
        return imagery_dataset.AutoencoderDataset(config.dataset.images(), (chunk_size, chunk_size),
                                                  tile_shape=config.io.tile_size(),
                                                  stride=config.train.spec().stride)
    return imagery_dataset.ImageryDataset(config.dataset.images(), config.dataset.labels(),
                                          (output_size, output_size),
                                          (chunk_size, chunk_size), tile_shape=config.io.tile_size(),
                                          stride=config.train.spec().stride)

@pytest.fixture(scope="function", params=range(NUM_SOURCES))
def dataset(all_sources, request):
    source = all_sources[request.param]
    return load_dataset(source, 1)

@pytest.fixture(scope="function", params=range(NUM_SOURCES))
def ae_dataset(all_sources, request):
    source = all_sources[request.param]
    return load_dataset(source, 1, autoencoder=True)

@pytest.fixture(scope="function")
def dataset_block_label(all_sources):
    return load_dataset(all_sources[0], 3)
