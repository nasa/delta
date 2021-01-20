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

import pytest
import numpy as np

import conftest

from delta.config import config
from delta.imagery import imagery_dataset, rectangle

def test_basics(dataset_block_label):
    """
    Tests basic methods of a dataset.
    """
    d = dataset_block_label
    assert d.chunk_shape() == (3, 3)
    assert d.input_shape() == (3, 3, 1)
    assert d.output_shape() == (3, 3, 1)
    assert len(d.image_set()) == len(d.label_set())
    assert d.tile_shape() == [256, 1024]
    assert d.tile_overlap() == (0, 0)

def test_block_label(dataset_block_label):
    """
    Tests basic functionality of a dataset on 3x3 blocks.
    """
    num_data = 0
    for image in dataset_block_label.data():
        img = image.numpy()
        assert img.dtype == np.float32
        unique = np.unique(img)
        assert (0 in unique or 1 in unique and len(unique) <= 2)
        num_data += 1
    num_label = 0
    for label in dataset_block_label.labels():
        num_label += 1
    assert num_label == num_data

    ds = dataset_block_label.dataset()
    for (image, label) in ds.take(100):
        if label[1, 1]:
            assert image[0][0][0] == 1
            assert image[0][1][0] == 1
            assert image[0][2][0] == 1
            assert image[1][0][0] == 1
            assert image[1][2][0] == 1
            assert image[2][0][0] == 1
            assert image[2][1][0] == 1
            assert image[2][2][0] == 1
        v1 = image[0][0][0] == 0
        v2 = image[0][1][0] == 0
        v3 = image[0][2][0] == 0
        if v1 or v2 or v3:
            assert label[1, 1] == 0
        v4 = image[1][0][0] == 0
        v5 = image[1][2][0] == 0
        if v4 or v5:
            assert label[1, 1] == 0
        v6 = image[2][0][0] == 0
        v7 = image[2][1][0] == 0
        v8 = image[2][2][0] == 0
        if v6 or v7 or v8:
            assert label[1, 1] == 0

def test_nodata(dataset_block_label):
    """
    Tests that this filters out blocks where labels are all 0.
    """
    dataset_block_label.label_set().set_nodata_value(0)
    try:
        ds = dataset_block_label.dataset()
        for (_, label) in ds.take(100):
            assert np.sum(label) > 0
    finally:
        dataset_block_label.label_set().set_nodata_value(None)

def test_class_weights(dataset_block_label):
    """
    Tests that this filters out blocks where labels are all 0.
    """
    lookup = np.asarray([1.0, 2.0])
    ds = dataset_block_label.dataset(class_weights=[1.0, 2.0])
    for (_, label, weights) in ds.take(100):
        assert np.all(lookup[label.numpy()] == weights)

def test_rectangle():
    """
    Tests the Rectangle class basics.
    """
    r = rectangle.Rectangle(5, 10, 15, 30)
    assert r.min_x == 5
    assert r.min_y == 10
    assert r.max_x == 15
    assert r.max_y == 30
    assert r.bounds() == (5, 15, 10, 30)
    assert r.has_area()
    assert r.get_min_coord() == (5, 10)
    assert r.perimeter() == 60
    assert r.area() == 200
    r.shift(-5, -10)
    assert r.bounds() == (0, 10, 0, 20)
    r.scale_by_constant(2, 1)
    assert r.bounds() == (0, 20, 0, 20)
    r.expand(0, 0, -10, -5)
    assert r.bounds() == (0, 10, 0, 15)
    r.expand_to_contain_pt(14, 14)
    assert r.bounds() == (0, 15, 0, 15)

    r2 = rectangle.Rectangle(-5, -5, 5, 10)
    assert r.get_intersection(r2).bounds() == (0, 5, 0, 10)
    assert not r.contains_rect(r2)
    assert r.overlaps(r2)
    assert not r.contains_pt(-1, -1)
    assert r2.contains_pt(-1, -1)
    r.expand_to_contain_rect(r2)
    assert r.bounds() == (-5, 15, -5, 15)

def test_rectangle_rois():
    """
    Tests make_tile_rois.
    """
    r = rectangle.Rectangle(0, 0, 10, 10)
    tiles = r.make_tile_rois((5, 5), include_partials=False)
    assert len(tiles) == 4
    for t in tiles:
        assert t.width() == 5 and t.height() == 5
    tiles = r.make_tile_rois((5, 10), include_partials=False)
    assert len(tiles) == 2
    tiles = r.make_tile_rois((11, 11), include_partials=False)
    assert len(tiles) == 0
    tiles = r.make_tile_rois((11, 11), include_partials=True)
    assert len(tiles) == 1
    assert tiles[0].bounds() == (0, 10, 0, 10)
    tiles = r.make_tile_rois((20, 20), include_partials=True, min_shape=(11, 11))
    assert len(tiles) == 0
    tiles = r.make_tile_rois((20, 20), include_partials=True, min_shape=(10, 10))
    assert len(tiles) == 1

    tiles = r.make_tile_rois((6, 6), include_partials=False)
    assert len(tiles) == 1
    tiles = r.make_tile_rois((6, 6), include_partials=False, overlap_shape=(2, 2))
    assert len(tiles) == 4
    tiles = r.make_tile_rois((6, 6), include_partials=False, partials_overlap=True)
    assert len(tiles) == 4
    for t in tiles:
        assert t.width() == 6 and t.height() == 6

    tiles = r.make_tile_rois((5, 5), include_partials=False, by_block=True)
    assert len(tiles) == 2
    for row in tiles:
        assert len(row) == 2

@pytest.fixture(scope="function")
def autoencoder(all_sources):
    source = all_sources[0]
    conftest.config_reset()
    (image_path, _) = source[0]
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
                    preprocess: ~''' %
                (os.path.dirname(image_path), source[2], os.path.dirname(image_path), source[1]))

    dataset = imagery_dataset.AutoencoderDataset(config.dataset.images(),
                                                 (3, 3), stride=config.train.spec().stride)
    return dataset

def test_autoencoder(autoencoder):
    """
    Test that the inputs and outputs of the dataset are the same.
    """
    ds = autoencoder.dataset()
    for (image, label) in ds.take(1000):
        assert (image.numpy() == label.numpy()).all()

def test_resume_mode(autoencoder, tmpdir):
    """
    Test imagery dataset's resume functionality.
    """
    try:
        autoencoder.set_resume_mode(True, str(tmpdir))
        autoencoder.reset_access_counts()
        for i in range(len(autoencoder.image_set())):
            autoencoder.resume_log_update(i, count=10000, need_check=True)
            assert autoencoder.resume_log_read(i) == (True, 10000)

        ds = autoencoder.dataset()
        count = 0
        for (_, unused_) in ds.take(100):
            count += 1
        assert count == 0
    finally:
        autoencoder.set_resume_mode(False, None)
