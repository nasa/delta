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

#pylint:disable=redefined-outer-name, protected-access
"""
Test for worldview class.
"""
import os
import pytest

import numpy as np

from delta.extensions.sources import landsat, worldview

TEST_BUF_HEIGHT = 64
TEST_BUF_WIDTH = 32

@pytest.fixture(scope="function")
def wv_image(worldview_filenames):
    return worldview.WorldviewImage(worldview_filenames[0])

@pytest.fixture(scope="function")
def landsat_image(landsat_filenames):
    return landsat.LandsatImage(landsat_filenames[0], bands=[1])

# very basic, doesn't actually look at content
def test_wv_image(wv_image):
    buf = wv_image.read()
    assert buf.shape == (TEST_BUF_HEIGHT, TEST_BUF_WIDTH, 1)
    assert buf[0, 0, 0] == 0.0

    assert wv_image.meta_path() is not None
    assert len(wv_image.scale()) == 1
    assert len(wv_image.bandwidth()) == 1

    worldview.toa_preprocess(wv_image, calc_reflectance=False)
    buf = wv_image.read()
    assert buf.shape == (TEST_BUF_HEIGHT, TEST_BUF_WIDTH, 1)
    assert buf[0, 0, 0] == 0.0

def test_landsat_image(landsat_image):
    buf = landsat_image.read()
    assert buf.shape == (TEST_BUF_HEIGHT, TEST_BUF_WIDTH, 1)
    assert buf[0, 0, 0] == 0.0

    assert landsat_image.radiance_mult()[0] == 2.0
    assert landsat_image.radiance_add()[0] == 2.0
    assert landsat_image.reflectance_mult()[0] == 2.0
    assert landsat_image.reflectance_add()[0] == 2.0
    assert landsat_image.k1_constant()[0] == 2.0
    assert landsat_image.k2_constant()[0] == 2.0
    assert landsat_image.sun_elevation() == 5.8

    landsat.toa_preprocess(landsat_image, calc_reflectance=True)
    buf = landsat_image.read()
    assert buf.shape == (TEST_BUF_HEIGHT, TEST_BUF_WIDTH, 1)
    assert buf[0, 0, 0] == 0.0

    landsat.toa_preprocess(landsat_image)
    buf = landsat_image.read()
    assert buf.shape == (TEST_BUF_HEIGHT, TEST_BUF_WIDTH, 1)
    assert buf[0, 0, 0] == 0.0

def test_wv_cache(wv_image):
    buf = wv_image.read()
    cached_path = wv_image._paths[0]
    mod_time = os.path.getmtime(cached_path)
    path = wv_image.path()
    new_image = worldview.WorldviewImage(path)
    buf2 = wv_image.read()
    assert np.all(buf == buf2)
    assert new_image._paths[0] == cached_path
    assert os.path.getmtime(cached_path) == mod_time

def test_landsat_cache(landsat_image):
    buf = landsat_image.read()
    cached_path = landsat_image._paths[0]
    mod_time = os.path.getmtime(cached_path)
    path = landsat_image.path()
    new_image = landsat.LandsatImage(path)
    buf2 = landsat_image.read()
    assert np.all(buf == buf2)
    assert new_image._paths[0] == cached_path
    assert os.path.getmtime(cached_path) == mod_time
