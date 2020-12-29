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

#pylint:disable=redefined-outer-name
"""
Test for worldview class.
"""
import pytest

from delta.extensions.sources import landsat, worldview

@pytest.fixture(scope="function")
def wv_image(worldview_filenames):
    return worldview.WorldviewImage(worldview_filenames[0])

@pytest.fixture(scope="function")
def landsat_image(landsat_filenames):
    return landsat.LandsatImage(landsat_filenames[0], bands=[1])

# very basic, doesn't actually look at content
def test_wv_image(wv_image):
    buf = wv_image.read()
    assert buf.shape == (64, 32, 1)
    assert buf[0, 0, 0] == 0.0

    assert wv_image.meta_path() is not None
    assert len(wv_image.scale()) == 1
    assert len(wv_image.bandwidth()) == 1

def test_landsat_image(landsat_image):
    buf = landsat_image.read()
    assert buf.shape == (64, 32, 1)
    assert buf[0, 0, 0] == 0.0

    assert landsat_image.radiance_mult()[0] == 2.0
    assert landsat_image.radiance_add()[0] == 2.0
    assert landsat_image.reflectance_mult()[0] == 2.0
    assert landsat_image.reflectance_add()[0] == 2.0
    assert landsat_image.k1_constant()[0] == 2.0
    assert landsat_image.k2_constant()[0] == 2.0
    assert landsat_image.sun_elevation() == 5.8

    landsat.toa_preprocess(landsat_image, calc_reflectance=True)
    landsat_image.read()
    assert buf.shape == (64, 32, 1)
    assert buf[0, 0, 0] == 0.0

    landsat.toa_preprocess(landsat_image)
    landsat_image.read()
    assert buf.shape == (64, 32, 1)
    assert buf[0, 0, 0] == 0.0
