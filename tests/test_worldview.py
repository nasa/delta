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

from delta.extensions.sources import worldview

@pytest.fixture(scope="function")
def wv_image(worldview_filenames):
    return worldview.WorldviewImage(worldview_filenames[0])

# very basic, doesn't actually look at content
def test_wv_image(wv_image):
    assert wv_image.meta_path() is not None
    buf = wv_image.read()
    assert buf.shape == (64, 32, 1)
    assert len(wv_image.scale()) == 1
    assert len(wv_image.bandwidth()) == 1
