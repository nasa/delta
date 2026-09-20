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

"""Tests for the on-disk imagery cache."""

from delta.imagery.disk_folder_cache import DiskCache


def test_evicts_file(tmp_path):
    """The cache supports evicting regular files."""
    cached_file = tmp_path / 'cached.bin'
    cached_file.write_bytes(b'cached')

    cache = DiskCache(str(tmp_path), limit=1)
    cache.register_item('next-item')

    assert not cached_file.exists()


def test_evicts_directory(tmp_path):
    """Directory eviction continues to remove the full directory tree."""
    cached_directory = tmp_path / 'cached'
    cached_directory.mkdir()
    (cached_directory / 'data.bin').write_bytes(b'cached')

    cache = DiskCache(str(tmp_path), limit=1)
    cache.register_item('next-item')

    assert not cached_directory.exists()


def test_ignores_list_files(tmp_path):
    """CSV and text list files are not registered as cached items."""
    (tmp_path / 'images.csv').write_text('image.tiff\n', encoding='utf-8')
    (tmp_path / 'images.txt').write_text('image.tiff\n', encoding='utf-8')

    cache = DiskCache(str(tmp_path), limit=1)

    assert cache.num_cached() == 0
