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
Caches large images.
"""
import os

class DiskCache:
    """
    Caches folders and files on disk with limits on how much is kept.
    It is safe to mix different datasets in the cache folder, though all items in
    the folder will count towards the limit.
    """
    def __init__(self, top_folder, limit):
        """
        The top level folder to store cached items in and the number to store
        are specified.
        """
        if limit < 1:
            raise Exception('Illegal limit passed to Disk Cache: ' + str(limit))

        if not os.path.exists(top_folder):
            try:
                os.mkdir(top_folder)
            except Exception as e:
                raise Exception('Could not create disk cache folder: ' + top_folder) from e

        self._limit  = limit
        self._folder = top_folder

        self._item_list = []
        self._update_items()

    def limit(self):
        """
        The number of items to store in the cache.
        """
        return self._limit

    def folder(self):
        """
        The directory to store cached items in.
        """
        return self._folder

    def num_cached(self):
        """
        The number of items currently cached.
        """
        return len(self._item_list)

    def register_item(self, name):
        """
        Register a new item with the cache manager and return the full path to it.
        """

        # If we already have the name just move it to the back of the list
        try:
            self._item_list.remove(name)
            self._item_list.append(name)
            return self._full_path(name)
        except ValueError:
            pass

        # Record the new name and delete the oldest item if our list got too large
        self._item_list.append(name)
        if self.num_cached() > self._limit:
            old_name = self._item_list.pop(0)
            old_path = self._full_path(old_name)
            os.system('rm -rf ' + old_path) # Delete the entire old folder/file

        # Return the full path to the new folder/file location
        return self._full_path(name)

    def _full_path(self, name):
        # Get the full path to one of the stored items by name
        return os.path.join(self._folder, name)


    def _update_items(self):
        """
        Update the list of all files and folders contained in the folder.
        Currently these items are not ordered.
        """

        self._item_list = []
        for f in os.listdir(self._folder):
            # Skip text files
            # -> It is important that we don't delete the list file if the user puts it here!
            ext = os.path.splitext(f)[1]
            if ext not in ['.csv', 'txt']:
                self._item_list.append(f)
