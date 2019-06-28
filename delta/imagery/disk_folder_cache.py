"""
Class to cap the number of unpacked images kept on disk at a fixed level.
"""
import os
from abc import ABC, abstractmethod


#============================================================================
# Classes

class DiskCache:
    """Base class for caching folders and files on disk with limits on how much is kept.
    """

    def __init__(self, top_folder, limit):

        if limit < 1:
            raise Exception('Illegal limit passed to Disk Cache: ' + str(limit))
        if not os.path.exists(top_folder):
            raise Exception('Folder passed to Disk Cache does not exist: ' + top_folder)

        self._limit  = limit
        self._folder = top_folder

        self._item_list = []
        self._update_items()


    def limit(self):
        return self._limit

    def folder(self):
        return self._folder

    def num_cached(self):
        return len(self._item_list)

    def register_item(self, name):
        """Register a new item with the cache manager and return the full path to it"""

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

    @abstractmethod
    def _update_items(self):
        """Update the list of all currently stored items"""
        pass



class DiskFileCache(DiskCache):
    """Class to keep track of the number of allocated files on disk and delete
       old ones when the list gets too large.
    """

    def _update_items(self):
        """Update the list of all files contained in the folder.
           Currently these files are not ordered.
        """

        self._item_list = []
        for f in os.listdir(self._folder):
            # Skip folders and text files
            # -> It is important that we don't delete the list file if the user puts it here!
            if os.path.isdir(f):
                continue
            ext = os.path.splitext(f)[1]
            if ext not in ['.csv', 'txt']:
                self._item_list.append(f)

class DiskFolderCache(DiskCache):
    """Class to keep track of the number of allocated folders on disk and delete
       old ones when the list gets too large.  The idea is that users will use a consistent
       naming convention and store data in the folders.
    """


    def _update_items(self):
        """Update the list of all folders contained in the top level folder.
           Currently these folders are not ordered.
        """

        self._item_list = []
        for f in os.listdir(self._folder):
            if os.path.isdir(f):
                self._item_list.append(f)
