"""
Class to cap the number of unpacked images kept on disk at a fixed level.
"""
import os

#============================================================================
# Classes

class DiskFolderCache:
    """Class to keep track of the number of allocated folders on disk and delete
       old ones when the list gets too large.  The idea is that users will use a consistent
       naming convention and store large files in the folders.
    """
    def __init__(self, folder, limit):

        if limit < 1:
            raise Exception('Illegal limit passed to DiskFolderCache: ' + str(limit))
        if not os.path.exists(folder):
            raise Exception('Folder passed to DiskFolderCache does not exist: ' + folder)

        self._limit  = limit
        self._folder = folder

        self._subfolder_list = []
        self._update_subfolders()


    def limit(self):
        return self._limit

    def folder(self):
        return self._folder

    def num_cached(self):
        return len(self._subfolder_list)

    def get_cache_folder(self, name):
        """Make a new folder for this item and keep track of it.
           Returns the full path to the folder for the user to put data in.
        """

        # If we already have the name just move it to the back of the list
        try:
            self._subfolder_list.remove(name)
            self._subfolder_list.append(name)
            return self._full_path(name)
        except ValueError:
            pass

        # Record the new name and delete the oldest folder if our list got too large
        self._subfolder_list.append(name)
        if self.num_cached() > self._limit:
            old_name = self._subfolder_list.pop(0)
            old_path = self._full_path(old_name)
            os.system('rm -rf ' + old_path) # Delete the entire old folder

        # Return the full path to the new folder location
        return self._full_path(name)

    def _full_path(self, name):
        # Get the full path to one of the stored items by name
        return os.path.join(self._folder, name)

    def _update_subfolders(self):
        """Update the list of all folders contained in the top level folder.
           Currently these folders are not ordered.
        """

        self._subfolder_list = []
        for f in os.listdir(self._folder):
            if os.path.isdir(f):
                self._subfolder_list.append(f)
