"""
Class to cap the number of unpacked images kept on disk at a fixed level.
"""
import os


#============================================================================
# Classes

class DiskCache:
    """Class for caching folders and files on disk with limits on how much is kept.
       It is safe to mix different datasets in the cache folder, though all items in
       the folder will count towards the limit.
    """

    def __init__(self, top_folder, limit):

        if limit < 1:
            raise Exception('Illegal limit passed to Disk Cache: ' + str(limit))

        if not os.path.exists(top_folder):
            try:
                os.mkdir(top_folder)
            except:
                raise Exception('Could not create disk cache folder: ' + top_folder)

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


    def _update_items(self):
        """Update the list of all files and folders contained in the folder.
           Currently these items are not ordered.
        """

        self._item_list = []
        for f in os.listdir(self._folder):
            # Skip text files
            # -> It is important that we don't delete the list file if the user puts it here!
            ext = os.path.splitext(f)[1]
            if ext not in ['.csv', 'txt']:
                self._item_list.append(f)
