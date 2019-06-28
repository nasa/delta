import configparser
import os
from pathlib import Path

from delta.imagery import disk_folder_cache

# Set up configuration values, based on the file:
#  ~/.config/delta/delta.ini

__config = configparser.ConfigParser()

# create defaults if no file exists
__config.add_section('delta')
__config['delta']['cache_dir'] = os.path.join(str(Path.home()), '.cache', 'delta')
__config['delta']['cache_limit'] = '4'

__config.read(os.path.join(str(Path.home()), '.config', 'delta', 'delta.ini'))

if not os.path.exists(__config['delta']['cache_dir']):
    os.mkdir(__config['delta']['cache_dir'])

def cache_dir():
    return __config['delta']['cache_dir']

def cache_limit():
    return int(__config['delta']['cache_limit'])

__cache = disk_folder_cache.DiskFolderCache(cache_dir(), cache_limit())

def cache_manager():
    return __cache
