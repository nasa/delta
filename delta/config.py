import configparser
import os
from pathlib import Path

__config = configparser.ConfigParser()

# create defaults if no file exists
__config.add_section('delta')
__config['delta']['cache_dir'] = os.path.join(str(Path.home()), '.cache', 'delta')

__config.read(os.path.join(str(Path.home()), '.config', 'delta', 'delta.ini'))

if not os.path.exists(__config['delta']['cache_dir']):
    os.mkdir(__config['delta']['cache_dir'])

def cache_dir():
    return __config['delta']['cache_dir']
