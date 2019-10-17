import collections
import configparser
import os
import os.path
import pkg_resources
import appdirs

__config_dict = {}

def __recursive_update(d, u):
    """
    Like dict.update, but recursively updates only
    values that have changed in sub-dictionaries.
    """
    for k, v in u.items():
        dv = d.get(k, {})
        if not isinstance(dv, collections.Mapping):
            d[k] = v
        elif isinstance(v, collections.Mapping):
            d[k] = __recursive_update(dv, v)
        else:
            d[k] = v
    return d

def load_config_file(config_path):
    """
    Loads a config file, then updates the default configuration
    with the loaded values.
    """
    global __config_dict #pylint: disable=global-statement
    __config_dict = __recursive_update(__config_dict, __parse_config_file(config_path))

def get_config():
    return __config_dict

def __parse_config_file(config_path):
    """
    Reads a config file on disk and returns it as a dictionary.
    """
    if not os.path.exists(config_path):
        raise Exception('Config file does not exist: ' + config_path)
    config_reader = configparser.ConfigParser()

    try:
        config_reader.read(config_path)
    except IndexError:
        raise Exception('Failed to read config file: ' + config_path)

    # Convert to a dictionary
    config_data = {s:dict(config_reader.items(s)) for s in config_reader.sections()}

    # Make sure all sections are there
    for section, items in config_data.items():
        for name, value in items.items():
            value = os.path.expanduser(os.path.expandvars(value))
            if value.lower() == 'none': # Useful in some cases
                value = None
            else:
                try: # Convert eligible values to integers
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            config_data[section][name] = value

    return config_data

__dirs = appdirs.AppDirs('delta', 'nasa')
DEFAULT_CONFIG_FILES = [pkg_resources.resource_filename('delta', 'config/delta.cfg'),
                        os.path.join(__dirs.site_config_dir, 'delta.cfg'),
                        os.path.join(__dirs.user_config_dir, 'delta.cfg')]

def __load_initial_config():
    # only contains things not in default config file
    global __config_dict #pylint: disable=global-statement
    __config_dict = {'cache' : {'cache_dir' : __dirs.user_cache_dir}}
    load_config_file(DEFAULT_CONFIG_FILES[0])
    for filename in DEFAULT_CONFIG_FILES[1:]:
        if os.path.exists(filename):
            load_config_file(filename)

__load_initial_config()
