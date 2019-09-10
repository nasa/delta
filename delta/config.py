import configparser
import os

from delta.imagery import disk_folder_cache


def parse_config_file(config_path, data_directory=None, image_type=None,
                      no_required=False):
    """Reads a config file on disk and substitutes in default values.
       Alternately, data_directory and image_type can be specified and the rest of
       the config data will be set to defaults.
       If no_required is set, no items will be treated as mandatory.

       Returns a dictionary containing all possible config values filled with either
       the values in the file or the default values.  The exception are the
       "input_dataset:extension" and "input_dataset:label_directory" fields which are
       populated with "None" instead of a default value.
       The values are generally not checked for correctness.
    """

    REQUIRED_ENTRIES = ['data_directory'] # Everything else has a default value

    # Specify all of variables that we accept and their default values
    # - Some input_dataset values are not handled in this list because they are more complicated.
    DEFAULT_CONFIG_VALUES = {'input_dataset':{'extension':None,
                                              'image_type':'tfrecord',
                                              'label_directory':None,
                                              'num_input_threads':1,
                                              'shuffle_buffer_size':2000,
                                              'num_regions':None # Default is in the image type classes
                                             },
                             'cache':{'cache_dir':disk_folder_cache.DEFAULT_CACHE_DIR,
                                      'cache_limit':disk_folder_cache.DEFAULT_CACHE_LIMIT
                                     },
                             'ml':{'chunk_size':17,
                                   'chunk_overlap':5,
                                   'num_epochs':5,
                                   'batch_size':2,
                                   'model_folder':None
                                  }
                            }

    if not config_path:
        if (not data_directory) and (not image_type):
            raise Exception('Error: Either a configuration file or a data directory '
                            'and image type must be specified!')

        # Manually generate a config object
        config_data = {'input_dataset':{'data_directory':data_directory,
                                        'image_type':image_type}
                      }

    else: # Config file was provided

        if not os.path.exists(config_path):
            raise Exception('Config file does not exist: ' + config_path)
        config_reader = configparser.ConfigParser()

        # Read in the config file and check for the only required section
        try:
            config_reader.read(config_path)
        except IndexError:
            raise Exception('Failed to read config file: ' + config_path)

        # Convert to a dictionary
        config_data = {s:dict(config_reader.items(s)) for s in config_reader.sections()}

    if not no_required:  # Check for required entries
        for entry in REQUIRED_ENTRIES:
            if entry not in config_data['input_dataset']:
                raise Exception('Missing required value "input_dataset:%s"' % (entry))

    # Make sure all sections are there
    for section, items in DEFAULT_CONFIG_VALUES.items():
        if section not in config_data:
            config_data[section] = {} # Init to empty dictionary
        for name, value in items.items(): # Make sure there is an entry for each item
            if name in config_data[section]:
                input_val = config_data[section][name]
                if input_val.lower() == 'none': # Useful in some cases
                    config_data[section][name] = None
                else:
                    try: # Convert eligible values to integers
                        config_data[section][name] = int(config_data[section][name])
                    except (ValueError, TypeError):
                        pass
            else: # Value not present, use the default value
                print('"%s:%s" value not found in config file, using default value %s'
                      % (section, name, str(value)))
                config_data[section][name] = value

    return config_data
