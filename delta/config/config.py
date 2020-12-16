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

import os.path

import yaml
import pkg_resources
import appdirs

def validate_path(path, base_dir):
    if path == 'default':
        return path
    path = os.path.expanduser(path)
    # make relative paths relative to this config file
    if base_dir:
        path = os.path.normpath(os.path.join(base_dir, path))
    return path

def validate_positive(num, _):
    if num <= 0:
        raise ValueError('%d is not positive' % (num))
    return num

def validate_non_negative(num, _):
    if num < 0:
        raise ValueError('%d is negative' % (num))
    return num

class _NotSpecified: #pylint:disable=too-few-public-methods
    pass

class DeltaConfigComponent:
    """
    DELTA configuration component.

    Handles one subsection of a config file. Generally subclasses
    will want to register fields and components in the constructor,
    and possibly override setup_arg_parser and parse_args to handle
    command line options.

    section_header is the title of the section for command line
    arguments in the help.
    """
    def __init__(self, section_header = None):
        """
        Constructs the component.
        """
        self._config_dict = {}
        self._components = {}
        self._fields = []
        self._validate = {}
        self._types = {}
        self._cmd_args = {}
        self._descs = {}
        self._section_header = section_header

    def reset(self):
        """
        Resets all state in the component.
        """
        self._config_dict = {}
        for c in self._components.values():
            c.reset()

    def register_component(self, component, name : str, attr_name = None):
        """
        Register a subcomponent with a name and attribute name (access as self.attr_name)
        """
        assert name not in self._components
        self._components[name] = component
        if attr_name is None:
            attr_name = name
        setattr(self, attr_name, component)

    def register_field(self, name : str, types, accessor = None, validate_fn = None, desc = None):
        """
        Register a field in this component of the configuration.

        types is a single type or a tuple of valid types

        validate_fn (optional) should take two strings as input, the field's value and
        the base directory, and return what to save to the config dictionary.
        It should raise an exception if the field is invalid.
        accessor is an optional name to create an accessor function with
        """
        self._fields.append(name)
        self._validate[name] = validate_fn
        self._types[name] = types
        self._descs[name] = desc
        if accessor:
            def access(self) -> types:
                return self._config_dict[name]#pylint:disable=protected-access
            access.__name__ = accessor
            access.__doc__ = desc
            setattr(self.__class__, accessor, access)

    def register_arg(self, field, argname, **kwargs):
        """
        Registers a command line argument in this component.

        field is the (registered) field this argument modifies.
        argname is the name of the flag on the command line (i.e., '--flag')
        **kwargs are arguments to ArgumentParser.add_argument.

        If help and type are not specified, will use the ones for the field.
        If default is not specified, will use the value from the config files.
        """
        assert field in self._fields, 'Field %s not registered.' % (field)
        if 'help' not in kwargs:
            kwargs['help'] = self._descs[field]
        if 'type' not in kwargs:
            kwargs['type'] = self._types[field]
        elif kwargs['type'] is None:
            del kwargs['type']
        if 'default' not in kwargs:
            kwargs['default'] = _NotSpecified
        self._cmd_args[argname] = (field, kwargs)

    def to_dict(self) -> dict:
        """
        Returns a dictionary representing the config object.
        """
        if isinstance(self._config_dict, dict):
            exp = self._config_dict.copy()
            for (name, c) in self._components.items():
                exp[name] = c.to_dict()
            return exp
        return self._config_dict

    def export(self) -> str:
        """
        Returns a YAML string of all configuration options, from to_dict.
        """
        return yaml.dump(self.to_dict())

    def _set_field(self, name : str, value : str, base_dir : str):
        if name not in self._fields:
            raise ValueError('Unexpected field %s in config file.' % (name))
        if value is not None and not isinstance(value, self._types[name]):
            raise TypeError('%s must be of type %s, is %s.' % (name, self._types[name], value))
        if self._validate[name] and value is not None:
            try:
                value = self._validate[name](value, base_dir)
            except:
                print('Value %s for %s is invalid.' % (value, name))
                raise
        self._config_dict[name] = value

    def _load_dict(self, d : dict, base_dir):
        """
        Loads the dictionary d, assuming it came from the given base_dir (for relative paths).
        """
        if not d:
            return
        for (k, v) in d.items():
            if k in self._components:
                self._components[k]._load_dict(v, base_dir) #pylint:disable=protected-access
            else:
                self._set_field(k, v, base_dir)

    def setup_arg_parser(self, parser, components = None) -> None:
        """
        Adds arguments to the parser. Must overridden by child classes.
        """
        if self._section_header is not None:
            parser = parser.add_argument_group(self._section_header)
        for (arg, value) in self._cmd_args.items():
            (field, kwargs) = value
            parser.add_argument(arg, dest=field, **kwargs)

        for (name, c) in self._components.items():
            if components is None or name in components:
                c.setup_arg_parser(parser)

    def parse_args(self, options):
        """
        Parse options extracted from an ArgParser configured with
        `setup_arg_parser` and override the appropriate
        configuration values.
        """
        d = {}
        for (field, _) in self._cmd_args.values():
            if not hasattr(options, field) or getattr(options, field) is None:
                continue
            if getattr(options, field) is _NotSpecified:
                continue
            d[field] = getattr(options, field)
        self._load_dict(d, None)

        for c in self._components.values():
            c.parse_args(options)

class DeltaConfig(DeltaConfigComponent):
    """
    DELTA configuration manager.

    Access and control all configuration parameters.
    """
    def load(self, yaml_file: str = None, yaml_str: str = None):
        """
        Loads a config file, then updates the default configuration
        with the loaded values.
        """
        base_path = None
        if yaml_file:
            #print("Loading config file: " + yaml_file)
            if not os.path.exists(yaml_file):
                raise Exception('Config file does not exist: ' + yaml_file)
            with open(yaml_file, 'r') as f:
                config_data = yaml.safe_load(f)
            base_path = os.path.normpath(os.path.dirname(yaml_file))
        else:
            config_data = yaml.safe_load(yaml_str)
        self._load_dict(config_data, base_path)

    def setup_arg_parser(self, parser, components=None) -> None:
        parser.add_argument('--config', dest='config', action='append', required=False, default=[],
                            help='Load configuration file (can pass multiple times).')
        super().setup_arg_parser(parser, components)

    def parse_args(self, options):
        for c in options.config:
            self.load(c)
        super().parse_args(options)

    def reset(self):
        super().reset()
        self.load(pkg_resources.resource_filename('delta', 'config/delta.yaml'))

    def initialize(self, options, config_files = None):
        """
        Loads the default files unless config_files is specified, in which case it
        loads them. Then loads options (from argparse).
        """
        self.reset()

        if config_files is None:
            dirs = appdirs.AppDirs('delta', 'nasa')
            config_files = [os.path.join(dirs.site_config_dir, 'delta.yaml'),
                            os.path.join(dirs.user_config_dir, 'delta.yaml')]

        for filename in config_files:
            if os.path.exists(filename):
                config.load(filename)

        if options is not None:
            config.parse_args(options)

config = DeltaConfig()
