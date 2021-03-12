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
Loading configuration from command line arguments and yaml files.

Most users will want to use the global object `delta.config.config.config`
to access configuration parameters.
"""

import os.path
from typing import Any, Callable, List, Optional, Tuple, Union

import yaml
import pkg_resources
import appdirs

def validate_path(path: str, base_dir: str) -> str:
    """
    Normalizes a path.

    Parameters
    ----------
    path: str
        Input path
    base_dir: str
        The base directory for relative paths.

    Returns
    -------
    str
        The normalized path.
    """
    if path == 'default':
        return path
    path = os.path.expanduser(path)
    # make relative paths relative to this config file
    if base_dir:
        path = os.path.normpath(os.path.join(base_dir, path))
    return path

def validate_positive(num: Union[int, float], _: str) -> Union[int, float]:
    """
    Checks that a number is positive.

    Parameters
    ----------
    num: Union[int, float]
        Input number
    _: str
        Unused base path.

    Raises
    ------
    ValueError
        If number is not positive.
    """
    if num <= 0:
        raise ValueError('%d is not positive' % (num))
    return num

def validate_non_negative(num: Union[int, float], _: str) -> Union[int, float]:
    """
    Checks that a number is not negative.

    Parameters
    ----------
    num: Union[int, float]
        Input number
    _: str
        Unused base path.

    Raises
    ------
    ValueError
        If number is negative.
    """
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
    and possibly override `setup_arg_parser` and `parse_args` to handle
    command line options.
    """
    def __init__(self, section_header: Optional[str] = None):
        """
        Parameters
        ----------
        section_header: Optional[str]
            The title of the section for command line arguments in the help.
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

    def register_component(self, component: 'DeltaConfigComponent', name : str, attr_name: Optional[str] = None):
        """
        Register a subcomponent.

        Parameters
        ----------
        component: DeltaConfigComponent
            The subcomponent to add.
        name: str
            Name of the subcomponent. Must be unique.
        attr_name: Optional[str]
            If specified, can access the component as self.attr_name.
        """
        assert name not in self._components
        self._components[name] = component
        if attr_name is None:
            attr_name = name
        setattr(self, attr_name, component)

    def register_field(self, name: str, types: Union[type, Tuple[type, ...]], accessor: Optional[str] = None,
                       validate_fn: Optional[Callable[[Any, str], Any]] = None, desc = None):
        """
        Register a field in this component of the configuration.

        Parameters
        ----------
        name: str
            Name of the field (must be unique).
        types: type or tuple of types
            Valid type or types for the field.
        accessor: Optional[str]
            If set, defines a function self.accessor() which retrieves the field.
        validate_fn: Optional[Callable[[Any, str], Any]]
            If specified, sets input = validate_fn(input, base_path) before using it, where
            base_path is the current directory. The validate function should raise an error
            if the input is invalid.
        desc: Optional[str]
            A description to use in help messages.
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

    def register_arg(self, field: str, argname: str, options_name: Optional[str] =None, **kwargs):
        """
        Registers a command line argument in this component. Command line arguments override the
        values in the config files when specified.

        Parameters
        ----------
        field: str
            The previously registered field this argument modifies.
        argname: str
            The name of the flag on the command line (i.e., '--flag')
        options_name: Optional[str]
            Name stored in the options object. It defaults to the
            field if not specified. Only needed for duplicates, such as for multiple image
            specifications.
        **kwargs:
            Further arguments are passed to ArgumentParser.add_argument.
            If `help` and `type` are not specified, will use the values from field registration.
            If `default` is not specified, will use the value from the config files.
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
        self._cmd_args[argname] = (field, field if options_name is None else options_name, kwargs)

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
            except Exception as e:
                raise AssertionError('Value %s for %s is invalid.' % (value, name)) from e
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

    def setup_arg_parser(self, parser : 'argparse.ArgumentParser', components: Optional[List[str]] = None) -> None:
        """
        Adds arguments to the parser. May be overridden by child classes.

        Parameters
        ----------
        parser: argparse.ArgumentParser
            The praser to set up arguments with and later pass the command line flags to.
        components: Optional[List[str]]
            If specified, only parse arguments from the given components, specified by name.
        """
        if self._section_header is not None:
            parser = parser.add_argument_group(self._section_header)
        for (arg, value) in self._cmd_args.items():
            (_, options_name, kwargs) = value
            parser.add_argument(arg, dest=options_name, **kwargs)

        for (name, c) in self._components.items():
            if components is None or name in components:
                c.setup_arg_parser(parser)

    def parse_args(self, options: 'argparse.Namespace'):
        """
        Parse options extracted from an `argparse.ArgumentParser` configured with
        `setup_arg_parser` and override the appropriate
        configuration values.

        Parameters
        ----------
        options: argparse.Namespace
            Options returned from a call to parse_args on a parser initialized with
            setup_arg_parser.
        """
        d = {}
        for (field, options_name, _) in self._cmd_args.values():
            if not hasattr(options, options_name) or getattr(options, options_name) is None:
                continue
            if getattr(options, options_name) is _NotSpecified:
                continue
            d[field] = getattr(options, options_name)
        self._load_dict(d, None)

        for c in self._components.values():
            c.parse_args(options)

class DeltaConfig(DeltaConfigComponent):
    """
    DELTA configuration manager. Access and control all configuration parameters.
    """
    def load(self, yaml_file: Optional[str] = None, yaml_str: Optional[str] = None):
        """
        Loads a config file, then updates the default configuration
        with the loaded values.

        Parameters
        ----------
        yaml_file: Optional[str]
            Filename of a yaml file to load.
        yaml_str: Optional[str]
            Load yaml directly from a str. Exactly one of `yaml_file` and `yaml_str`
            must be specified.
        """
        base_path = None
        if yaml_file:
            if not os.path.exists(yaml_file):
                raise FileNotFoundError('Config file does not exist: ' + yaml_file)
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

    def initialize(self, options: 'argparse.Namespace', config_files: Optional[List[str]] = None):
        """
        Loads all config files, then parses all command line arguments.
        Parameters
        ----------
        options: argparse.Namespace
            Command line options from `setup_arg_parser` to parse.
        config_files: Optional[List[str]]
            If specified, loads only the listed files. Otherwise, loads the default config
            files.
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
"""Global config object. Use this to access all configuration."""
