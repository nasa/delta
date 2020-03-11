"""
Configuration via YAML files and command line options.

Access the singleton `delta.config.config` to get configuration
values, specified either in YAML files or on the command line,
and to load additional YAML files.

For a list of all options and their defaults, see
`delta/config/delta.yaml`.
"""

from .config import config
