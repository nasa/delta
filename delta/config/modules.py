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
Registers all config modules.
"""

import delta.imagery.imagery_config
import delta.ml.ml_config
from .config import config, DeltaConfigComponent
from .extensions import register_extension

class ExtensionsConfig(DeltaConfigComponent):
    """
    Configuration component for extensions.
    """
    def __init__(self):
        super().__init__()

    # register immediately, don't override
    def _load_dict(self, d : dict, base_dir):
        if not d:
            return
        if isinstance(d, list):
            for ext in d:
                register_extension(ext)
        elif isinstance(d, str):
            register_extension(d)
        else:
            raise ValueError('extensions should be a list or string.')

_config_initialized = False
def register_all():
    """
    Register all default config modules.
    """
    global _config_initialized #pylint: disable=global-statement
    # needed to call twice when testing subcommands and when not
    if _config_initialized:
        return
    config.register_component(DeltaConfigComponent('General'), 'general')
    config.general.register_component(ExtensionsConfig(), 'extensions')
    config.general.register_field('extensions', list, 'extensions', None,
                                  'Python modules to import as extensions.')
    config.general.register_field('verbose', bool, 'verbose', None,
                                  'Print debugging information.')
    config.general.register_arg('verbose', '--verbose', action='store_const',
                                const=True, type=None)
    delta.imagery.imagery_config.register()
    delta.ml.ml_config.register()
    _config_initialized = True
