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

_config_initialized = False
def register_all():
    global _config_initialized #pylint: disable=global-statement
    # needed to call twice when testing subcommands and when not
    if _config_initialized:
        return
    config.register_component(DeltaConfigComponent('General'), 'general')
    config.general.register_field('verbose', bool, 'verbose', None,
                                  'Print debugging information.')
    delta.imagery.imagery_config.register()
    delta.ml.ml_config.register()
    _config_initialized = True
