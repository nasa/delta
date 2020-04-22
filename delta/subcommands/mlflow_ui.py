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
Run the MLFlow UI to visualize the history of training runs.
"""
import os

from delta.config import config

def setup_parser(subparsers):
    sub = subparsers.add_parser('mlflow_ui', help='Launch mlflow user interface to visualize run history.')

    sub.set_defaults(function=main)
    config.setup_arg_parser(sub, general=False, images=False, labels=False)

def main(_):
    os.system('mlflow ui --backend-store-uri %s' % (config.mlflow_uri()))
    return 0
